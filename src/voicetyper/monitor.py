"""轻量级进程资源监控器。

一行代码启动后台线程，每隔固定间隔采集并输出当前进程的 CPU / 内存 /
GPU 显存使用情况。适用于开发调试阶段快速观察资源占用趋势。

基本用法::

    # 作为 context manager（推荐）
    with ResourceMonitor(interval=1.0):
        do_something()

    # 手动控制
    mon = ResourceMonitor(interval=2.0)
    mon.start()
    ...
    mon.stop()

    # 绑定已有的 threading.Event，外部退出时自动停止
    exit_event = threading.Event()
    mon = ResourceMonitor(exit_event=exit_event)
    mon.start()

自定义输出::

    import logging
    mon = ResourceMonitor(sink=logging.info)

    # 或写入文件
    f = open("perf.log", "a")
    mon = ResourceMonitor(sink=lambda msg: f.write(msg + "\\n"))
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

import psutil


def _try_init_nvml():
    """尝试初始化 NVML 并返回 (handle, True)，失败返回 (None, False)。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return handle, True
    except Exception:
        return None, False


def _format_gpu(handle) -> Optional[str]:
    """查询 GPU 显存并格式化，查询失败返回 None。"""
    try:
        import pynvml

        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / (1024 * 1024)
        total = info.total / (1024 * 1024)
        return f"GPU-VRAM: {used:7.1f} / {total:.0f} MB"
    except Exception:
        return None


class ResourceMonitor:
    """进程级资源监控器。

    Parameters
    ----------
    interval:
        采集间隔（秒），默认 ``1.0``。
    prefix:
        每行输出的前缀标签，默认 ``"[Monitor]"``。
    sink:
        输出函数，签名 ``(str) -> None``。默认 ``print``。
        可传入 ``logging.info``、文件 write 等。
    exit_event:
        可选的外部 ``threading.Event``。当该事件被 set 时监控自动停止。
        不传则内部自行创建。
    gpu:
        是否尝试采集 GPU 显存，默认 ``True``。
        设为 ``False`` 可跳过 pynvml 初始化。
    """

    def __init__(
        self,
        interval: float = 1.0,
        prefix: str = "[Monitor]",
        sink: Callable[[str], None] = print,
        exit_event: Optional[threading.Event] = None,
        gpu: bool = True,
    ) -> None:
        self.interval = max(0.1, interval)
        self.prefix = prefix
        self.sink = sink
        self._exit_event = exit_event or threading.Event()
        self._own_event = exit_event is None  # 是否由自己管理 event
        self._gpu = gpu
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())
        self._nvml_handle = None
        self._nvml_ok = False

    # -- public API -----------------------------------------------------------

    def start(self) -> "ResourceMonitor":
        """启动监控线程，返回 self 以便链式调用。"""
        if self._thread is not None and self._thread.is_alive():
            return self

        # CPU percent 首次调用返回 0，先预热
        self._proc.cpu_percent(interval=None)

        if self._gpu:
            self._nvml_handle, self._nvml_ok = _try_init_nvml()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """停止监控线程。"""
        if self._own_event:
            self._exit_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
            self._thread = None

    def snapshot(self) -> dict:
        """立即采集一次快照并以 dict 返回（不打印）。

        Returns
        -------
        dict
            包含 ``cpu_percent``, ``rss_mb``, 以及可选的
            ``gpu_used_mb`` / ``gpu_total_mb``。
        """
        data: dict = {
            "cpu_percent": self._proc.cpu_percent(interval=None),
            "rss_mb": self._proc.memory_info().rss / (1024 * 1024),
        }
        if self._nvml_ok and self._nvml_handle is not None:
            try:
                import pynvml

                info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                data["gpu_used_mb"] = info.used / (1024 * 1024)
                data["gpu_total_mb"] = info.total / (1024 * 1024)
            except Exception:
                pass
        return data

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- internals ------------------------------------------------------------

    def _format(self) -> str:
        cpu = self._proc.cpu_percent(interval=None)
        rss_mb = self._proc.memory_info().rss / (1024 * 1024)

        parts = [
            f"CPU: {cpu:5.1f}%",
            f"RAM: {rss_mb:7.1f} MB",
        ]

        if self._nvml_ok and self._nvml_handle is not None:
            gpu_str = _format_gpu(self._nvml_handle)
            if gpu_str:
                parts.append(gpu_str)

        return f"{self.prefix} {' | '.join(parts)}"

    def _loop(self) -> None:
        while not self._exit_event.is_set():
            try:
                self.sink(self._format())
            except Exception:
                pass
            self._exit_event.wait(self.interval)
