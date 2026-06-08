"""按键说话（Push-to-Talk）录音器。

提供 ``PushToTalkRecorder``：在后台线程中持续采集麦克风 PCM 数据，
调用 ``start()`` 开始录音、``stop()`` 结束录音并返回完整 PCM 字节流。

典型用法::

    with PushToTalkRecorder(device_index=0) as recorder:
        recorder.start()
        ...  # 用户按住按键说话
        pcm_data = recorder.stop()
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)


@dataclass
class RecorderConfig:
    """录音参数配置。"""

    rate: int = 16000
    """采样率（Hz）。"""

    channels: int = 1
    """声道数。"""

    frames_per_buffer: int = 1024
    """每次 read 的帧数。"""

    sample_width: int = 2
    """每个采样的字节数（2 = int16）。"""


class PushToTalkRecorder:
    """按键说话录音器，管理 PyAudio 流的开启/录制/停止。

    支持 context manager（``with`` 语句），退出时自动释放 PortAudio 资源。

    注意：一个实例可以多次 ``start()`` / ``stop()`` 循环使用，
    但 ``close()`` 后不可再使用。
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        config: Optional[RecorderConfig] = None,
    ) -> None:
        self.device_index = device_index
        self.config = config or RecorderConfig()
        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._frames: list[bytes] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._actual_rate: int = self.config.rate
        self._actual_channels: int = self.config.channels
        # 每个 device_index 已知可用的 (rate, channels)。
        # None 值表示"配置中的格式直接可用"。命中缓存可避免重复试错与日志。
        self._format_cache: dict[Optional[int], Optional[tuple[int, int]]] = {}
        # 串行化对 PyAudio 的开流操作：让 probe_format 与 start() 不并发 pa.open。
        self._open_lock = threading.Lock()

    # ── context manager ──────────────────────────────────────────

    def __enter__(self) -> "PushToTalkRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── 录音控制 ─────────────────────────────────────────────────

    def start(self) -> None:
        """开始录音。若已在录音中则忽略。

        若设备不支持配置中的采样率/声道（常见于 24bit/48kHz 蓝牙或 USB 麦克风），
        会自动回退到设备原生格式，``_loop`` 中将原生 PCM 下混并重采样回
        ``config.rate``，因此对外暴露的录音数据始终是配置里的目标格式。
        每个设备的格式协商结果会缓存——通常由 ``probe_format`` 在切设备/启动时
        预先填好，``start()`` 直接命中缓存零失败重试；缓存未命中时也会就地兜底。
        """
        if self._thread and self._thread.is_alive():
            return
        self._frames = []
        self._stop_event.clear()

        with self._open_lock:
            stream, rate, channels = self._open_with_cache(self.device_index)
        self._stream = stream
        self._actual_rate = rate
        self._actual_channels = channels
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open_with_cache(
        self, device_index: Optional[int]
    ) -> tuple[pyaudio.Stream, int, int]:
        """按 ``_format_cache`` 决策打开输入流。调用方需持有 ``_open_lock``。

        返回：(stream, rate, channels)。缓存未命中时会原地协商并写入缓存。
        """
        target_rate = self.config.rate
        target_channels = self.config.channels
        cached = self._format_cache.get(device_index, "miss")
        logger.debug(
            "打开输入流：device=%s cache=%s target=%dHz/%dch",
            device_index, cached, target_rate, target_channels,
        )

        if cached is None:
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=target_channels,
                rate=target_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.frames_per_buffer,
            )
            logger.debug(
                "已开流（cache hit, native）：%dHz/%dch", target_rate, target_channels
            )
            return stream, target_rate, target_channels
        if isinstance(cached, tuple):
            rate, channels = cached
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.frames_per_buffer,
            )
            logger.debug(
                "已开流（cache hit, fallback）：%dHz/%dch -> 重采样到 %dHz",
                rate, channels, target_rate,
            )
            return stream, rate, channels

        # cache miss：试目标格式，失败就回退到设备原生格式
        try:
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=target_channels,
                rate=target_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.frames_per_buffer,
            )
            self._format_cache[device_index] = None
            logger.debug(
                "已开流（cache miss, native ok）：%dHz/%dch", target_rate, target_channels,
            )
            return stream, target_rate, target_channels
        except OSError as exc:
            dev_rate, dev_channels = self._device_native_format(device_index)
            if dev_rate is None:
                raise
            logger.warning(
                "设备不支持 %dHz/%dch（%s），回退到设备原生 %dHz/%dch 并自动重采样到 %dHz",
                target_rate, target_channels, exc, dev_rate, dev_channels, target_rate,
            )
            stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=dev_channels,
                rate=dev_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.config.frames_per_buffer,
            )
            self._format_cache[device_index] = (dev_rate, dev_channels)
            logger.debug(
                "已开流（cache miss, fallback）：%dHz/%dch", dev_rate, dev_channels,
            )
            return stream, dev_rate, dev_channels

    def probe_format(
        self,
        device_index: Optional[int] = ...,
        *,
        async_: bool = True,
    ) -> Optional[threading.Thread]:
        """预先协商指定设备的输入格式，把结果写入 ``_format_cache``。

        UI 切换设备或软件启动时调用一次，可让首次 ``start()`` 直接命中缓存，
        避免按下 PTT 时再多一次失败 ``open()`` 造成的开头丢音。

        参数：
            device_index: 要探测的设备索引；保持默认会使用当前 ``self.device_index``。
            async_: True 时在后台线程探测并立刻返回 Thread；False 时同步阻塞。

        返回：
            后台线程对象（``async_=True`` 时）；同步模式返回 None。
        """
        idx = self.device_index if device_index is ... else device_index

        def _do_probe() -> None:
            with self._open_lock:
                if idx in self._format_cache:
                    logger.debug("跳过探测：device=%s 已在缓存中", idx)
                    return
                logger.debug("开始探测 device=%s 的输入格式", idx)
                try:
                    stream, rate, channels = self._open_with_cache(idx)
                except Exception as exc:
                    logger.warning("探测设备 %s 输入格式失败: %s", idx, exc)
                    return
                logger.debug(
                    "探测完成：device=%s 实际 %dHz/%dch", idx, rate, channels,
                )
                try:
                    stream.close()
                except Exception:
                    pass

        if not async_:
            _do_probe()
            return None
        thread = threading.Thread(
            target=_do_probe, name=f"probe-format-{idx}", daemon=True
        )
        thread.start()
        return thread

    def _device_native_format(
        self, device_index: Optional[int] = ...
    ) -> tuple[Optional[int], int]:
        """读取指定设备的原生采样率与最大输入通道数。

        参数：
            device_index: 设备索引；保持默认会使用 ``self.device_index``，
                None 表示系统默认输入。

        返回：
            (rate, channels)；若无法获取设备信息则 rate 为 None。
        """
        idx = self.device_index if device_index is ... else device_index
        try:
            if idx is None:
                info = self._pa.get_default_input_device_info()
            else:
                info = self._pa.get_device_info_by_index(idx)
            rate = int(info.get("defaultSampleRate", 0)) or None
            channels = max(1, int(info.get("maxInputChannels", 1)))
            return rate, channels
        except Exception:
            return None, 1

    def stop(self) -> bytes:
        """停止录音并返回已录制的完整 PCM 字节流。

        返回：
            bytes: 拼接后的 int16 PCM 数据。若未录音则返回空 bytes。
        """
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        return b"".join(self._frames)

    def set_device_index(
        self, device_index: Optional[int], *, probe: bool = True
    ) -> None:
        """切换输入设备索引。若正在录音，会先停止当前流再切换。

        参数：
            device_index: 新的输入设备索引；传 None 表示使用系统默认设备。
            probe: 切换后是否在后台预协商一次输入格式，把首次 ``start()`` 的
                开流延迟摊到 UI 等待中（默认开启）。
        """
        if self._thread and self._thread.is_alive():
            self.stop()
        self.device_index = device_index
        if probe:
            self.probe_format(device_index, async_=True)

    def invalidate_format_cache(self, device_index: Optional[int] = ...) -> None:
        """清除已缓存的设备格式协商结果。

        参数：
            device_index: 指定时只清掉该设备；保持默认会清空全部缓存。
                适用于设备热插拔或驱动重置后强制重新协商。
        """
        if device_index is ...:
            self._format_cache.clear()
        else:
            self._format_cache.pop(device_index, None)

    def close(self) -> None:
        """释放 PortAudio 资源。可安全重复调用。"""
        try:
            if self._stream:
                self._stream.close()
                self._stream = None
        except Exception:
            pass
        try:
            self._pa.terminate()
        except Exception:
            pass

    # ── 内部 ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        """后台录音线程循环。

        多声道输入下混为单声道；非目标采样率的输入用线性插值重采样到
        ``config.rate``，避免在识别端再启动一次 resampler。
        """
        import numpy as np

        assert self._stream is not None
        channels = self._actual_channels
        in_rate = self._actual_rate
        out_rate = self.config.rate
        ratio = in_rate / out_rate  # 输入到输出的步长（in samples per out sample）
        # 跨 chunk 的"残余尾样本"，用于无缝拼接重采样
        tail = np.empty(0, dtype=np.float32)
        # 累积的小数相位偏移，下一段插值的起始位置
        phase = 0.0

        if in_rate == out_rate and channels == 1:
            logger.debug("录音循环启动：%dHz/1ch 直通（无下混无重采样）", in_rate)
        elif in_rate == out_rate:
            logger.debug("录音循环启动：%dHz/%dch -> 仅下混到单声道", in_rate, channels)
        else:
            logger.debug(
                "录音循环启动：%dHz/%dch -> %dHz/1ch（ratio=%.4f，下混 + 线性重采样）",
                in_rate, channels, out_rate, ratio,
            )

        in_samples_total = 0
        out_samples_total = 0

        while not self._stop_event.is_set():
            try:
                data = self._stream.read(
                    self.config.frames_per_buffer, exception_on_overflow=False
                )
            except Exception:
                continue

            samples = np.frombuffer(data, dtype=np.int16)
            in_samples_total += samples.size // max(1, channels)
            if channels > 1:
                usable = (samples.size // channels) * channels
                if usable == 0:
                    continue
                frames = samples[:usable].reshape(-1, channels).astype(np.int32)
                mono = frames.mean(axis=1).astype(np.float32)
            else:
                mono = samples.astype(np.float32)

            if in_rate == out_rate:
                self._frames.append(mono.astype(np.int16).tobytes())
                out_samples_total += mono.size
                continue

            # 与上次残留拼接后做线性插值
            block = np.concatenate((tail, mono))
            if block.size < 2:
                tail = block
                continue
            # 线性插值需要 block[i0+1]，因此最大 i0 为 block.size - 2
            # 取 idx[-1] ≤ block.size - 2 以保证整数采样率比下也安全
            n_out = int(np.floor((block.size - 2 - phase) / ratio)) + 1
            if n_out <= 0:
                tail = block
                continue
            idx = phase + np.arange(n_out, dtype=np.float64) * ratio
            i0 = idx.astype(np.int64)
            frac = (idx - i0).astype(np.float32)
            out = block[i0] * (1.0 - frac) + block[i0 + 1] * frac
            np.clip(out, -32768.0, 32767.0, out=out)
            self._frames.append(out.astype(np.int16).tobytes())
            out_samples_total += out.size

            consumed_last = float(idx[-1])
            next_phase = consumed_last + ratio
            if next_phase >= block.size:
                # 下一个采样点落在当前 block 之外，用 phase 偏移记录到下一段
                tail = np.empty(0, dtype=np.float32)
                phase = next_phase - block.size
            else:
                keep_from = int(np.floor(next_phase))
                tail = block[keep_from:].copy()
                phase = next_phase - keep_from

        in_dur = in_samples_total / in_rate if in_rate else 0.0
        out_dur = out_samples_total / out_rate if out_rate else 0.0
        logger.debug(
            "录音循环结束：采集 %d 样本/%.2fs（%dHz），输出 %d 样本/%.2fs（%dHz）",
            in_samples_total, in_dur, in_rate,
            out_samples_total, out_dur, out_rate,
        )
