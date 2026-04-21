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

    # ── context manager ──────────────────────────────────────────

    def __enter__(self) -> "PushToTalkRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── 录音控制 ─────────────────────────────────────────────────

    def start(self) -> None:
        """开始录音。若已在录音中则忽略。"""
        if self._thread and self._thread.is_alive():
            return
        self._frames = []
        self._stop_event.clear()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.config.frames_per_buffer,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

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
        """后台录音线程循环。"""
        assert self._stream is not None
        while not self._stop_event.is_set():
            try:
                data = self._stream.read(
                    self.config.frames_per_buffer, exception_on_overflow=False
                )
            except Exception:
                continue
            self._frames.append(data)
