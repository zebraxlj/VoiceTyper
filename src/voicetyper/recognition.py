import logging
import math
import struct
import threading
import time
from enum import Enum
from queue import Queue, Empty
from typing import Optional, Callable, Union

import speech_recognition as sr

logger = logging.getLogger(__name__)


def _rms(data: bytes, sample_width: int) -> int:
    """计算 PCM 音频的 RMS（均方根），替代已废弃的 audioop.rms。"""
    if not data or sample_width <= 0:
        return 0
    # struct 格式：1 字节有符号 / 2 字节小端有符号 / 4 字节小端有符号
    fmt_char_map = {1: "b", 2: "h", 4: "i"}
    fmt_char = fmt_char_map.get(sample_width)
    if fmt_char is None:
        return 0
    n_samples = len(data) // sample_width
    if n_samples == 0:
        return 0
    samples = struct.unpack_from(f"<{n_samples}{fmt_char}", data)
    sum_sq = sum(s * s for s in samples)
    return int(math.sqrt(sum_sq / n_samples))


class AsrEngine(str, Enum):
    SENSEVOICE_SMALL = "sensevoice_small"
    GOOGLE = "google"

    @classmethod
    def from_value(cls, value: Union["AsrEngine", str]) -> "AsrEngine":
        if isinstance(value, cls):
            return value
        v = value.strip().lower()
        if v in {"sensevoice_small", "sensevoice", "local"}:
            return cls.SENSEVOICE_SMALL
        if v in {"google", "google_api", "recognize_google"}:
            return cls.GOOGLE
        raise ValueError(f"Unknown ASR engine: {value}")


class BackgroundSTT:
    """后台语音转文字服务（分段收音 + 可选拼接修正）。

    说明：
    - 音频分段由 ``speech_recognition`` 的静音检测驱动（``pause_threshold`` 等参数会影响切段边界）。
    - 默认使用本地 SenseVoiceSmall（sherpa-onnx）；本地引擎初始化失败时回退到 Google 识别。
    - 支持 context manager（``with`` 语句），退出时自动停止后台线程和监听。
    """

    def __init__(
        self,
        recognizer: sr.Recognizer,
        language: str = "zh-CN",
        phrase_time_limit: int = 30,  # 默认调整为 30 秒，避免长语音被硬切
        stitch_threshold: float = 1.0,  # 两段语音间隔小于此值则尝试拼接
        max_stitch_duration: float = 60.0,  # 最大拼接时长（秒），防止无限拼接
        overlap_ms: int = 200,  # 将上一段末尾 N ms 音频"叠加"到下一段开头，降低边界漏字
        min_rms: int = 150,  # 低于该能量阈值的片段视为静音/底噪，直接跳过识别
        engine: Union[AsrEngine, str] = AsrEngine.SENSEVOICE_SMALL,
        local_model_dir: Optional[str] = None,  # 本地模型目录
        strip_trailing_period: bool = True,  # 是否去除识别结果末尾的句号/句点
    ) -> None:
        """初始化后台语音转文字服务。

        参数：
        - recognizer: ``speech_recognition.Recognizer`` 实例（用于 listen_in_background 与可选 Google 识别）。
        - language: 语言代码（仅 Google 识别使用；SenseVoiceSmall 不依赖该参数）。
        - phrase_time_limit: 单段最大录音时长（秒）。
        - stitch_threshold: 两段间隔小于该阈值时，认为可能是同一句，允许触发拼接重识。
        - max_stitch_duration: 拼接后的最长语音时长（秒），防止无限拼接导致延迟/开销过大。
        - overlap_ms: 仅对 SenseVoiceSmall 生效；把上一段末尾 ``overlap_ms`` 的 PCM 叠加到下一段开头以降低边界漏字。
        - min_rms: 仅对 SenseVoiceSmall 生效；静音门限（RMS），低于阈值的片段视为底噪并跳过识别，降低"没说话也出词"的误触发。
        - engine: 选择识别引擎（``AsrEngine.SENSEVOICE_SMALL`` 或 ``AsrEngine.GOOGLE``）。
        - local_model_dir: 本地模型目录（SenseVoiceSmall 使用；None 时使用默认缓存目录）。
        - strip_trailing_period: 是否自动去除识别结果末尾的句号/句点（默认开启）。
        """
        self.recognizer: sr.Recognizer = recognizer
        self.language: str = language
        self.phrase_time_limit: int = phrase_time_limit
        self.stitch_threshold: float = stitch_threshold
        self.max_stitch_duration: float = max_stitch_duration
        self.overlap_ms: int = overlap_ms
        self.min_rms: int = min_rms
        self.engine: AsrEngine = AsrEngine.from_value(engine)

        # 队列中存储元组：(AudioData, capture_timestamp)
        self.audio_queue: Queue[tuple[sr.AudioData, float]] = Queue()
        self.stop_event: threading.Event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_listening: Optional[Callable[..., None]] = None

        # 拼接相关状态（仅由 worker 线程读写，通过 _stitch_lock 保护以确保 stop() 安全）
        self._stitch_lock: threading.Lock = threading.Lock()
        self._last_audio: Optional[sr.AudioData] = None
        self._last_audio_end_time: float = 0.0
        self._overlap_tail: bytes = b""
        self._overlap_sample_rate: int = 0
        self._overlap_sample_width: int = 0
        self._last_voiced_time: float = 0.0

        # 回调引用（由 start_worker 设置）
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_result: Optional[Callable[[str, bool], None]] = None
        self._on_unintelligible: Optional[Callable[[], None]] = None
        self._on_request_error: Optional[Callable[[Exception], None]] = None

        # 初始化本地引擎
        self.local_engine: Optional[object] = None
        if self.engine == AsrEngine.SENSEVOICE_SMALL:
            try:
                from .models import SenseVoiceSmallEngine

                self.local_engine = SenseVoiceSmallEngine(
                    model_dir=local_model_dir,
                    strip_trailing_period=strip_trailing_period,
                )
            except Exception as e:
                logger.warning("本地引擎初始化失败: %s，回退到 Google API。", e)
                self.engine = AsrEngine.GOOGLE

    # ── context manager ──────────────────────────────────────────

    def __enter__(self) -> "BackgroundSTT":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── 拼接状态管理 ─────────────────────────────────────────────

    def _reset_stitch_state(self) -> None:
        """重置拼接相关状态，避免脏数据污染下一句。"""
        with self._stitch_lock:
            self._last_audio = None
            self._overlap_tail = b""
            self._overlap_sample_rate = 0
            self._overlap_sample_width = 0

    # ── 音频采集回调 ─────────────────────────────────────────────

    def _on_phrase(self, recognizer_inner: sr.Recognizer, audio: sr.AudioData) -> None:
        try:
            now = time.time()
            self.audio_queue.put_nowait((audio, now))
        except Exception:
            pass

    def start_listening(self, mic: sr.Microphone) -> None:
        self.stop_listening = self.recognizer.listen_in_background(
            mic, self._on_phrase, phrase_time_limit=self.phrase_time_limit
        )

    # ── 静音过滤 ─────────────────────────────────────────────────

    def _is_silence(self, raw_data: bytes, sample_width: int) -> bool:
        """判断音频片段是否为静音/底噪（RMS 低于阈值）。

        SenseVoiceSmall 在纯静音/底噪时也可能"幻觉输出"（如单个标点、短英文词），
        这里用 RMS 能量阈值做一个简单 VAD：低能量片段不送入模型。
        """
        return _rms(raw_data, sample_width) < self.min_rms

    # ── 拼接逻辑 ─────────────────────────────────────────────────

    def _try_stitch(
        self, current_audio: sr.AudioData, capture_time: float
    ) -> tuple[Optional[sr.AudioData], bool, Optional[float]]:
        """尝试将当前音频与上一段拼接。

        返回：
            (stitched_audio_or_None, is_stitched, delta)
        """
        with self._stitch_lock:
            if self._last_audio is None:
                return None, False, None

            delta = capture_time - self._last_audio_end_time
            if delta >= self.stitch_threshold:
                return None, False, delta

            # 预判拼接后的时长
            prev_duration = (
                len(self._last_audio.frame_data)
                / self._last_audio.sample_rate
                / self._last_audio.sample_width
            )
            curr_duration = (
                len(current_audio.frame_data)
                / current_audio.sample_rate
                / current_audio.sample_width
            )
            if prev_duration + curr_duration > self.max_stitch_duration:
                return None, False, delta

            # 判定为同一句话被切断且未超时，执行拼接
            raw_data_prev = self._last_audio.get_raw_data()
            raw_data_curr = current_audio.get_raw_data()
            stitched_audio = sr.AudioData(
                raw_data_prev + raw_data_curr,
                current_audio.sample_rate,
                current_audio.sample_width,
            )
            return stitched_audio, True, delta

    # ── 识别 ─────────────────────────────────────────────────────

    def _recognize(
        self,
        audio_to_recognize: sr.AudioData,
        raw_current: bytes,
        sample_rate_current: int,
        sample_width_current: int,
        is_stitched: bool,
        delta: Optional[float],
    ) -> str:
        """对音频执行 ASR 识别，返回识别文本。"""
        if self.engine == AsrEngine.SENSEVOICE_SMALL and self.local_engine is not None:
            raw_base = audio_to_recognize.get_raw_data()
            sample_rate = audio_to_recognize.sample_rate
            sample_width = audio_to_recognize.sample_width

            # 仅在"相邻两段"（delta 较小）且采样参数一致时启用 overlap，避免跨句/跨格式污染。
            with self._stitch_lock:
                use_overlap = (
                    not is_stitched
                    and self.overlap_ms > 0
                    and delta is not None
                    and delta < self.stitch_threshold
                    and self._overlap_tail
                    and sample_rate_current == self._overlap_sample_rate
                    and sample_width_current == self._overlap_sample_width
                )
                raw_for_asr = (self._overlap_tail + raw_current) if use_overlap else raw_base

            text = self.local_engine.transcribe(raw_for_asr, sample_rate)  # type: ignore[attr-defined]

            self._save_overlap(raw_current, sample_rate, sample_width)
            return text
        else:
            # 使用 Google API
            return self.recognizer.recognize_google(  # type: ignore
                audio_to_recognize, language=self.language
            )

    def _save_overlap(self, raw_current: bytes, sample_rate: int, sample_width: int) -> None:
        """保存当前段的尾部 PCM 数据，用于下一段 overlap。"""
        if self.overlap_ms <= 0:
            return
        bytes_per_second = sample_rate * sample_width
        tail_len = int(bytes_per_second * (self.overlap_ms / 1000.0))
        with self._stitch_lock:
            if tail_len > 0:
                self._overlap_tail = (
                    raw_current[-tail_len:] if tail_len < len(raw_current) else raw_current
                )
                self._overlap_sample_rate = sample_rate
                self._overlap_sample_width = sample_width
            else:
                self._overlap_tail = b""

    # ── Worker 主循环 ────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Worker 线程主循环：从队列取音频 → 过滤 → 拼接 → 识别 → 回调。"""
        while not self.stop_event.is_set():
            try:
                item = self.audio_queue.get(timeout=0.5)
                current_audio, capture_time = item
            except Empty:
                continue

            try:
                self._process_audio(current_audio, capture_time)
            except sr.UnknownValueError:
                if self._on_unintelligible:
                    self._on_unintelligible()
                self._reset_stitch_state()
            except sr.RequestError as e:
                if self._on_request_error:
                    self._on_request_error(e)
                self._reset_stitch_state()
            except Exception as e:
                if self._on_request_error:
                    self._on_request_error(e)
                self._reset_stitch_state()
            finally:
                try:
                    self.audio_queue.task_done()
                except Exception:
                    pass

    def _process_audio(self, current_audio: sr.AudioData, capture_time: float) -> None:
        """处理单段音频：静音过滤 → 拼接 → 识别 → 回调。"""
        raw_current = b""
        sample_rate_current = 0
        sample_width_current = 0

        # 本地引擎：提取 PCM 数据并做静音过滤
        if self.engine == AsrEngine.SENSEVOICE_SMALL and self.local_engine is not None:
            raw_current = current_audio.get_raw_data()
            sample_rate_current = current_audio.sample_rate
            sample_width_current = current_audio.sample_width
            if self._is_silence(raw_current, sample_width_current):
                with self._stitch_lock:
                    self._last_audio = None
                    self._last_audio_end_time = capture_time
                return

        if self._on_status:
            self._on_status("recognizing")

        # 1. 尝试拼接
        stitched_audio, is_stitched, delta = self._try_stitch(current_audio, capture_time)

        # 2. 识别（拼接成功则识别拼接后的音频，否则识别当前段）
        audio_to_recognize = stitched_audio if is_stitched and stitched_audio else current_audio
        text = self._recognize(
            audio_to_recognize, raw_current,
            sample_rate_current, sample_width_current,
            is_stitched, delta,
        )

        # 3. 结果处理与回调
        with self._stitch_lock:
            if is_stitched:
                if self._on_result:
                    self._on_result(text, True)  # True = is_correction
                self._last_audio = stitched_audio
            else:
                if self._on_result:
                    self._on_result(text, False)  # False = new sentence
                self._last_audio = current_audio
            self._last_audio_end_time = capture_time
            self._last_voiced_time = capture_time

    # ── 公共控制接口 ─────────────────────────────────────────────

    def start_worker(
        self,
        on_status: Callable[[str], None],
        on_result: Callable[[str, bool], None],
        on_unintelligible: Callable[[], None],
        on_request_error: Callable[[Exception], None],
    ) -> None:
        """启动 worker 线程，开始消费音频队列。

        参数：
        - on_status: 状态变化回调（如 ``"recognizing"``）。
        - on_result: 识别结果回调 ``(text, is_correction)``。
        - on_unintelligible: 无法识别时的回调。
        - on_request_error: 请求/引擎异常时的回调。
        """
        self._on_status = on_status
        self._on_result = on_result
        self._on_unintelligible = on_unintelligible
        self._on_request_error = on_request_error

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        """停止后台监听和 worker 线程。可安全重复调用。"""
        self.stop_event.set()
        if self.stop_listening:
            try:
                self.stop_listening(wait_for_stop=False)
            except TypeError:
                self.stop_listening()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

