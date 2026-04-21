import threading
import time
from enum import Enum
from queue import Queue, Empty
from typing import Optional, Callable, Union

import math
import struct

import speech_recognition as sr


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
    def __init__(
        self,
        recognizer: sr.Recognizer,
        language: str = "zh-CN",
        phrase_time_limit: int = 30,  # 默认调整为 30 秒，避免长语音被硬切
        stitch_threshold: float = 1.0,  # 两段语音间隔小于此值则尝试拼接
        max_stitch_duration: float = 60.0,  # 最大拼接时长（秒），防止无限拼接
        overlap_ms: int = 200,  # 将上一段末尾 N ms 音频“叠加”到下一段开头，降低边界漏字
        min_rms: int = 150,  # 低于该能量阈值的片段视为静音/底噪，直接跳过识别
        engine: Union[AsrEngine, str] = AsrEngine.SENSEVOICE_SMALL,
        local_model_dir: Optional[str] = None,  # 本地模型目录
        strip_trailing_period: bool = True,  # 是否去除识别结果末尾的句号/句点
    ) -> None:
        """
        后台语音转文字服务（分段收音 + 可选拼接修正）。

        说明：
        - 音频分段由 `speech_recognition` 的静音检测驱动（`pause_threshold` 等参数会影响切段边界）。
        - 默认使用本地 SenseVoiceSmall（sherpa-onnx）；本地引擎初始化失败时回退到 Google 识别。

        参数：
        - recognizer: `speech_recognition.Recognizer` 实例（用于 listen_in_background 与可选 Google 识别）。
        - language: 语言代码（仅 Google 识别使用；SenseVoiceSmall 不依赖该参数）。
        - phrase_time_limit: 单段最大录音时长（秒）。
        - stitch_threshold: 两段间隔小于该阈值时，认为可能是同一句，允许触发拼接重识。
        - max_stitch_duration: 拼接后的最长语音时长（秒），防止无限拼接导致延迟/开销过大。
        - overlap_ms: 仅对 SenseVoiceSmall 生效；把上一段末尾 `overlap_ms` 的 PCM 叠加到下一段开头以降低边界漏字。
        - min_rms: 仅对 SenseVoiceSmall 生效；静音门限（RMS），低于阈值的片段视为底噪并跳过识别，降低“没说话也出词”的误触发。
        - engine: 选择识别引擎（`AsrEngine.SENSEVOICE_SMALL` 或 `AsrEngine.GOOGLE`）。
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

        # 拼接相关状态
        self._last_audio: Optional[sr.AudioData] = None
        self._last_audio_end_time: float = 0.0
        self._overlap_tail: bytes = b""
        self._overlap_sample_rate: int = 0
        self._overlap_sample_width: int = 0
        self._last_voiced_time: float = 0.0

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
                print(f"Warning: Failed to initialize local engine: {e}")
                print("Falling back to Google API.")
                self.engine = AsrEngine.GOOGLE

    def _on_phrase(self, recognizer_inner: sr.Recognizer, audio: sr.AudioData) -> None:
        try:
            # 记录收到音频的时间戳
            now = time.time()
            self.audio_queue.put_nowait((audio, now))
        except Exception:
            pass

    def start_listening(self, mic: sr.Microphone) -> None:
        self.stop_listening = self.recognizer.listen_in_background(
            mic, self._on_phrase, phrase_time_limit=self.phrase_time_limit
        )

    def start_worker(
        self,
        on_status: Callable[[str], None],
        on_result: Callable[[str, bool], None],  # changed: (text, is_correction)
        on_unintelligible: Callable[[], None],
        on_request_error: Callable[[Exception], None],
    ) -> None:
        def worker() -> None:
            while not self.stop_event.is_set():
                try:
                    # 获取新音频
                    item = self.audio_queue.get(timeout=0.5)
                    current_audio, capture_time = item
                except Empty:
                    continue

                try:
                    raw_current = b""
                    sample_rate_current = 0
                    sample_width_current = 0
                    if self.engine == AsrEngine.SENSEVOICE_SMALL and self.local_engine is not None:
                        raw_current = current_audio.get_raw_data()
                        sample_rate_current = current_audio.sample_rate
                        sample_width_current = current_audio.sample_width
                        rms = _rms(raw_current, sample_width_current)
                        # SenseVoiceSmall 在纯静音/底噪时也可能“幻觉输出”（如单个标点、短英文词）。
                        # 这里用 RMS 能量阈值做一个简单 VAD：低能量片段不送入模型。
                        if rms < self.min_rms:
                            self._last_audio = None
                            self._last_audio_end_time = capture_time
                            continue

                    if on_status:
                        on_status("recognizing")

                    # 1. 尝试拼接逻辑
                    stitched_audio = None
                    is_stitched = False
                    delta: Optional[float] = None

                    if self._last_audio is not None:
                        # 计算时间差：本次捕获时间 - 上次结束时间（近似）
                        delta = capture_time - self._last_audio_end_time

                        if delta < self.stitch_threshold:
                            # 预判拼接后的时长
                            prev_duration = len(self._last_audio.frame_data) / self._last_audio.sample_rate / self._last_audio.sample_width
                            curr_duration = len(current_audio.frame_data) / current_audio.sample_rate / current_audio.sample_width
                            total_duration = prev_duration + curr_duration

                            if total_duration <= self.max_stitch_duration:
                                # 判定为同一句话被切断且未超时，执行拼接
                                # 获取原始字节数据
                                raw_data_prev = self._last_audio.get_raw_data()
                                raw_data_curr = current_audio.get_raw_data()
                                # 拼接字节流
                                new_data = raw_data_prev + raw_data_curr
                                # 创建新的 AudioData 对象
                                stitched_audio = sr.AudioData(
                                    new_data,
                                    current_audio.sample_rate,
                                    current_audio.sample_width,
                                )
                                is_stitched = True

                    # 2. 识别
                    # 如果拼接了，优先识别拼接后的完整音频
                    audio_to_recognize = stitched_audio if is_stitched and stitched_audio else current_audio

                    text = ""
                    if self.engine == AsrEngine.SENSEVOICE_SMALL and self.local_engine is not None:
                        raw_base = audio_to_recognize.get_raw_data()
                        sample_rate = audio_to_recognize.sample_rate
                        sample_width = audio_to_recognize.sample_width

                        use_overlap = (
                            not is_stitched
                            and self.overlap_ms > 0
                            and delta is not None
                            and delta < self.stitch_threshold
                            and self._overlap_tail
                            and sample_rate_current == self._overlap_sample_rate
                            and sample_width_current == self._overlap_sample_width
                        )
                        # 仅在“相邻两段”（delta 较小）且采样参数一致时启用 overlap，避免跨句/跨格式污染。
                        raw_for_asr = (self._overlap_tail + raw_current) if use_overlap else raw_base

                        text = self.local_engine.transcribe(raw_for_asr, sample_rate)  # type: ignore[attr-defined]

                        if self.overlap_ms > 0:
                            bytes_per_second = sample_rate * sample_width
                            tail_len = int(bytes_per_second * (self.overlap_ms / 1000.0))
                            if tail_len > 0:
                                # 保存当前段的尾巴，用于下一段 overlap；尾巴长度按 ms 转为字节数。
                                self._overlap_tail = raw_current[-tail_len:] if tail_len < len(raw_current) else raw_current
                                self._overlap_sample_rate = sample_rate
                                self._overlap_sample_width = sample_width
                            else:
                                self._overlap_tail = b""
                        self._last_voiced_time = capture_time
                    else:
                        # 使用 Google API
                        text = self.recognizer.recognize_google(audio_to_recognize, language=self.language)  # type: ignore

                    # 3. 结果处理与回调
                    if is_stitched:
                        # 这是一个修正结果，通知 UI 覆盖上一条
                        on_result(text, True)  # True = is_correction
                        # 拼接后的结果作为“最新完整句”，继续参与后续拼接。
                        self._last_audio = stitched_audio
                    else:
                        # 这是一个新句子
                        on_result(text, False)  # False = new sentence
                        self._last_audio = current_audio

                    # 更新结束时间
                    self._last_audio_end_time = capture_time

                except sr.UnknownValueError:
                    on_unintelligible()
                    # 无法识别时，重置拼接状态，避免脏数据污染下一句
                    self._last_audio = None
                    self._overlap_tail = b""
                    self._overlap_sample_rate = 0
                    self._overlap_sample_width = 0
                except sr.RequestError as e:
                    on_request_error(e)
                    self._last_audio = None
                    self._overlap_tail = b""
                    self._overlap_sample_rate = 0
                    self._overlap_sample_width = 0
                except Exception as e:
                    # 捕获其他异常（如本地引擎报错）
                    on_request_error(e)
                    self._last_audio = None
                    self._overlap_tail = b""
                    self._overlap_sample_rate = 0
                    self._overlap_sample_width = 0
                finally:
                    try:
                        self.audio_queue.task_done()
                    except Exception:
                        pass

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.stop_listening:
            try:
                self.stop_listening(wait_for_stop=False)
            except TypeError:
                self.stop_listening()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

