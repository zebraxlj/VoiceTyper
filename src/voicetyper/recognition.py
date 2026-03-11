import threading
import time
from queue import Queue, Empty
from typing import Optional, Callable
import speech_recognition as sr

class BackgroundSTT:
    def __init__(
        self,
        recognizer: sr.Recognizer,
        language: str = "zh-CN",
        phrase_time_limit: int = 30,  # 默认调整为 30 秒，避免长语音被硬切
        stitch_threshold: float = 1.0,  # 两段语音间隔小于此值则尝试拼接
        max_stitch_duration: float = 60.0 # 最大拼接时长（秒），防止无限拼接
    ) -> None:
        self.recognizer: sr.Recognizer = recognizer
        self.language: str = language
        self.phrase_time_limit: int = phrase_time_limit
        self.stitch_threshold: float = stitch_threshold
        self.max_stitch_duration: float = max_stitch_duration

        # 队列中存储元组：(AudioData, capture_timestamp)
        self.audio_queue: Queue[tuple[sr.AudioData, float]] = Queue()
        self.stop_event: threading.Event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_listening: Optional[Callable[..., None]] = None

        # 拼接相关状态
        self._last_audio: Optional[sr.AudioData] = None
        self._last_audio_end_time: float = 0.0

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
        on_request_error: Callable[[Exception], None]
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
                    if on_status:
                        on_status("recognizing")

                    # 1. 尝试拼接逻辑
                    stitched_audio = None
                    is_stitched = False

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
                                    current_audio.sample_width
                                )
                                is_stitched = True

                    # 2. 识别
                    # 如果拼接了，优先识别拼接后的完整音频
                    audio_to_recognize = stitched_audio if is_stitched else current_audio

                    text = self.recognizer.recognize_google(audio_to_recognize, language=self.language)  # type: ignore

                    # 3. 结果处理与回调
                    if is_stitched:
                        # 这是一个修正结果，通知 UI 覆盖上一条
                        on_result(text, True)  # True = is_correction
                        # 拼接后的结果作为“最新完整句”，继续参与后续拼接。
                        self._last_audio = stitched_audio
                    else:
                        # 这是一个新句子
                        on_result(text, False) # False = new sentence
                        self._last_audio = current_audio

                    # 更新结束时间
                    self._last_audio_end_time = capture_time

                except sr.UnknownValueError:
                    on_unintelligible()
                    # 无法识别时，重置拼接状态，避免脏数据污染下一句
                    self._last_audio = None
                except sr.RequestError as e:
                    on_request_error(e)
                    self._last_audio = None
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
