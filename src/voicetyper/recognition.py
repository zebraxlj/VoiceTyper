import threading
from queue import Queue, Empty
from typing import Optional, Callable

import speech_recognition as sr


class BackgroundSTT:
    def __init__(self, recognizer: sr.Recognizer, language: str = "zh-CN", phrase_time_limit: int = 10) -> None:
        self.recognizer: sr.Recognizer = recognizer
        self.language: str = language
        self.phrase_time_limit: int = phrase_time_limit
        self.audio_queue: Queue[sr.AudioData] = Queue()
        self.stop_event: threading.Event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_listening: Optional[Callable[..., None]] = None

    def _on_phrase(self, recognizer_inner: sr.Recognizer, audio: sr.AudioData) -> None:
        try:
            self.audio_queue.put_nowait(audio)
        except Exception:
            pass

    def start_listening(self, mic: sr.Microphone) -> None:
        self.stop_listening = self.recognizer.listen_in_background(
            mic, self._on_phrase, phrase_time_limit=self.phrase_time_limit
        )

    def start_worker(
        self,
        on_status: Callable[[str], None],
        on_result: Callable[[str], None],
        on_unintelligible: Callable[[], None],
        on_request_error: Callable[[Exception], None]
    ) -> None:
        def worker() -> None:
            while not self.stop_event.is_set():
                try:
                    audio = self.audio_queue.get(timeout=0.5)
                except Empty:
                    continue
                try:
                    if on_status:
                        on_status("recognizing")
                    text = self.recognizer.recognize_google(audio, language=self.language)  # type: ignore
                    on_result(text)
                except sr.UnknownValueError:
                    on_unintelligible()
                except sr.RequestError as e:
                    on_request_error(e)
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
                # Some versions might not support wait_for_stop or behave differently
                self.stop_listening()
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
