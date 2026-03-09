import sys
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Union, Callable

import speech_recognition as sr
import pyaudio as _p
from colorama import init, Fore, Style

init(autoreset=True)

class AudioDeviceResolver:
    def __init__(self) -> None:
        self._pa: Optional[_p.PyAudio] = None
        self._available: bool = False
        try:
            self._pa = _p.PyAudio()
            self._available = True
        except Exception:
            self._available = False

    def default_input(self) -> Optional[Dict[str, Union[int, str]]]:
        if not self._available or self._pa is None:
            return None
        try:
            dev = self._pa.get_default_input_device_info()
            if 'index' not in dev:
                return None
            return {"index": int(dev["index"]), "name": str(dev.get("name", "Unknown"))}
        except Exception:
            return None

    def close(self) -> None:
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass


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


def on_status_print(msg: str) -> None:
    if msg == "recognizing":
        sys.stdout.write("\033[K")
        print(f"{Fore.YELLOW}识别中...{Style.RESET_ALL}", end="\r")


def on_result_print(text: str) -> None:
    sys.stdout.write("\033[K")
    print(f"{Fore.GREEN}结果: {Style.BRIGHT}{text}{Style.RESET_ALL}")


def on_unintelligible_print() -> None:
    sys.stdout.write("\033[K")
    print(f"{Fore.RED}[无法识别]{Style.RESET_ALL}")


def on_request_error_print(e: Exception) -> None:
    sys.stdout.write("\033[K")
    print(f"{Fore.RED}无法连接到 Google API: {e}{Style.RESET_ALL}")


def main() -> None:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    recognizer.non_speaking_duration = 0.3

    print(f"{Fore.CYAN}=== 实时连续语音转文字（Google API）==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}正在初始化麦克风...{Style.RESET_ALL}")

    try:
        resolver = AudioDeviceResolver()
        info = resolver.default_input()
        print('info:', info)
        if info:
            print(f"{Fore.YELLOW}默认麦克风设备: [{info['index']}] {info['name']}{Style.RESET_ALL}")
            mic = sr.Microphone(device_index=info["index"])
        else:
            print(f"{Fore.YELLOW}默认麦克风设备: 未知{Style.RESET_ALL}")
            mic = sr.Microphone()
        resolver.close()
        print(f"{Fore.GREEN}麦克风就绪!{Style.RESET_ALL}")
        print("正在进行环境噪声校准，请保持安静 1 秒...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"{Fore.GREEN}校准完成!{Style.RESET_ALL}")

        service = BackgroundSTT(recognizer, language="zh-CN", phrase_time_limit=10)
        service.start_listening(mic)
        service.start_worker(
            on_status_print, on_result_print, on_unintelligible_print, on_request_error_print
        )

        print(f"\n{Fore.MAGENTA}>>> 开始讲话（Ctrl+C 退出），持续收听中... <<<{Style.RESET_ALL}\n")

        try:
            while True:
                print(f"{Fore.BLUE}收听中...{Style.RESET_ALL}", end="\r")
                if service.worker_thread:
                    service.worker_thread.join(0.2)
        except KeyboardInterrupt:
            pass

        service.stop()

    except OSError as e:
        print(f"{Fore.RED}错误: 无法访问麦克风。请检查麦克风设置或权限。{Style.RESET_ALL}")
        print(f"详情: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{Fore.CYAN}=== 程序已退出 ==={Style.RESET_ALL}")

if __name__ == "__main__":
    main()
