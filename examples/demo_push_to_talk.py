import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import argparse
import pyaudio
import signal
import speech_recognition as sr
from colorama import init, Fore, Style
from pynput import keyboard

import demo_consts

sys.path.append(demo_consts.SRC_DIR)

from voicetyper import AudioDeviceResolver, AsrEngine  # noqa: E402
from voicetyper.models import SenseVoiceSmallEngine  # noqa: E402

init(autoreset=True)


@dataclass
class RecorderConfig:
    rate: int = 16000
    channels: int = 1
    frames_per_buffer: int = 1024
    sample_width: int = 2


class PushToTalkRecorder:
    def __init__(self, device_index: Optional[int], config: RecorderConfig) -> None:
        self.device_index = device_index
        self.config = config
        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._frames: list[bytes] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
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

    def _loop(self) -> None:
        assert self._stream is not None
        while not self._stop_event.is_set():
            try:
                data = self._stream.read(self.config.frames_per_buffer, exception_on_overflow=False)
            except Exception:
                continue
            self._frames.append(data)

    def stop(self) -> bytes:
        self._stop_event.set()
        if self._thread:
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
        try:
            if self._stream:
                self._stream.close()
        except Exception:
            pass
        try:
            self._pa.terminate()
        except Exception:
            pass


def transcribe_google(recognizer: sr.Recognizer, pcm16: bytes, rate: int, language: str) -> str:
    audio = sr.AudioData(pcm16, rate, 2)
    return recognizer.recognize_google(audio, language=language)  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(prog="demo_push_to_talk")
    parser.add_argument(
        "--hold-ms",
        type=int,
        default=300,
        help="长按判定阈值（毫秒）。按住超过该阈值才开始录音，松开后才送入识别。",
    )
    parser.add_argument(
        "--keep-period",
        action="store_true",
        default=False,
        help="保留识别结果末尾的句号/句点（默认自动去除）。",
    )
    args = parser.parse_args()
    hold_ms = args.hold_ms
    strip_trailing_period = not args.keep_period

    engine = AsrEngine.SENSEVOICE_SMALL
    language = "zh-CN"
    exit_event = threading.Event()

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    recognizer.non_speaking_duration = 0.3

    print(f"{Fore.CYAN}=== Push-to-Talk（左 Ctrl 长按说话，松开识别）==={Style.RESET_ALL}")

    recording = False
    recorder: Optional[PushToTalkRecorder] = None

    try:
        with AudioDeviceResolver() as resolver:
            info = resolver.default_input()
            device_index = int(info["index"]) if info else None
            if info:
                print(f"{Fore.YELLOW}默认麦克风设备: [{info['index']}] {info['name']}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}默认麦克风设备: 未知{Style.RESET_ALL}")

        model: Optional[SenseVoiceSmallEngine] = None
        if engine == AsrEngine.SENSEVOICE_SMALL:
            try:
                model = SenseVoiceSmallEngine(
                    strip_trailing_period=strip_trailing_period
                )
            except Exception as e:
                print(f"{Fore.RED}本地模型初始化失败: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}回退到 Google 识别（需要联网）{Style.RESET_ALL}")
                engine = AsrEngine.GOOGLE

        config = RecorderConfig()
        recorder = PushToTalkRecorder(device_index=device_index, config=config)

        hold_s = max(0.0, hold_ms / 1000.0)
        last_down_time = 0.0
        record_start_time = 0.0
        ctrl_down = False
        start_timer: Optional[threading.Timer] = None

        def on_press(key) -> None:
            nonlocal recording, last_down_time, ctrl_down, start_timer, record_start_time
            if isinstance(key, keyboard.KeyCode) and key.char in {"c", "C"} and ctrl_down:
                exit_event.set()
                return
            if key != keyboard.Key.ctrl_l:
                return
            ctrl_down = True
            if recording:
                return
            if start_timer and start_timer.is_alive():
                return
            last_down_time = time.time()

            def _start_if_still_held() -> None:
                nonlocal recording, record_start_time
                if not ctrl_down or recording:
                    return
                recording = True
                record_start_time = time.time()
                print(f"{Fore.MAGENTA}录音中...{Style.RESET_ALL}")
                recorder.start()

            start_timer = threading.Timer(hold_s, _start_if_still_held)
            start_timer.daemon = True
            start_timer.start()

        def on_release(key):
            nonlocal recording, ctrl_down, start_timer
            if key != keyboard.Key.ctrl_l:
                return True
            ctrl_down = False
            if start_timer and start_timer.is_alive():
                try:
                    start_timer.cancel()
                except Exception:
                    pass
            if not recording:
                return True
            recording = False
            pcm16 = recorder.stop()
            dur_base = record_start_time if record_start_time > 0 else last_down_time
            dur = max(0.0, time.time() - dur_base)
            if not pcm16:
                print(f"{Fore.RED}[空音频]{Style.RESET_ALL}")
                return True

            print(f"{Fore.YELLOW}识别中...{Style.RESET_ALL} ({dur:.1f}s)")
            try:
                if engine == AsrEngine.SENSEVOICE_SMALL and model is not None:
                    text = model.transcribe(pcm16, config.rate)
                else:
                    text = transcribe_google(recognizer, pcm16, config.rate, language)
                print(f"{Fore.GREEN}结果: {Style.BRIGHT}{text}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}识别失败: {e}{Style.RESET_ALL}")
            return True

        print(f"{Fore.BLUE}左 Ctrl 按住超过 {hold_ms}ms 开始录音，松开结束并识别；Ctrl+C 退出{Style.RESET_ALL}")

        def _sigint_handler(signum, frame) -> None:
            exit_event.set()

        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except Exception:
            pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        while not exit_event.is_set():
            time.sleep(0.05)
        try:
            listener.stop()
        except Exception:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        if recording and recorder:
            try:
                recorder.stop()
                print(f"{Fore.MAGENTA}录音已停止{Style.RESET_ALL}")
            except Exception:
                pass
        if recorder:
            recorder.close()
            print(f"{Fore.MAGENTA}录音器已关闭{Style.RESET_ALL}")
        print(f"{Fore.CYAN}=== 程序已退出 ==={Style.RESET_ALL}")


if __name__ == "__main__":
    main()
