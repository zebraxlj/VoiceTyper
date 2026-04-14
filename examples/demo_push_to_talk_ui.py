import ctypes
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import pyaudio
import re
import signal
import tkinter as tk
from pynput import keyboard as pynput_keyboard

import demo_consts

sys.path.append(demo_consts.SRC_DIR)

from voicetyper import AudioDeviceResolver  # noqa: E402
from voicetyper.models import SenseVoiceSmallEngine  # noqa: E402


@dataclass
class RecorderConfig:
    rate: int = 16000
    channels: int = 1
    frames_per_buffer: int = 1024


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
                data = self._stream.read(
                    self.config.frames_per_buffer, exception_on_overflow=False
                )
            except Exception:
                continue
            self._frames.append(data)

    def stop(self) -> bytes:
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
        try:
            if self._stream:
                self._stream.close()
        except Exception:
            pass
        try:
            self._pa.terminate()
        except Exception:
            pass


class OverlayUI:
    def __init__(self) -> None:
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.92)
        try:
            self._root.attributes("-toolwindow", True)
            self._root.attributes("-disabled", True)
        except Exception:
            pass
        self._root.configure(bg="#121212")

        frame = tk.Frame(
            self._root,
            bg="#121212",
            bd=1,
            highlightthickness=1,
            highlightbackground="#2a2a2a",
        )
        frame.pack(padx=12, pady=10)

        self._label = tk.Label(
            frame,
            text="🎤 正在收音",
            font=("Segoe UI", 12, "bold"),
            fg="#f6f6f6",
            bg="#121212",
        )
        self._label.pack()

    def _position_bottom_center(self) -> None:
        self._root.update_idletasks()
        width = self._root.winfo_width()
        height = self._root.winfo_height()
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = int((screen_w - width) / 2)
        y = int(screen_h - height - 40)
        self._root.geometry(f"{width}x{height}+{x}+{y}")

    def show(self) -> None:
        def _do_show() -> None:
            self._root.deiconify()
            self._root.lift()
            self._position_bottom_center()

        self._root.after(0, _do_show)

    def hide(self) -> None:
        self._root.after(0, self._root.withdraw)

    def set_clipboard(self, text: str) -> None:
        done = threading.Event()

        def _do_set() -> None:
            try:
                self._root.clipboard_clear()
                self._root.clipboard_append(text)
                self._root.update()
            finally:
                done.set()

        self._root.after(0, _do_set)
        done.wait(timeout=1.0)

    def run(self, exit_event: threading.Event) -> None:
        def _poll() -> None:
            if exit_event.is_set():
                self._root.quit()
                return
            self._root.after(100, _poll)

        self._root.after(100, _poll)
        self._root.mainloop()


def _is_meaningful_text(text: str) -> bool:
    # Filter out empty/garbage results produced by silence.
    value = text.strip()
    if not value:
        return False
    # Reject pure punctuation/symbols (no letters/digits/CJK)
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", value):
        return False
    lowered = value.lower()
    fillers = {
        "yeah",
        "yea",
        "yep",
        "yah",
        "ya",
        "uh",
        "um",
        "er",
        "ah",
        "eh",
        "mm",
        "hmm",
        "嗯",
        "嗯嗯",
        "啊",
        "呃",
        "额",
        "嗯哼",
        "em",
    }
    if lowered in fillers:
        return False
    return True


def _start_tray_icon(exit_event: threading.Event):
    # Windows-only tray icon with a simple "Exit" menu.
    if sys.platform != "win32":
        return True, None
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"Tray icon unavailable (pystray/Pillow not installed): {exc}")
        return False, None

    def _build_cursor_icon(size: int = 16):
        # Minimal cursor glyph to avoid external icon files.
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        # Simple cursor arrow
        points = [
            (1, 1),
            (1, size - 2),
            (6, size - 7),
            (9, size - 4),
            (11, size - 6),
            (6, size - 11),
        ]
        draw.polygon(points, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        return image

    def _on_exit(icon, item):
        exit_event.set()
        icon.stop()

    try:
        icon = pystray.Icon(
            "VoiceTyper",
            _build_cursor_icon(),
            "VoiceTyper",
            menu=pystray.Menu(pystray.MenuItem("Exit", _on_exit)),
        )
        thread = threading.Thread(target=icon.run, daemon=True)
        thread.start()
    except Exception as exc:
        print(f"Tray icon failed to start: {exc}")
        return False, None
    return True, icon


def main() -> None:
    hold_ms = 300
    strip_trailing_period = True  # 设为 False 可保留识别结果末尾的句号/句点
    hold_s = max(0.0, hold_ms / 1000.0)
    exit_event = threading.Event()
    transcribe_lock = threading.Lock()

    print("=== UI Push-to-Talk Demo ===")
    print(
        "Hold left left Shift+Win to record, release to transcribe and type. Use tray menu to exit."
    )

    with AudioDeviceResolver() as resolver:
        info = resolver.default_input()
        device_index = int(info["index"]) if info else None
        if info:
            print(f"默认麦克风设备: [{info['index']}] {info['name']}")
        else:
            print("默认麦克风设备: 未知")

    try:
        # Load model upfront so first use is fast.
        model = SenseVoiceSmallEngine(
            strip_trailing_period=strip_trailing_period,
            quantized=False,
        )
    except Exception as exc:
        print(f"模型加载失败: {exc}")
        return

    config = RecorderConfig()
    recorder = PushToTalkRecorder(device_index=device_index, config=config)
    ui = OverlayUI()
    kb_controller = pynput_keyboard.Controller()
    tray_ok, tray_icon = _start_tray_icon(exit_event)
    if not tray_ok:
        print("Tray icon unavailable on Windows; aborting startup.")
        try:
            recorder.close()
        except Exception:
            pass
        return

    recording = False
    ctrl_down = False
    shift_down = False
    win_down = False
    last_down_time = 0.0
    record_start_time = 0.0
    start_timer: Optional[threading.Timer] = None
    state_lock = threading.Lock()

    # Windows 虚拟键码
    VK_LSHIFT = 0xA0
    VK_LWIN = 0x5B

    def _is_key_physically_pressed(vk: int) -> bool:
        """通过 Win32 GetAsyncKeyState 查询按键是否真的被按着。"""
        # 最高位 (0x8000) 表示此刻按键处于按下状态
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)

    def _hotkey_active() -> bool:
        return shift_down and win_down
        # return ctrl_down or (shift_down and win_down)

    def _hotkey_physically_held() -> bool:
        """在关键时刻用 Win32 API 二次确认两个键是否真的同时按着。"""
        return _is_key_physically_pressed(VK_LSHIFT) and _is_key_physically_pressed(
            VK_LWIN
        )

    def _sync_modifier_state() -> None:
        """根据物理按键状态修正软件追踪的标志位，防止漏掉 release 事件导致状态残留。"""
        nonlocal shift_down, win_down
        if shift_down and not _is_key_physically_pressed(VK_LSHIFT):
            shift_down = False
        if win_down and not _is_key_physically_pressed(VK_LWIN):
            win_down = False

    def start_recording() -> None:
        # Called after hold threshold is met.
        nonlocal recording, record_start_time
        with state_lock:
            # 用物理按键状态做最终确认，防止 release 事件被系统吞掉后状态残留
            _sync_modifier_state()
            if not _hotkey_active() or not _hotkey_physically_held() or recording:
                return
            recording = True
            record_start_time = time.time()
        ui.show()
        recorder.start()

    def transcribe_and_type(pcm16: bytes) -> None:
        # Run ASR, then write to clipboard and current cursor.
        with transcribe_lock:
            try:
                text = model.transcribe(pcm16, config.rate).strip()
            except Exception as exc:
                print(f"识别失败: {exc}")
                return
            if not text:
                print("识别结果为空。")
                return
            if not _is_meaningful_text(text):
                print(f"忽略无意义结果: {text}")
                return
            ui.set_clipboard(text)
            kb_controller.type(text)
            print(f"识别结果: {text}")

    def on_press(key) -> None:
        # Start hold timer on left Shift+Win press.
        # Guard against key auto-repeat: only react to genuine new presses.
        nonlocal ctrl_down, shift_down, win_down, start_timer, last_down_time
        if key == pynput_keyboard.Key.ctrl_l:
            if ctrl_down:
                return  # auto-repeat, ignore
            ctrl_down = True
        elif key == pynput_keyboard.Key.shift_l:
            if shift_down:
                return  # auto-repeat, ignore
            shift_down = True
        elif key == pynput_keyboard.Key.cmd_l:
            if win_down:
                return  # auto-repeat, ignore
            win_down = True
        else:
            return
        with state_lock:
            if recording:
                return
            _sync_modifier_state()
            if not _hotkey_active():
                return
            if start_timer and start_timer.is_alive():
                return
            last_down_time = time.time()

            start_timer = threading.Timer(hold_s, start_recording)
            start_timer.daemon = True
            start_timer.start()

    def on_release(key):
        # On release, stop recording and kick off transcription.
        nonlocal recording, ctrl_down, shift_down, win_down, start_timer
        if key == pynput_keyboard.Key.ctrl_l:
            ctrl_down = False
        elif key == pynput_keyboard.Key.shift_l:
            shift_down = False
        elif key == pynput_keyboard.Key.cmd_l:
            win_down = False
        else:
            return True
        with state_lock:
            if start_timer and start_timer.is_alive() and not _hotkey_active():
                try:
                    start_timer.cancel()
                except Exception:
                    pass
                start_timer = None
            if not recording:
                return True
            if _hotkey_active():
                return True
            recording = False
        ui.hide()
        pcm16 = recorder.stop()
        dur_base = record_start_time if record_start_time > 0 else last_down_time
        dur = max(0.0, time.time() - dur_base)
        if not pcm16:
            print("录音为空。")
            return True
        print(f"识别中... ({dur:.1f}s)")
        threading.Thread(target=transcribe_and_type, args=(pcm16,), daemon=True).start()
        return True

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    def _sigint_handler(signum, frame) -> None:
        exit_event.set()

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except Exception:
        pass

    try:
        ui.run(exit_event)
    except KeyboardInterrupt:
        exit_event.set()
        print("结束进程：收到 KeyboardInterrupt，准备退出")
    finally:
        if recording:
            try:
                recorder.stop()
            except Exception:
                pass
        try:
            listener.stop()
            print("结束进程：键盘 listener 已终止")
        except Exception as e:
            print(f"结束进程：键盘 listener 终止出错：{e}")
        if tray_icon:
            try:
                tray_icon.stop()
            except Exception:
                pass
        try:
            recorder.close()
            print("结束进程：收音已终止")
        except Exception as e:
            print(f"结束进程：收音终止出错：{e}")
        print("结束进程：已退出")


if __name__ == "__main__":
    main()
