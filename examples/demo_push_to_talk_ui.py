import ctypes
import logging
import re
import signal
import sys
import threading
import time
import tkinter as tk
from typing import Optional

from pynput import keyboard as pynput_keyboard

from voicetyper import AudioDeviceResolver, PushToTalkRecorder, RecorderConfig
from voicetyper.models import SenseVoiceSmallEngine
from voicetyper.monitor import ResourceMonitor

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

SHOW_RESOURCE_USAGE = False


if sys.platform == "win32":
    _INPUT_KEYBOARD = 1
    _KEYEVENTF_UNICODE = 0x0004
    _KEYEVENTF_KEYUP = 0x0002

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", ctypes.c_ulong),
            ("wParamL", ctypes.c_ushort),
            ("wParamH", ctypes.c_ushort),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", _MOUSEINPUT),
            ("ki", _KEYBDINPUT),
            ("hi", _HARDWAREINPUT),
        ]

    class _INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("union", _INPUT_UNION),
        ]

    _SendInput = ctypes.windll.user32.SendInput
    _SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_INPUT), ctypes.c_int]
    _SendInput.restype = ctypes.c_uint

    def _send_text(text: str, char_delay: float = 0.01) -> None:
        """Type text via Win32 SendInput with KEYEVENTF_UNICODE, bypassing IME."""
        for ch in text:
            code = ord(ch)
            inp_down = _INPUT()
            inp_down.type = _INPUT_KEYBOARD
            inp_down.union.ki.wVk = 0
            inp_down.union.ki.wScan = code
            inp_down.union.ki.dwFlags = _KEYEVENTF_UNICODE
            inp_down.union.ki.time = 0
            inp_down.union.ki.dwExtraInfo = None

            inp_up = _INPUT()
            inp_up.type = _INPUT_KEYBOARD
            inp_up.union.ki.wVk = 0
            inp_up.union.ki.wScan = code
            inp_up.union.ki.dwFlags = _KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP
            inp_up.union.ki.time = 0
            inp_up.union.ki.dwExtraInfo = None

            inputs = (_INPUT * 2)(inp_down, inp_up)
            _SendInput(2, inputs, ctypes.sizeof(_INPUT))
            if char_delay > 0:
                time.sleep(char_delay)

else:
    _kb_controller = pynput_keyboard.Controller()

    def _send_text(text: str, char_delay: float = 0.01) -> None:
        """Type text via pynput (non-Windows fallback)."""
        _kb_controller.type(text)


class OverlayUI:
    _DEFAULT_TEXT = "🎤 正在收音"

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
            text=self._DEFAULT_TEXT,
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

    def show(self, text: Optional[str] = None) -> None:
        def _do_show() -> None:
            if text is not None:
                self._label.configure(text=text)
            else:
                self._label.configure(text=self._DEFAULT_TEXT)
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


def _is_admin() -> bool:
    """Check if the current process has administrator privileges."""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _restart_as_admin() -> None:
    """Relaunch the current script/exe with elevated privileges via UAC."""
    if getattr(sys, "frozen", False):
        executable = sys.executable
        params = " ".join(sys.argv[1:])
    else:
        executable = sys.executable
        params = " ".join([f'"{sys.argv[0]}"'] + sys.argv[1:])

    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", executable, params, None, 1
    )
    sys.exit(0)


def _open_settings(root: tk.Tk, exit_event: threading.Event) -> None:
    """Open a settings window showing admin status and restart option."""

    def _create() -> None:
        win = tk.Toplevel(root)
        win.title("VoiceTyper Settings")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.configure(bg="#1e1e1e")

        frame = tk.Frame(win, bg="#1e1e1e", padx=20, pady=16)
        frame.pack()

        is_admin = _is_admin()
        status_value = "Yes" if is_admin else "No"
        status_color = "#4ec959" if is_admin else "#e05252"

        status_frame = tk.Frame(frame, bg="#1e1e1e")
        status_frame.pack(anchor="w", pady=(0, 12))

        tk.Label(
            status_frame,
            text="Running as Administrator: ",
            font=("Segoe UI", 10),
            fg="#cccccc",
            bg="#1e1e1e",
        ).pack(side="left")

        tk.Label(
            status_frame,
            text=status_value,
            font=("Segoe UI", 10, "bold"),
            fg=status_color,
            bg="#1e1e1e",
        ).pack(side="left")

        if not is_admin:
            tk.Label(
                frame,
                text="Admin is needed for hotkeys to work in Store apps\n(Windows Terminal, VSCode Insider, etc.)",
                font=("Segoe UI", 9),
                fg="#999999",
                bg="#1e1e1e",
                justify="left",
            ).pack(anchor="w", pady=(0, 12))

        tk.Button(
            frame,
            text="Restart as Administrator",
            font=("Segoe UI", 10),
            command=lambda: [win.destroy(), exit_event.set(), _restart_as_admin()],
            state="disabled" if is_admin else "normal",
        ).pack(anchor="w")

        win.update_idletasks()
        w = win.winfo_width()
        h = win.winfo_height()
        x = (win.winfo_screenwidth() - w) // 2
        y = (win.winfo_screenheight() - h) // 2
        win.geometry(f"+{x}+{y}")

    root.after(0, _create)


def _start_tray_icon(tk_root: tk.Tk, exit_event: threading.Event):
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

    def _on_settings(icon, item):
        _open_settings(tk_root, exit_event)

    def _on_exit(icon, item):
        exit_event.set()
        icon.stop()

    try:
        icon = pystray.Icon(
            "VoiceTyper",
            _build_cursor_icon(),
            "VoiceTyper",
            menu=pystray.Menu(
                pystray.MenuItem("Settings", _on_settings),
                pystray.MenuItem("Exit", _on_exit),
            ),
        )
        thread = threading.Thread(target=icon.run, daemon=True)
        thread.start()
    except Exception as exc:
        print(f"Tray icon failed to start: {exc}")
        return False, None
    return True, icon


class PushToTalkSession:
    VK_LSHIFT = 0xA0
    VK_LWIN = 0x5B

    def __init__(
        self,
        ui: OverlayUI,
        recorder: PushToTalkRecorder,
        config: RecorderConfig,
        exit_event: threading.Event,
        hold_s: float,
        strip_trailing_period: bool,
    ) -> None:
        self._ui = ui
        self._recorder = recorder
        self._config = config
        self._exit_event = exit_event
        self._hold_s = hold_s
        self._strip_trailing_period = strip_trailing_period

        self._model: Optional[SenseVoiceSmallEngine] = None
        self._model_ready = threading.Event()
        self._transcribe_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._recording = False
        self._shift_down = False
        self._win_down = False
        self._last_down_time = 0.0
        self._record_start_time = 0.0
        self._start_timer: Optional[threading.Timer] = None

    def load_model(self) -> None:
        try:
            self._model = SenseVoiceSmallEngine(
                strip_trailing_period=self._strip_trailing_period,
                quantized=False,
            )
        except Exception as exc:
            print(f"模型加载失败: {exc}")
            self._exit_event.set()
        finally:
            self._model_ready.set()
            self._ui.hide()

    def _hotkey_active(self) -> bool:
        return self._shift_down and self._win_down

    def _is_key_physically_pressed(self, vk: int) -> bool:
        if sys.platform != "win32":
            return True
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)

    def _sync_modifier_state(self) -> None:
        if sys.platform != "win32":
            return
        if self._shift_down and not self._is_key_physically_pressed(self.VK_LSHIFT):
            self._shift_down = False
        if self._win_down and not self._is_key_physically_pressed(self.VK_LWIN):
            self._win_down = False

    def _start_recording(self) -> None:
        with self._state_lock:
            self._sync_modifier_state()
            if not self._hotkey_active() or self._recording:
                return
            if sys.platform == "win32" and not (
                self._is_key_physically_pressed(self.VK_LSHIFT)
                and self._is_key_physically_pressed(self.VK_LWIN)
            ):
                return
            if not self._model_ready.is_set():
                print("模型尚未加载完成，请稍候...")
                return
            if self._model is None:
                return
            self._recording = True
            self._record_start_time = time.time()
        self._ui.show()
        self._recorder.start()

    def _stop_recording(self) -> None:
        self._ui.hide()
        pcm16 = self._recorder.stop()
        dur_base = self._record_start_time if self._record_start_time > 0 else self._last_down_time
        dur = max(0.0, time.time() - dur_base)
        if not pcm16:
            print("录音为空。")
            return
        print(f"识别中... ({dur:.1f}s)")
        threading.Thread(
            target=self._transcribe_and_type, args=(pcm16,), daemon=True
        ).start()

    def _transcribe_and_type(self, pcm16: bytes) -> None:
        with self._transcribe_lock:
            if self._model is None:
                print("模型未就绪，跳过识别。")
                return
            try:
                text = self._model.transcribe(pcm16, self._config.rate).strip()
            except Exception as exc:
                print(f"识别失败: {exc}")
                return
            if not text:
                print("识别结果为空。")
                return
            if not _is_meaningful_text(text):
                print(f"忽略无意义结果: {text}")
                return
            self._ui.set_clipboard(text)
            _send_text(text)
            print(f"识别结果: {text}")

    def on_press(self, key) -> None:
        if key == pynput_keyboard.Key.shift_l:
            if self._shift_down:
                return
            self._shift_down = True
        elif key == pynput_keyboard.Key.cmd_l:
            if self._win_down:
                return
            self._win_down = True
        else:
            return
        with self._state_lock:
            if self._recording:
                return
            self._sync_modifier_state()
            if not self._hotkey_active():
                return
            if self._start_timer and self._start_timer.is_alive():
                return
            self._last_down_time = time.time()
            self._start_timer = threading.Timer(self._hold_s, self._start_recording)
            self._start_timer.daemon = True
            self._start_timer.start()

    def on_release(self, key):
        if key == pynput_keyboard.Key.shift_l:
            self._shift_down = False
        elif key == pynput_keyboard.Key.cmd_l:
            self._win_down = False
        else:
            return True
        with self._state_lock:
            if self._start_timer and self._start_timer.is_alive() and not self._hotkey_active():
                try:
                    self._start_timer.cancel()
                except Exception:
                    pass
                self._start_timer = None
            if not self._recording:
                return True
            if self._hotkey_active():
                return True
            self._recording = False
        self._stop_recording()
        return True

    def run(self, tray_icon) -> None:
        self._ui.show(text="⏳ 模型加载中...")
        threading.Thread(target=self.load_model, daemon=True).start()

        listener = pynput_keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        listener.start()

        def _sigint_handler(signum, frame) -> None:
            self._exit_event.set()

        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except Exception:
            pass

        try:
            self._ui.run(self._exit_event)
        except KeyboardInterrupt:
            self._exit_event.set()
        finally:
            if self._recording:
                try:
                    self._recorder.stop()
                except Exception:
                    pass
            try:
                listener.stop()
            except Exception:
                pass
            if tray_icon:
                try:
                    tray_icon.stop()
                except Exception:
                    pass
            try:
                self._recorder.close()
            except Exception:
                pass


def main() -> None:
    hold_ms = 300
    strip_trailing_period = True
    exit_event = threading.Event()

    if SHOW_RESOURCE_USAGE:
        ResourceMonitor(interval=1.0, exit_event=exit_event).start()

    print("=== UI Push-to-Talk Demo ===")
    print(
        "Hold left Shift+Win to record, release to transcribe and type. Use tray menu to exit."
    )

    with AudioDeviceResolver() as resolver:
        info = resolver.default_input()
        device_index = int(info["index"]) if info else None
        if info:
            print(f"默认麦克风设备: [{info['index']}] {info['name']}")
        else:
            print("默认麦克风设备: 未知")

    config = RecorderConfig()
    recorder = PushToTalkRecorder(device_index=device_index, config=config)
    ui = OverlayUI()
    tray_ok, tray_icon = _start_tray_icon(ui._root, exit_event)
    if not tray_ok:
        print("Tray icon unavailable on Windows; aborting startup.")
        recorder.close()
        return

    session = PushToTalkSession(
        ui=ui,
        recorder=recorder,
        config=config,
        exit_event=exit_event,
        hold_s=max(0.0, hold_ms / 1000.0),
        strip_trailing_period=strip_trailing_period,
    )
    session.run(tray_icon=tray_icon)


if __name__ == "__main__":
    main()
