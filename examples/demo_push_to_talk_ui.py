import ctypes
import logging
import re
import signal
import sys
import threading
import time
import tkinter as tk
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import ttk
from typing import Optional

from pynput import keyboard as pynput_keyboard

from voicetyper import PushToTalkRecorder, RecorderConfig
from voicetyper.models import SenseVoiceSmallEngine
from voicetyper.monitor import ResourceMonitor

# 日志开关：True = 控制台显示完整格式（含日期/模块名/DEBUG），False = 简洁格式。
# 仅影响开发时的控制台输出（打包后无控制台）；需要调试时手动改这一行即可。
# 文件日志始终为 DEBUG + 完整格式，不受此开关影响。
VERBOSE_CONSOLE = False

# 配置日志：同时输出到控制台和文件
# 开发环境：项目根目录下的 logs
# 打包后的 exe：exe 所在目录下的 logs
if getattr(sys, "frozen", False):
    log_dir = Path(sys.executable).parent / "logs"
else:
    log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "voicetyper.log"

# 完整格式（含日期、毫秒、模块名）
_VERBOSE_FMT = logging.Formatter(
    "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# 简洁格式（仅时间、级别、消息）
_SIMPLE_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)

# 文件处理器：始终使用完整格式，便于排查
file_handler = RotatingFileHandler(
    log_file, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(_VERBOSE_FMT)

# 控制台处理器：根据 VERBOSE_CONSOLE 切换格式和级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if VERBOSE_CONSOLE else logging.INFO)
console_handler.setFormatter(_VERBOSE_FMT if VERBOSE_CONSOLE else _SIMPLE_FMT)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger("voicetyper.app")

SHOW_RESOURCE_USAGE = False


def _resource_path(relative: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative
    return Path(__file__).resolve().parent.parent / relative


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

    _ICON_PATH = _resource_path("UI/assets/IconApp.png")

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
        if self._ICON_PATH.exists():
            _icon_img = tk.PhotoImage(file=str(self._ICON_PATH))
            self._root.iconphoto(True, _icon_img)
            self._icon_photo_ref = _icon_img
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


def _audio_devices():
    """Import the UI-agnostic input-device model (repo-root ``UI`` package).

    The package lives at the repo root, not under ``examples/``, so make sure
    the root is importable before pulling it in.
    """
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from UI import audio_devices

    return audio_devices


def _open_settings(
    root: tk.Tk,
    exit_event: threading.Event,
    recorder: "PushToTalkRecorder",
) -> None:
    """Open a settings window showing admin status, input device picker, and restart option."""

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
        ).pack(anchor="w", pady=(0, 12))

        # ── Input device picker ──────────────────────────────
        tk.Label(
            frame,
            text="Input Device",
            font=("Segoe UI", 10, "bold"),
            fg="#cccccc",
            bg="#1e1e1e",
        ).pack(anchor="w", pady=(4, 4))

        # UI 无关的设备选择逻辑（枚举/持久化/断开重连）都在 UI.audio_devices 里；
        # 这里只负责渲染成 Combobox 并把"何时刷新"接上去。正式 UI 可复用同一套。
        audio_devices = _audio_devices()
        selector = audio_devices.InputDeviceSelector()
        selector.select_by_index(recorder.device_index)

        device_var = tk.StringVar(value=selector.selected_label)
        combo = ttk.Combobox(
            frame,
            textvariable=device_var,
            values=selector.labels(),
            state="readonly",
            width=50,
        )
        combo.pack(anchor="w", pady=(0, 6))

        status_msg = tk.Label(
            frame,
            text="",
            font=("Segoe UI", 9),
            fg="#999999",
            bg="#1e1e1e",
        )
        status_msg.pack(anchor="w", pady=(0, 6))

        def _on_device_change(_event=None) -> None:
            label = device_var.get()
            new_idx = selector.select(label)
            try:
                recorder.set_device_index(new_idx)
                status_msg.configure(text=f"Switched to: {label}", fg="#4ec959")
                logger.info(f"输入设备已切换: {label}")
                selector.persist()  # 记住本次选择，下次启动自动恢复
            except Exception as exc:
                status_msg.configure(text=f"Failed: {exc}", fg="#e05252")
                logger.exception(f"切换输入设备失败: {exc}")

        def _apply_list() -> None:
            # 主线程：重建后重新枚举并刷新下拉。
            if not combo.winfo_exists():
                return
            recorder.rescan_devices()  # 重建 PortAudio 以发现热插拔
            selector.refresh()
            combo.configure(values=selector.labels())
            device_var.set(selector.selected_label)

        # 设备热插拔监听：Windows 上事件驱动，其它平台/无 comtypes 时回退到轮询。
        # 关键约束：PortAudio 的 WASAPI 把 COM 套间绑在调用线程上，且 Tk 不是线程
        # 安全的——所以重扫和 UI 刷新必须都在主线程。COM 回调只置一个标志位，真正
        # 的重扫由主线程轮询循环 _poll_devices 触发，避免临时线程导致的枚举不全与
        # 访问越界崩溃（0xC0000005）。
        device_dirty = threading.Event()
        watcher = audio_devices.DeviceChangeWatcher()
        watching = watcher.start(device_dirty.set)

        if watching:
            device_dirty.set()  # 打开时刷新一次，纳入启动后插拔的设备

            def _poll_devices() -> None:
                if not combo.winfo_exists():
                    return  # 设置窗口已关闭，停止轮询
                if device_dirty.is_set():
                    device_dirty.clear()
                    _apply_list()
                root.after(400, _poll_devices)

            root.after(400, _poll_devices)

            def _refresh_devices() -> None:
                # 下拉只读缓存 → 秒开（列表由 _poll_devices 在主线程保持最新）。
                combo.configure(values=selector.labels())
                device_var.set(selector.selected_label)
        else:
            # 回退：打开下拉时在主线程同步重扫。
            def _refresh_devices() -> None:
                _apply_list()

        combo.configure(postcommand=_refresh_devices)
        combo.bind("<<ComboboxSelected>>", _on_device_change)

        # 关闭设置窗口时停止监听，释放 COM 资源。
        win.bind(
            "<Destroy>",
            lambda e: watcher.stop() if e.widget is win else None,
        )

        win.update_idletasks()
        w = win.winfo_width()
        h = win.winfo_height()
        x = (win.winfo_screenwidth() - w) // 2
        y = (win.winfo_screenheight() - h) // 2
        win.geometry(f"+{x}+{y}")

    root.after(0, _create)


def _start_tray_icon(
    tk_root: tk.Tk,
    exit_event: threading.Event,
    recorder: "PushToTalkRecorder",
):
    # Windows-only tray icon with a simple "Exit" menu.
    if sys.platform != "win32":
        return True, None
    try:
        import pystray
        from PIL import Image, ImageDraw
    except Exception as exc:
        logger.warning(f"Tray icon unavailable (pystray/Pillow not installed): {exc}")
        return False, None

    def _load_icon(admin: bool = False):
        icon_path = _resource_path("UI/assets/IconTaskTray.png")
        if icon_path.exists():
            base = Image.open(icon_path).convert("RGBA")
        else:
            base = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            draw = ImageDraw.Draw(base)
            points = [
                (1, 1),
                (1, 14),
                (6, 9),
                (9, 12),
                (11, 10),
                (6, 5),
            ]
            draw.polygon(points, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
        if admin:
            # Green badge in the bottom-right corner to signal elevated privileges.
            w, h = base.size
            r = max(3, min(w, h) // 4)
            margin = max(1, r // 4)
            x1, y1 = w - margin, h - margin
            x0, y0 = x1 - 2 * r, y1 - 2 * r
            draw = ImageDraw.Draw(base)
            draw.ellipse(
                (x0 - 1, y0 - 1, x1 + 1, y1 + 1),
                fill=(0, 0, 0, 255),
            )
            draw.ellipse(
                (x0, y0, x1, y1),
                fill=(78, 201, 89, 255),
            )
        return base

    def _on_settings(icon, item):
        _open_settings(tk_root, exit_event, recorder)

    def _on_exit(icon, item):
        exit_event.set()
        icon.stop()

    is_admin = _is_admin()
    title = "VoiceTyper (Admin)" if is_admin else "VoiceTyper"
    try:
        icon = pystray.Icon(
            "VoiceTyper",
            _load_icon(admin=is_admin),
            title,
            menu=pystray.Menu(
                pystray.MenuItem("Settings", _on_settings),
                pystray.MenuItem("Exit", _on_exit),
            ),
        )
        thread = threading.Thread(target=icon.run, daemon=True)
        thread.start()
    except Exception as exc:
        logger.exception(f"Tray icon failed to start: {exc}")
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
            logger.info("开始加载模型...")
            self._model = SenseVoiceSmallEngine(
                strip_trailing_period=self._strip_trailing_period,
                quantized=False,
            )
            logger.info("模型加载完成")
        except Exception as exc:
            logger.exception(f"模型加载失败: {exc}")
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
                logger.warning("模型尚未加载完成，请稍候...")
                return
            if self._model is None:
                logger.error("模型为 None，无法开始录音")
                return
            self._recording = True
            self._record_start_time = time.time()
        try:
            self._ui.show()
            self._recorder.start()
            logger.debug(f"开始录音 (record_start_time={self._record_start_time})")
        except Exception as exc:
            logger.exception(f"启动录音失败: {exc}")
            self._recording = False

    def _stop_recording(self) -> None:
        try:
            self._ui.hide()
            pcm16 = self._recorder.stop()
            dur_base = self._record_start_time if self._record_start_time > 0 else self._last_down_time
            dur = max(0.0, time.time() - dur_base)
            logger.debug(f"停止录音，时长: {dur:.1f}s，数据大小: {len(pcm16) if pcm16 else 0} bytes")
            if not pcm16:
                logger.warning("录音为空")
                return
            logger.info(f"识别中... ({dur:.1f}s)")
            threading.Thread(
                target=self._transcribe_and_type, args=(pcm16,), daemon=True
            ).start()
        except Exception as exc:
            logger.exception(f"停止录音失败: {exc}")

    def _transcribe_and_type(self, pcm16: bytes) -> None:
        with self._transcribe_lock:
            if self._model is None:
                logger.error("模型未就绪，跳过识别")
                return
            try:
                logger.debug("开始语音识别...")
                text = self._model.transcribe(pcm16, self._config.rate).strip()
                logger.debug(f"识别原始结果: '{text}'")
            except Exception as exc:
                logger.exception(f"识别失败: {exc}")
                return
            if not text:
                logger.debug("识别结果为空")
                return
            if not _is_meaningful_text(text):
                logger.debug(f"忽略无意义结果: {text}")
                return
            try:
                self._ui.set_clipboard(text)
                _send_text(text)
                logger.info(f"识别结果: {text}")
            except Exception as exc:
                logger.exception(f"输入文本失败: {exc}")

    def on_press(self, key) -> None:
        try:
            if key == pynput_keyboard.Key.shift_l:
                if self._shift_down:
                    return
                self._shift_down = True
                logger.debug("Left Shift 按下")
            elif key == pynput_keyboard.Key.cmd_l:
                if self._win_down:
                    return
                self._win_down = True
                logger.debug("Left Win 按下")
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
                logger.debug(f"快捷键激活，启动 {self._hold_s}s 计时器")
                self._start_timer = threading.Timer(self._hold_s, self._start_recording)
                self._start_timer.daemon = True
                self._start_timer.start()
        except Exception as exc:
            logger.exception(f"按键处理异常: {exc}")

    def on_release(self, key):
        try:
            if key == pynput_keyboard.Key.shift_l:
                self._shift_down = False
                logger.debug("Left Shift 释放")
            elif key == pynput_keyboard.Key.cmd_l:
                self._win_down = False
                logger.debug("Left Win 释放")
            else:
                return True
            with self._state_lock:
                if self._start_timer and self._start_timer.is_alive() and not self._hotkey_active():
                    try:
                        self._start_timer.cancel()
                        logger.debug("取消录音计时器")
                    except Exception as exc:
                        logger.exception(f"取消计时器失败: {exc}")
                    self._start_timer = None
                if not self._recording:
                    return True
                if self._hotkey_active():
                    return True
                self._recording = False
                logger.debug("快捷键释放，准备停止录音")
            self._stop_recording()
            return True
        except Exception as exc:
            logger.exception(f"按键释放处理异常: {exc}")
            return True

    def run(self, tray_icon) -> None:
        logger.info("=== VoiceTyper 启动 ===")
        self._ui.show(text="⏳ 模型加载中...")
        threading.Thread(target=self.load_model, daemon=True).start()

        listener = pynput_keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        listener.start()
        logger.info("键盘监听器已启动")

        def _sigint_handler(signum, frame) -> None:
            logger.info("收到 SIGINT 信号，准备退出")
            self._exit_event.set()

        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except Exception as exc:
            logger.warning(f"无法设置 SIGINT 处理器: {exc}")

        try:
            self._ui.run(self._exit_event)
        except KeyboardInterrupt:
            logger.info("收到 KeyboardInterrupt，准备退出")
            self._exit_event.set()
        except Exception as exc:
            logger.exception(f"UI 运行异常: {exc}")
        finally:
            logger.info("开始清理资源...")
            if self._recording:
                try:
                    self._recorder.stop()
                    logger.debug("已停止录音")
                except Exception as exc:
                    logger.exception(f"停止录音失败: {exc}")
            try:
                listener.stop()
                logger.debug("已停止键盘监听器")
            except Exception as exc:
                logger.exception(f"停止监听器失败: {exc}")
            if tray_icon:
                try:
                    tray_icon.stop()
                    logger.debug("已停止托盘图标")
                except Exception as exc:
                    logger.exception(f"停止托盘图标失败: {exc}")
            try:
                self._recorder.close()
                logger.debug("已关闭录音器")
            except Exception as exc:
                logger.exception(f"关闭录音器失败: {exc}")
            logger.info("=== VoiceTyper 已退出 ===")


def main() -> None:
    hold_ms = 300
    strip_trailing_period = True
    exit_event = threading.Event()

    if SHOW_RESOURCE_USAGE:
        ResourceMonitor(interval=1.0, exit_event=exit_event).start()

    logger.info("=== UI Push-to-Talk Demo ===")
    logger.info(
        "Hold left Shift+Win to record, release to transcribe and type. Use tray menu to exit."
    )

    # 解析启动设备：恢复上次选择（按设备名匹配，index 不稳定），否则用系统默认。
    # 该逻辑与设置里的 picker 共用 UI.audio_devices，避免两处重复。
    startup = _audio_devices().resolve_startup_device()
    device_index = startup.index
    if startup.saved_name and startup.available:
        logger.info(f"已恢复输入设备: [{device_index}] {startup.saved_name}")
    elif startup.saved_name:
        logger.warning(f'已保存设备 "{startup.saved_name}" 不可用，回退默认设备')
    elif startup.default_name:
        logger.info(f"默认麦克风设备: [{device_index}] {startup.default_name}")
    else:
        logger.info("默认麦克风设备: 未知")

    config = RecorderConfig()
    recorder = PushToTalkRecorder(device_index=device_index, config=config)
    # 启动时后台探测一次输入格式，把首次按 PTT 时的开流延迟与可能的回退握手
    # 摊到模型加载/UI 初始化的等待里，避免开头丢音。
    recorder.probe_format(async_=True)
    ui = OverlayUI()
    tray_ok, tray_icon = _start_tray_icon(ui._root, exit_event, recorder)
    if not tray_ok:
        logger.error("Tray icon unavailable on Windows; aborting startup.")
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
