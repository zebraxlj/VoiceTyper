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
from typing import Any, Optional

from pynput import keyboard as pynput_keyboard

from voicetyper import PushToTalkRecorder, RecorderConfig
from voicetyper import devices, device_watch
from voicetyper.app_config import AppConfig, MODE_HOLD, MODE_TOGGLE, load as load_app_config, save as save_app_config
from voicetyper.hotkey import PRESETS as HOTKEY_PRESETS, Hotkey
from voicetyper.models import SenseVoiceSmallEngine
from voicetyper.monitor import ResourceMonitor

# 启动期临时读一次配置以决定日志格式。后续改配置不会回溯影响日志（重启生效）。
_BOOT_CFG = load_app_config()

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

# 控制台处理器：根据配置切换格式和级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if _BOOT_CFG.verbose_console else logging.INFO)
console_handler.setFormatter(_VERBOSE_FMT if _BOOT_CFG.verbose_console else _SIMPLE_FMT)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger("voicetyper.app")


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


def _capture_hotkey(parent: tk.Misc) -> Optional[str]:
    """Modal listener that captures one keyboard combo and returns it serialized.

    打开一个置顶的小窗口，把目前正按下的 token 集合实时显示出来；
    所有键松开（即出现一次"全部释放"边沿）时把这次按下过的最大 token
    集合返回。ESC 取消，返回 None。
    """
    dlg = tk.Toplevel(parent)
    dlg.title("Capture Hotkey")
    dlg.resizable(False, False)
    dlg.attributes("-topmost", True)
    dlg.configure(bg="#1e1e1e")
    dlg.transient(parent)
    dlg.grab_set()

    pad = tk.Frame(dlg, bg="#1e1e1e", padx=24, pady=18)
    pad.pack()
    tk.Label(
        pad,
        text="Press the keys you want to use, then release.",
        font=("Segoe UI", 10),
        fg="#cccccc",
        bg="#1e1e1e",
    ).pack(anchor="w", pady=(0, 6))
    tk.Label(
        pad,
        text="ESC to cancel.",
        font=("Segoe UI", 9),
        fg="#999999",
        bg="#1e1e1e",
    ).pack(anchor="w", pady=(0, 10))
    preview = tk.Label(
        pad,
        text="(waiting...)",
        font=("Segoe UI", 12, "bold"),
        fg="#f6f6f6",
        bg="#1e1e1e",
    )
    preview.pack(anchor="w", pady=(0, 4))

    state: dict[str, Any] = {"down": set(), "max": set(), "result": None, "done": False}

    def _finish(result: Optional[str]) -> None:
        if state["done"]:
            return
        state["done"] = True
        state["result"] = result
        try:
            listener.stop()
        except Exception:
            pass
        dlg.after(0, dlg.destroy)

    def _refresh_preview() -> None:
        if state["done"]:
            return
        tokens = state["max"] or state["down"]
        if tokens:
            try:
                preview.configure(text=Hotkey.from_tokens(tokens).display())
            except ValueError:
                preview.configure(text="(unsupported keys)")
        dlg.after(60, _refresh_preview)

    def _on_press(key) -> None:
        if state["done"]:
            return
        # ESC 取消
        if key == pynput_keyboard.Key.esc:
            _finish(None)
            return
        token = Hotkey.token_for_pynput(key)
        if token is None:
            return
        state["down"].add(token)
        # 持续累积"曾经同时按下"的最大集合：避免用户提前松开一两个键导致少记
        state["max"] = set(state["max"]) | set(state["down"])

    def _on_release(key) -> None:
        if state["done"]:
            return
        token = Hotkey.token_for_pynput(key)
        if token is None:
            return True
        state["down"].discard(token)
        if not state["down"] and state["max"]:
            try:
                hk = Hotkey.from_tokens(state["max"])
                _finish(hk.serialize())
            except ValueError:
                _finish(None)
        return True

    from pynput import keyboard as _kb_mod  # local import keeps top imports tidy
    listener = _kb_mod.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    dlg.bind("<Escape>", lambda e: _finish(None))
    _refresh_preview()
    dlg.update_idletasks()
    w = dlg.winfo_width()
    h = dlg.winfo_height()
    sw = dlg.winfo_screenwidth()
    sh = dlg.winfo_screenheight()
    dlg.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
    parent.wait_window(dlg)
    return state["result"]


def _build_hotkey_section(
    frame: tk.Frame, win: tk.Toplevel, session: "PushToTalkSession",
) -> None:
    """Render the hotkey + mode + hold_ms controls inside the Settings window."""

    cfg = load_app_config()
    current_hotkey = cfg.hotkey

    # ── Hotkey ──
    tk.Label(
        frame,
        text="Hotkey",
        font=("Segoe UI", 10, "bold"),
        fg="#cccccc",
        bg="#1e1e1e",
    ).pack(anchor="w", pady=(8, 4))

    preset_labels = [label for label, _ in HOTKEY_PRESETS]
    preset_serial = {label: serial for label, serial in HOTKEY_PRESETS}
    serial_to_preset = {serial: label for label, serial in HOTKEY_PRESETS}
    custom_label = "Custom..."
    dropdown_values = preset_labels + [custom_label]

    def _initial_label() -> str:
        if current_hotkey in serial_to_preset:
            return serial_to_preset[current_hotkey]
        try:
            return f"Custom: {Hotkey.parse(current_hotkey).display()}"
        except ValueError:
            return preset_labels[0]

    hotkey_var = tk.StringVar(value=_initial_label())
    hotkey_combo = ttk.Combobox(
        frame,
        textvariable=hotkey_var,
        values=dropdown_values,
        state="readonly",
        width=42,
    )
    hotkey_combo.pack(anchor="w", pady=(0, 4))
    hotkey_status = tk.Label(
        frame,
        text="",
        font=("Segoe UI", 9),
        fg="#999999",
        bg="#1e1e1e",
    )
    hotkey_status.pack(anchor="w", pady=(0, 6))

    # 把"当前热键序列化值"绑到 win 上，给保存按钮取
    win._hotkey_serial = current_hotkey  # type: ignore[attr-defined]

    def _set_serial(serial: str) -> None:
        win._hotkey_serial = serial  # type: ignore[attr-defined]
        if serial in serial_to_preset:
            hotkey_var.set(serial_to_preset[serial])
        else:
            try:
                hotkey_var.set(f"Custom: {Hotkey.parse(serial).display()}")
            except ValueError:
                hotkey_var.set(preset_labels[0])
                win._hotkey_serial = preset_serial[preset_labels[0]]  # type: ignore[attr-defined]

    def _on_hotkey_change(_event=None) -> None:
        label = hotkey_var.get()
        if label == custom_label:
            captured = _capture_hotkey(win)
            if captured:
                _set_serial(captured)
                hotkey_status.configure(text=f"Captured: {Hotkey.parse(captured).display()}", fg="#4ec959")
            else:
                # 取消 → 恢复显示当前 serial
                _set_serial(win._hotkey_serial)  # type: ignore[attr-defined]
                hotkey_status.configure(text="Capture canceled", fg="#999999")
            return
        if label in preset_serial:
            _set_serial(preset_serial[label])
            hotkey_status.configure(text="", fg="#999999")

    hotkey_combo.bind("<<ComboboxSelected>>", _on_hotkey_change)

    # ── Mode (Hold / Toggle) ──
    tk.Label(
        frame,
        text="Mode",
        font=("Segoe UI", 10, "bold"),
        fg="#cccccc",
        bg="#1e1e1e",
    ).pack(anchor="w", pady=(8, 4))

    mode_var = tk.StringVar(value=cfg.mode)
    mode_frame = tk.Frame(frame, bg="#1e1e1e")
    mode_frame.pack(anchor="w", pady=(0, 6))
    for value, label in ((MODE_HOLD, "Hold (press and hold)"), (MODE_TOGGLE, "Toggle (press once to start, again to stop)")):
        tk.Radiobutton(
            mode_frame,
            text=label,
            variable=mode_var,
            value=value,
            font=("Segoe UI", 9),
            fg="#cccccc",
            bg="#1e1e1e",
            selectcolor="#1e1e1e",
            activebackground="#1e1e1e",
            activeforeground="#ffffff",
        ).pack(anchor="w")

    # ── Hold debounce (ms) ──
    hold_row = tk.Frame(frame, bg="#1e1e1e")
    hold_row.pack(anchor="w", pady=(8, 4))
    tk.Label(
        hold_row,
        text="Hold debounce (ms): ",
        font=("Segoe UI", 10),
        fg="#cccccc",
        bg="#1e1e1e",
    ).pack(side="left")
    hold_var = tk.StringVar(value=str(cfg.hold_ms))
    hold_entry = tk.Entry(hold_row, textvariable=hold_var, width=8, font=("Segoe UI", 10))
    hold_entry.pack(side="left")
    tk.Label(
        frame,
        text="Hold mode only — how long the hotkey must be held before recording starts.",
        font=("Segoe UI", 9),
        fg="#999999",
        bg="#1e1e1e",
    ).pack(anchor="w", pady=(0, 8))

    # ── Apply ──
    apply_status = tk.Label(
        frame,
        text="",
        font=("Segoe UI", 9),
        fg="#999999",
        bg="#1e1e1e",
    )

    def _on_apply() -> None:
        try:
            hold_ms_val = int(hold_var.get())
        except ValueError:
            apply_status.configure(text="Hold debounce must be an integer (ms).", fg="#e05252")
            return
        new_cfg = AppConfig(
            hotkey=win._hotkey_serial,  # type: ignore[attr-defined]
            mode=mode_var.get(),
            hold_ms=hold_ms_val,
            strip_trailing_period=cfg.strip_trailing_period,
            verbose_console=cfg.verbose_console,
            show_resource_usage=cfg.show_resource_usage,
        ).normalized()
        save_app_config(new_cfg)
        session.update_config(new_cfg)
        apply_status.configure(
            text=f"Applied: {new_cfg.hotkey_obj().display()} ({new_cfg.mode})",
            fg="#4ec959",
        )

    tk.Button(
        frame,
        text="Apply",
        font=("Segoe UI", 10),
        command=_on_apply,
    ).pack(anchor="w", pady=(0, 4))
    apply_status.pack(anchor="w", pady=(0, 8))


def _open_settings(
    root: tk.Tk,
    exit_event: threading.Event,
    recorder: "PushToTalkRecorder",
    session: "PushToTalkSession",
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

        _build_hotkey_section(frame, win, session)

        # ── Input device picker ──────────────────────────────
        tk.Label(
            frame,
            text="Input Device",
            font=("Segoe UI", 10, "bold"),
            fg="#cccccc",
            bg="#1e1e1e",
        ).pack(anchor="w", pady=(4, 4))

        # UI 无关的设备选择逻辑（枚举/持久化/断开重连）都在 voicetyper.devices 里；
        # 这里只负责渲染成 Combobox 并把"何时刷新"接上去。正式 UI 可复用同一套。
        selector = devices.InputDeviceSelector()
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
            rescanned = recorder.rescan_devices()  # 重建 PortAudio 以发现热插拔
            selector.refresh()
            combo.configure(values=selector.labels())
            device_var.set(selector.selected_label)
            # 重扫后底层 index 可能漂移（被拔设备在列表中靠前时尤甚）；把 selector
            # 按名称对齐后的最新 index 同步回 recorder，否则 recorder 仍用旧 index，
            # 下次 start() 会打开错位/失效的设备而报 -9998。录音中 rescan 会返回
            # False（不重建、index 不漂移），此时用返回值门控跳过回写。
            if rescanned and selector.selected_index != recorder.device_index:
                recorder.set_device_index(selector.selected_index)

        # 设备热插拔监听：Windows 上事件驱动，其它平台/无 comtypes 时回退到轮询。
        # 关键约束：PortAudio 的 WASAPI 把 COM 套间绑在调用线程上，且 Tk 不是线程
        # 安全的——所以重扫和 UI 刷新必须都在主线程。COM 回调只置一个标志位，真正
        # 的重扫由主线程轮询循环 _poll_devices 触发，避免临时线程导致的枚举不全与
        # 访问越界崩溃（0xC0000005）。
        device_dirty = threading.Event()
        watcher = device_watch.DeviceChangeWatcher()
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
    session: "PushToTalkSession",
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

    def _load_icon(admin: bool = False, paused: bool = False):
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
        if paused:
            # 右下角圆形黄色徽标 + 内嵌"‖"，与 admin 绿点同位但更大；不遮挡原 icon 主体。
            w, h = base.size
            r = max(6, min(w, h) // 3)         # 半径取图标短边 1/3，醒目但不压制图案
            margin = max(1, r // 5)
            x1, y1 = w - margin, h - margin
            x0, y0 = x1 - 2 * r, y1 - 2 * r
            yellow = (255, 204, 0, 255)
            outline = (0, 0, 0, 255)
            draw = ImageDraw.Draw(base)
            # 黑色描边底，避免与浅色任务栏融成一片。
            draw.ellipse(
                (x0 - 1, y0 - 1, x1 + 1, y1 + 1),
                fill=outline,
            )
            draw.ellipse((x0, y0, x1, y1), fill=yellow)
            # 圆内黑色暂停双竖条。
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            bar_w = max(1, r // 4)
            bar_h = int(r * 0.95)
            gap = max(1, r // 3)
            draw.rectangle(
                (cx - gap // 2 - bar_w, cy - bar_h // 2,
                 cx - gap // 2,         cy + bar_h // 2),
                fill=outline,
            )
            draw.rectangle(
                (cx + gap // 2,         cy - bar_h // 2,
                 cx + gap // 2 + bar_w, cy + bar_h // 2),
                fill=outline,
            )
        return base

    def _on_settings(icon, item):
        _open_settings(tk_root, exit_event, recorder, session)

    def _on_pause(icon, item):
        # 切换暂停状态；菜单项的 checked 回调会读取最新状态自动刷新勾选标记。
        if session.is_paused():
            session.resume()
        else:
            session.pause()
        # 同步刷新菜单勾选 + 托盘图标（暂停时叠加黄色 ‖ 标识）。
        icon.icon = _load_icon(admin=is_admin, paused=session.is_paused())
        icon.title = (
            f"{title} — Paused" if session.is_paused() else title
        )
        icon.update_menu()

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
                pystray.MenuItem(
                    "Pause (disable hotkey)",
                    _on_pause,
                    checked=lambda item: session.is_paused(),
                ),
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
    def __init__(
        self,
        ui: OverlayUI,
        recorder: PushToTalkRecorder,
        config: RecorderConfig,
        exit_event: threading.Event,
        app_config: AppConfig,
    ) -> None:
        self._ui = ui
        self._recorder = recorder
        self._config = config
        self._exit_event = exit_event
        self._app_config = app_config
        self._hotkey = app_config.hotkey_obj()
        self._mode = app_config.mode
        self._hold_s = max(0.0, app_config.hold_ms / 1000.0)

        self._model: Optional[SenseVoiceSmallEngine] = None
        self._model_ready = threading.Event()
        self._transcribe_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._recording = False
        # 当前实际按下的 token 集合；hotkey 触发条件 = self._hotkey.tokens.issubset(self._down_tokens)
        self._down_tokens: set[str] = set()
        self._last_down_time = 0.0
        self._record_start_time = 0.0
        self._start_timer: Optional[threading.Timer] = None
        self._paused = False
        # 进入 toggle 模式后，每次组合键的"上升沿"才是触发点；同一次按住期间只触发一次
        self._toggle_armed = True

    def is_paused(self) -> bool:
        return self._paused

    def update_config(self, app_config: AppConfig) -> None:
        """Hot-apply config changes from the Settings window.

        切换 hotkey/mode 时若处于录音/计时中，先安全停下来再切——避免新 hotkey
        生效后旧的"释放即停止"逻辑找不到对应 token，永远停不下来。
        """
        with self._state_lock:
            cancel_timer = self._start_timer
            self._start_timer = None
            was_recording = self._recording
            self._recording = False
            self._down_tokens.clear()
            self._toggle_armed = True
            self._app_config = app_config
            self._hotkey = app_config.hotkey_obj()
            self._mode = app_config.mode
            self._hold_s = max(0.0, app_config.hold_ms / 1000.0)
        if cancel_timer and cancel_timer.is_alive():
            try:
                cancel_timer.cancel()
            except Exception:
                pass
        if was_recording:
            try:
                self._recorder.stop()
            except Exception as exc:
                logger.exception(f"应用配置时停止录音失败: {exc}")
            self._ui.hide()
        logger.info(
            f"配置已更新: hotkey={self._hotkey.serialize()} mode={self._mode} hold_ms={app_config.hold_ms}"
        )

    def pause(self) -> None:
        # 暂停热键监听：开发版与打包版可同时运行，避免双触发。
        # 若正在录音/计时，立刻终止；按键状态清零，避免恢复后误触发。
        with self._state_lock:
            if self._paused:
                return
            self._paused = True
            if self._start_timer and self._start_timer.is_alive():
                try:
                    self._start_timer.cancel()
                except Exception:
                    pass
                self._start_timer = None
            was_recording = self._recording
            self._recording = False
            self._down_tokens.clear()
            self._toggle_armed = True
        if was_recording:
            try:
                self._recorder.stop()
            except Exception as exc:
                logger.exception(f"暂停时停止录音失败: {exc}")
            self._ui.hide()
        logger.info("热键已暂停")

    def resume(self) -> None:
        with self._state_lock:
            if not self._paused:
                return
            self._paused = False
        logger.info("热键已恢复")

    def load_model(self) -> None:
        try:
            logger.info("开始加载模型...")
            self._model = SenseVoiceSmallEngine(
                strip_trailing_period=self._app_config.strip_trailing_period,
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
        # 所有 token 都按下 → hotkey 处于 active 状态。注意 set.issubset 在
        # tokens 为空时返回 True，但 Hotkey.parse 会拒绝空热键，所以无需额外保护。
        return self._hotkey.tokens.issubset(self._down_tokens)

    def _is_key_physically_pressed(self, vk: int) -> bool:
        if sys.platform != "win32":
            return True
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)

    def _sync_modifier_state(self) -> None:
        # Windows 偶尔会吞掉 release 事件（窗口切换、UAC 弹窗后），导致 token 残留。
        # 用 GetAsyncKeyState 二次校验：物理上不在按下状态的 token 必须从集合里清掉。
        if sys.platform != "win32":
            return
        for token in list(self._down_tokens):
            for vk in self._token_vks(token):
                if not self._is_key_physically_pressed(vk):
                    self._down_tokens.discard(token)
                    break

    @staticmethod
    def _token_vks(token: str) -> list[int]:
        # 复用 Hotkey 的 vks() 实现，但只针对单个 token。
        return Hotkey.from_tokens([token]).vks()

    def all_keys_physically_pressed(self) -> bool:
        # Win32 兜底校验：所有热键 token 都仍在物理按下状态。VK 解析不出来时跳过。
        if sys.platform != "win32":
            return True
        for vk in self._hotkey.vks():
            if not self._is_key_physically_pressed(vk):
                return False
        return True

    def _start_recording(self) -> None:
        with self._state_lock:
            self._sync_modifier_state()
            if not self._hotkey_active() or self._recording:
                return
            # Hold 模式：计时器到点时再次确认所有键仍在物理按下状态，防止"按下→
            # 立刻松开"的快闪误触发；Toggle 模式跳过此校验，因为按一下就该触发。
            if self._mode == MODE_HOLD and not self.all_keys_physically_pressed():
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
        if self._paused:
            return
        try:
            token = Hotkey.token_for_pynput(key)
            if token is None or token not in self._hotkey.tokens:
                return
            if token in self._down_tokens:
                return  # 操作系统的自动重复，忽略
            self._down_tokens.add(token)
            logger.debug(f"按下: {token}")

            with self._state_lock:
                self._sync_modifier_state()
                if not self._hotkey_active():
                    return
                if self._mode == MODE_TOGGLE:
                    # Toggle：每次组合键的"上升沿"才触发；同一次按住期间已触发过则跳过
                    if not self._toggle_armed:
                        return
                    self._toggle_armed = False
                    if self._recording:
                        self._recording = False
                        stop_now = True
                    else:
                        stop_now = False
                else:
                    # Hold：装计时器，hold_s 后没释放就开始录音
                    stop_now = False
                    if self._recording:
                        return
                    if self._start_timer and self._start_timer.is_alive():
                        return
                    self._last_down_time = time.time()
                    logger.debug(f"快捷键激活，启动 {self._hold_s}s 计时器")
                    self._start_timer = threading.Timer(self._hold_s, self._start_recording)
                    self._start_timer.daemon = True
                    self._start_timer.start()

            # Toggle 模式的转场出锁后再做（涉及 PyAudio + UI，不能持锁）
            if self._mode == MODE_TOGGLE and self._hotkey_active():
                if stop_now:
                    self._stop_recording()
                else:
                    threading.Thread(target=self._start_recording, daemon=True).start()
        except Exception as exc:
            logger.exception(f"按键处理异常: {exc}")

    def on_release(self, key):
        if self._paused:
            return True
        try:
            token = Hotkey.token_for_pynput(key)
            if token is None or token not in self._hotkey.tokens:
                return True
            self._down_tokens.discard(token)
            logger.debug(f"释放: {token}")

            with self._state_lock:
                # Toggle：组合键全部松开后重新 arm，下一次按下才会再次触发
                if self._mode == MODE_TOGGLE:
                    if not self._hotkey_active():
                        self._toggle_armed = True
                    return True

                # Hold 模式
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
    app_cfg = load_app_config()
    exit_event = threading.Event()

    if app_cfg.show_resource_usage:
        ResourceMonitor(interval=1.0, exit_event=exit_event).start()

    logger.info("=== UI Push-to-Talk Demo ===")
    logger.info(
        f"Hotkey: {app_cfg.hotkey_obj().display()} | Mode: {app_cfg.mode} | Hold: {app_cfg.hold_ms}ms"
    )

    # 解析启动设备：恢复上次选择（按设备名匹配，index 不稳定），否则用系统默认。
    # 该逻辑与设置里的 picker 共用 voicetyper.devices，避免两处重复。
    startup = devices.resolve_startup_device()
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
    session = PushToTalkSession(
        ui=ui,
        recorder=recorder,
        config=config,
        exit_event=exit_event,
        app_config=app_cfg,
    )
    tray_ok, tray_icon = _start_tray_icon(ui._root, exit_event, recorder, session)
    if not tray_ok:
        logger.error("Tray icon unavailable on Windows; aborting startup.")
        recorder.close()
        return

    session.run(tray_icon=tray_icon)


if __name__ == "__main__":
    main()
