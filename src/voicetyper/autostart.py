"""Per-user autostart via the Windows ``HKCU\\...\\Run`` registry key.

The registry value IS the single source of truth for "is autostart on" — there
is no mirrored flag in :mod:`voicetyper.settings`, so the UI can never drift out
of sync with reality. The registered command runs at logon with **normal**
privileges (``HKCU\\Run`` does not trigger UAC elevation); "start elevated at
boot" would need Task Scheduler and is intentionally out of scope here.

Windows-only. Mirrors :mod:`voicetyper.device_watch`'s degrade-off-platform
contract: on non-Windows, :func:`is_supported` returns ``False`` and every other
call is a harmless no-op so consumers need no platform branches of their own.
"""

import logging
import sys

try:  # Windows-only stdlib module.
    import winreg
except ImportError:  # pragma: no cover - exercised only off-Windows
    winreg = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

APP_NAME = "VoiceTyper"
"""Registry value name written under the Run key."""

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def enable(command: str) -> None:
    """Register ``command`` to run at logon (overwrites any existing entry).

    No-op off Windows. ``command`` should be the fully-quoted launch command line
    (see the demo's ``_current_launch_command``).
    """
    if not is_supported():
        logger.debug("开机启动不受支持（非 Windows），忽略 enable()")
        return
    # CreateKey opens the key if it already exists; the Run key normally does.
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command)
    logger.info("已设置开机启动: %s", command)


def disable() -> None:
    """Remove the autostart entry. Idempotent — no error if it was never set."""
    if not is_supported():
        return
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, APP_NAME)
        logger.info("已取消开机启动")
    except FileNotFoundError:
        pass  # already absent
    except OSError as exc:
        logger.warning("取消开机启动失败: %s", exc)


def get_command() -> str | None:
    """Return the registered launch command, or ``None`` if not set / unsupported."""
    if not is_supported():
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_READ) as key:
            value, _ = winreg.QueryValueEx(key, APP_NAME)
            return value
    except FileNotFoundError:
        return None
    except OSError as exc:  # unexpected registry failure — report, don't crash caller
        logger.warning("读取开机启动项失败: %s", exc)
        return None


def is_enabled() -> bool:
    """Return ``True`` if an autostart entry for this app currently exists."""
    return get_command() is not None


def is_supported() -> bool:
    """Return ``True`` only where the Run-key mechanism exists (Windows)."""
    return sys.platform == "win32" and winreg is not None
