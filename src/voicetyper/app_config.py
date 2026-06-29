"""Application-level config persisted in ``~/.voicetyper/settings.json``.

Distinct from :mod:`voicetyper.settings`, which is the raw JSON store. This
module owns the *typed schema* for runtime preferences that previously lived
as module-level constants in the demo (hold delay, hotkey, mode, debug
toggles). Both demo and a future formal UI consume the same object.

Storage layout — values nest under ``"app"`` so the existing flat keys
(``audio_device_name``, etc.) keep working untouched.
"""

import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from . import settings
from .hotkey import DEFAULT_HOTKEY, Hotkey

logger = logging.getLogger(__name__)

_APP_KEY = "app"

MODE_HOLD = "hold"
MODE_TOGGLE = "toggle"
_VALID_MODES = (MODE_HOLD, MODE_TOGGLE)


@dataclass
class AppConfig:
    """Typed view of user-tunable preferences for the push-to-talk app."""

    hotkey: str = DEFAULT_HOTKEY
    """Serialized hotkey, e.g. ``"shift_l+cmd_l"``. See :mod:`voicetyper.hotkey`."""

    mode: str = MODE_HOLD
    """``"hold"`` (press-and-hold) or ``"toggle"`` (press once, press again to stop)."""

    hold_ms: int = 300
    """Hold-mode debounce: how long the hotkey must be held before recording starts.

    Ignored in toggle mode (transitions fire immediately on rising edge).
    """

    strip_trailing_period: bool = True
    """Strip the punctuation SenseVoice tends to append at the end of an utterance."""

    verbose_console: bool = False
    """Console log format toggle. File log is always verbose regardless."""

    show_resource_usage: bool = False
    """Spawn :class:`voicetyper.monitor.ResourceMonitor` to log CPU/RAM/GPU."""

    log_dir: str = ""
    """Custom log directory. Empty = use the app's default location.

    Read once at startup; changing it takes effect only after a restart.
    """

    def hotkey_obj(self) -> Hotkey:
        """Parse :attr:`hotkey` into a :class:`Hotkey`, falling back to default on error."""
        try:
            return Hotkey.parse(self.hotkey)
        except ValueError as exc:
            logger.warning("配置中的热键 %r 无效（%s），回退到默认 %s", self.hotkey, exc, DEFAULT_HOTKEY)
            return Hotkey.parse(DEFAULT_HOTKEY)

    def normalized(self) -> "AppConfig":
        """Return a copy with values clamped/coerced to valid ranges.

        Defensive — protects against hand-edited settings.json with bad values.
        """
        mode = self.mode if self.mode in _VALID_MODES else MODE_HOLD
        hold_ms = max(0, min(int(self.hold_ms), 5000))
        try:
            Hotkey.parse(self.hotkey)
            hotkey = self.hotkey
        except ValueError:
            hotkey = DEFAULT_HOTKEY
        return AppConfig(
            hotkey=hotkey,
            mode=mode,
            hold_ms=hold_ms,
            strip_trailing_period=bool(self.strip_trailing_period),
            verbose_console=bool(self.verbose_console),
            show_resource_usage=bool(self.show_resource_usage),
            log_dir=str(self.log_dir or "").strip(),
        )


def load() -> AppConfig:
    """Load app config from ``settings.json``; missing keys fall back to defaults."""
    raw = settings.load().get(_APP_KEY) or {}
    valid = {f.name for f in fields(AppConfig)}
    cleaned: dict[str, Any] = {k: v for k, v in raw.items() if k in valid}
    return AppConfig(**cleaned).normalized()


def save(cfg: AppConfig) -> None:
    """Persist ``cfg`` under the ``"app"`` key, preserving sibling keys (e.g. device)."""
    data = settings.load()
    data[_APP_KEY] = asdict(cfg.normalized())
    settings.save(data)
