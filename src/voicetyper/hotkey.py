"""Hotkey representation, parsing, and pynput/Win32 matching.

A hotkey is an unordered set of tokens, e.g. ``shift_l + cmd_l``, ``f13``,
``ctrl_l + space``. Tokens are stable, lower-case strings — both presets
listed in the UI and a captured custom combo round-trip through this format.

Serialized form is ``"+"``-joined tokens in canonical order (modifiers first,
then the trigger key). That ordering is purely cosmetic; equality is
set-based.

Token vocabulary
----------------
Modifiers: ``shift_l shift_r ctrl_l ctrl_r alt_l alt_r cmd_l cmd_r``
Specials:  ``space tab enter esc backspace caps_lock f1..f24``
Chars:     single printable characters (``a``, ``b``, ``;`` …)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)

DEFAULT_HOTKEY = "shift_l+cmd_l"

# Tokens that count as modifiers — used for canonical sort order and to detect
# "modifier-only" hotkeys (where every token is a modifier).
_MODIFIERS: FrozenSet[str] = frozenset({
    "shift_l", "shift_r", "ctrl_l", "ctrl_r",
    "alt_l", "alt_r", "cmd_l", "cmd_r",
})

# Stable display order for serialization (modifiers in this order, then trigger).
_MODIFIER_ORDER = (
    "ctrl_l", "ctrl_r", "alt_l", "alt_r",
    "shift_l", "shift_r", "cmd_l", "cmd_r",
)

# pynput Key.<name> → token. Only names we accept; everything else falls through
# to ``KeyCode(char=...)`` handling below.
_PYNPUT_KEY_TO_TOKEN = {
    "shift_l": "shift_l", "shift": "shift_l", "shift_r": "shift_r",
    "ctrl_l": "ctrl_l", "ctrl": "ctrl_l", "ctrl_r": "ctrl_r",
    "alt_l": "alt_l", "alt": "alt_l", "alt_r": "alt_r", "alt_gr": "alt_r",
    "cmd_l": "cmd_l", "cmd": "cmd_l", "cmd_r": "cmd_r",
    "space": "space", "tab": "tab", "enter": "enter", "esc": "esc",
    "backspace": "backspace", "caps_lock": "caps_lock",
}
for _i in range(1, 25):
    _PYNPUT_KEY_TO_TOKEN[f"f{_i}"] = f"f{_i}"

# Win32 virtual-key codes for physical-state checks (GetAsyncKeyState).
_VK_TABLE = {
    "shift_l": 0xA0, "shift_r": 0xA1,
    "ctrl_l": 0xA2, "ctrl_r": 0xA3,
    "alt_l": 0xA4, "alt_r": 0xA5,
    "cmd_l": 0x5B, "cmd_r": 0x5C,
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "esc": 0x1B,
    "backspace": 0x08, "caps_lock": 0x14,
}
for _i in range(1, 25):
    _VK_TABLE[f"f{_i}"] = 0x6F + _i  # F1=0x70 ... F24=0x87

# Curated preset list shown in the Settings dropdown. Order matters for UI.
PRESETS: tuple[tuple[str, str], ...] = (
    ("Left Shift + Left Win", "shift_l+cmd_l"),
    ("Left Ctrl + Space", "ctrl_l+space"),
    ("Right Alt", "alt_r"),
    ("F13", "f13"),
    ("CapsLock", "caps_lock"),
    ("Left Ctrl + Left Shift", "ctrl_l+shift_l"),
    ("Left Alt + Space", "alt_l+space"),
)


def _normalize_token(raw: str) -> str:
    token = raw.strip().lower().replace("-", "_")
    aliases = {
        "win": "cmd_l", "win_l": "cmd_l", "win_r": "cmd_r",
        "lwin": "cmd_l", "rwin": "cmd_r",
        "lshift": "shift_l", "rshift": "shift_r",
        "lctrl": "ctrl_l", "rctrl": "ctrl_r",
        "lalt": "alt_l", "ralt": "alt_r", "alt_gr": "alt_r",
        "shift": "shift_l", "ctrl": "ctrl_l", "alt": "alt_l",
        "escape": "esc", "return": "enter",
    }
    return aliases.get(token, token)


def _is_valid_token(token: str) -> bool:
    if token in _MODIFIERS:
        return True
    if token in _VK_TABLE:
        return True
    # Single printable char (covers letters/digits/punct typed by the user).
    return len(token) == 1 and token.isprintable() and not token.isspace()


@dataclass(frozen=True)
class Hotkey:
    """Order-independent set of tokens that together arm the hotkey."""

    tokens: FrozenSet[str]

    @classmethod
    def parse(cls, spec: str) -> "Hotkey":
        if not spec or not spec.strip():
            raise ValueError("empty hotkey spec")
        raw_tokens = [_normalize_token(p) for p in spec.split("+") if p.strip()]
        if not raw_tokens:
            raise ValueError(f"no tokens in {spec!r}")
        bad = [t for t in raw_tokens if not _is_valid_token(t)]
        if bad:
            raise ValueError(f"unsupported token(s): {bad}")
        return cls(frozenset(raw_tokens))

    @classmethod
    def from_tokens(cls, tokens) -> "Hotkey":
        normalized = [_normalize_token(t) for t in tokens]
        bad = [t for t in normalized if not _is_valid_token(t)]
        if bad:
            raise ValueError(f"unsupported token(s): {bad}")
        if not normalized:
            raise ValueError("empty token set")
        return cls(frozenset(normalized))

    @property
    def is_modifier_only(self) -> bool:
        """All tokens are modifiers (e.g. ``shift_l + cmd_l``)."""
        return bool(self.tokens) and self.tokens.issubset(_MODIFIERS)

    def serialize(self) -> str:
        """Stable string form: modifiers in canonical order, then the trigger."""
        mods = [t for t in _MODIFIER_ORDER if t in self.tokens]
        rest = sorted(t for t in self.tokens if t not in _MODIFIERS)
        return "+".join(mods + rest)

    def display(self) -> str:
        """Human-readable label, e.g. ``"Ctrl L + Space"``."""
        pretty = {
            "shift_l": "Shift L", "shift_r": "Shift R",
            "ctrl_l": "Ctrl L", "ctrl_r": "Ctrl R",
            "alt_l": "Alt L", "alt_r": "Alt R",
            "cmd_l": "Win L", "cmd_r": "Win R",
            "space": "Space", "tab": "Tab", "enter": "Enter",
            "esc": "Esc", "backspace": "Backspace", "caps_lock": "CapsLock",
        }
        mods = [pretty.get(t, t) for t in _MODIFIER_ORDER if t in self.tokens]
        rest = [pretty.get(t, t.upper() if len(t) == 1 else t.title())
                for t in sorted(self.tokens) if t not in _MODIFIERS]
        return " + ".join(mods + rest)

    def vks(self) -> list[int]:
        """Win32 virtual-key codes for physical-state verification.

        Tokens without a known VK (rare; e.g. exotic chars) are skipped — the
        caller treats an empty list as "can't physically verify, trust pynput".
        """
        out: list[int] = []
        for t in self.tokens:
            vk = _VK_TABLE.get(t)
            if vk is None and len(t) == 1:
                vk = _vk_for_char(t)
            if vk is not None:
                out.append(vk)
        return out

    # ── matching against pynput events ────────────────────────────

    @staticmethod
    def token_for_pynput(key) -> Optional[str]:
        """Convert a pynput ``Key`` / ``KeyCode`` into our token, or None.

        ``None`` means "not a key we care about" — the listener simply ignores it.
        """
        # pynput.keyboard.Key has .name; KeyCode has .char
        name = getattr(key, "name", None)
        if name is not None:
            return _PYNPUT_KEY_TO_TOKEN.get(name)
        char = getattr(key, "char", None)
        if isinstance(char, str) and len(char) == 1 and char.isprintable():
            return char.lower()
        return None


def _vk_for_char(ch: str) -> Optional[int]:
    """Resolve a single printable char to a Win32 VK via ``VkKeyScanW``.

    Returns the low byte of the scan result (the VK), ignoring the high-byte
    shift-state hint — physical-state checks only care about the key itself,
    not which modifier produced the character.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        result = ctypes.windll.user32.VkKeyScanW(ord(ch))
        if result == -1:
            return None
        return result & 0xFF
    except Exception:
        return None
