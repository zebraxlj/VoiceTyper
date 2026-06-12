"""UI-agnostic input-device selection model.

Holds the *logic* of an input-device picker — enumeration, the "System Default"
entry, persistence of the chosen device (by name, since PyAudio indices drift
across hot-plug), and the refresh/reconcile behaviour (hot-plug, index drift,
disconnect, reconnect).

Contains **no GUI code** and is fully cross-platform, so both the demo overlay
and a future formal UI can reuse it: a view layer renders ``entries`` /
``selected_label`` and calls ``refresh()`` / ``select()``. Hot-plug *events*
(Windows-only) live separately in :mod:`voicetyper.device_watch`.
"""

from dataclasses import dataclass
from typing import Optional

from .audio import AudioDeviceResolver
from . import settings

# settings.json key under which the preferred device *name* is stored.
_DEVICE_KEY = "audio_device_name"

# Identity of the synthetic "System Default" entry. Real devices use their
# name as identity; this sentinel keeps the default distinct from a device that
# merely happens to share the default's name.
DEFAULT_IDENTITY = "__default__"

# Suffix appended to a selection whose device has been unplugged.
DISCONNECTED_SUFFIX = " (已断开)"


@dataclass(frozen=True)
class DeviceEntry:
    """One selectable input device (UI-agnostic)."""

    label: str  # display text, e.g. "System Default (Realtek)" or "[3] USB Mic"
    index: Optional[int]  # PyAudio device index; None => use system default
    identity: str  # DEFAULT_IDENTITY or the device name (stable across hot-plug)
    available: bool = True  # False for a synthetic "disconnected" placeholder


def enumerate_devices() -> list[DeviceEntry]:
    """Enumerate current input devices, prefixed with a "System Default" entry."""
    with AudioDeviceResolver() as resolver:
        # 只用过滤+去重后的端点列表（Windows 上即系统面板的 WASAPI 输入设备）。
        # 不回退到 list_inputs()：那是跨所有 host API 的原始列表，会冒出大量
        # 同名不同 index 的重复项，且其中很多打开即报错（code -1）。
        devices = resolver.list_user_endpoints()
        default_info = resolver.default_input()

    label = "System Default"
    if default_info:
        label = f"System Default ({default_info['name']})"
    entries = [DeviceEntry(label=label, index=None, identity=DEFAULT_IDENTITY)]
    for d in devices:
        entries.append(
            DeviceEntry(
                label=f"[{d['index']}] {d['name']}",
                index=int(d["index"]),
                identity=d["name"],
            )
        )
    return entries


class InputDeviceSelector:
    """Stateful, UI-agnostic model backing an input-device picker.

    Typical view wiring::

        sel = InputDeviceSelector()
        sel.select_by_index(recorder.device_index)   # seed initial selection
        render(values=sel.labels(), current=sel.selected_label)

        # when the dropdown is about to open:
        sel.refresh(); render(values=sel.labels(), current=sel.selected_label)

        # when the user picks `label`:
        index = sel.select(label)
        recorder.set_device_index(index)
        sel.persist()
    """

    def __init__(self) -> None:
        self.entries: list[DeviceEntry] = []
        self.selected_label: str = ""
        self._selected_identity: str = DEFAULT_IDENTITY
        self.refresh()

    # ── queries for the view ──────────────────────────────────
    def labels(self) -> list[str]:
        return [e.label for e in self.entries]

    @property
    def disconnected(self) -> bool:
        """True when the current selection's device is not present."""
        entry = self._entry_for_label(self.selected_label)
        return entry is not None and not entry.available

    # ── selection ─────────────────────────────────────────────
    def select(self, label: str) -> Optional[int]:
        """Mark ``label`` as the current selection; return its device index.

        Returns None for "System Default" or a disconnected placeholder (both
        mean "let PyAudio use the default device"). The caller applies the
        index to the recorder and calls :meth:`persist` to remember it.
        """
        entry = self._entry_for_label(label)
        if entry is None:
            return None
        self.selected_label = entry.label
        self._selected_identity = entry.identity
        return entry.index

    def select_by_index(self, index: Optional[int]) -> None:
        """Seed the selection from a known device index (e.g. the recorder's)."""
        entry = None
        if index is not None:
            entry = next((e for e in self.entries if e.index == index), None)
        entry = entry or self.entries[0]
        self.selected_label = entry.label
        self._selected_identity = entry.identity

    # ── refresh / reconcile ───────────────────────────────────
    def refresh(self) -> None:
        """Re-enumerate devices, preserving the selection by *identity*.

        Handles, without surprising the caller:
        - hot-plug: new devices simply appear;
        - index drift: same device under a new index stays selected (matched
          by name, not by the index embedded in the label);
        - disconnect: the selected device is kept on screen with a
          "(已断开)" marker; the recorder is **not** touched;
        - reconnect: a previously-disconnected device drops its marker and is
          re-selected automatically.
        """
        prev_identity = self._selected_identity
        prev_label = self.selected_label

        self.entries = enumerate_devices()

        if prev_identity == DEFAULT_IDENTITY:
            chosen = self.entries[0]  # default label may have changed
        else:
            match = next(
                (e for e in self.entries if e.identity == prev_identity), None
            )
            if match is not None:
                chosen = match
            else:
                # Disconnected: keep showing it (marked), recorder untouched.
                base = prev_label
                if base.endswith(DISCONNECTED_SUFFIX):
                    base = base[: -len(DISCONNECTED_SUFFIX)]
                chosen = DeviceEntry(
                    label=base + DISCONNECTED_SUFFIX,
                    index=None,
                    identity=prev_identity,
                    available=False,
                )
                self.entries.append(chosen)

        self.selected_label = chosen.label
        self._selected_identity = chosen.identity

    # ── persistence ───────────────────────────────────────────
    def persist(self) -> None:
        """Remember the current selection (by name) for the next launch."""
        name = "" if self._selected_identity == DEFAULT_IDENTITY else self._selected_identity
        data = settings.load()
        data[_DEVICE_KEY] = name
        settings.save(data)

    # ── helpers ───────────────────────────────────────────────
    def _entry_for_label(self, label: str) -> Optional[DeviceEntry]:
        return next((e for e in self.entries if e.label == label), None)


@dataclass(frozen=True)
class StartupDevice:
    """Resolved input device for application startup."""

    index: Optional[int]  # device index to open the recorder with (None => default)
    saved_name: str  # the persisted preference ("" if none was saved)
    available: bool  # False only when a saved device could not be found
    default_name: Optional[str]  # current system-default device name, for logging


def resolve_startup_device() -> StartupDevice:
    """Resolve which input device to open at startup from the saved preference.

    Falls back to the system default when nothing is saved or the saved device
    is unavailable. This is the single source of truth shared with the picker
    (both match the saved *name* against the current devices).
    """
    saved = settings.load().get(_DEVICE_KEY) or ""
    with AudioDeviceResolver() as resolver:
        default_info = resolver.default_input()
        default_index = int(default_info["index"]) if default_info else None
        match = None
        if saved:
            endpoints = resolver.list_user_endpoints()
            match = next((d for d in endpoints if d["name"] == saved), None)

    default_name = default_info["name"] if default_info else None
    if saved and match:
        return StartupDevice(
            index=int(match["index"]),
            saved_name=saved,
            available=True,
            default_name=default_name,
        )
    # Saved-but-gone, or nothing saved → fall back to the system default device.
    return StartupDevice(
        index=default_index,
        saved_name=saved,
        available=not saved,
        default_name=default_name,
    )
