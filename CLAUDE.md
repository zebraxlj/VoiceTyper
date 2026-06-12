# VoiceTyper

Real-time offline speech-to-text library for Windows, powered by SenseVoiceSmall (sherpa-onnx).

## Tech Stack

- Python 3.11+, uv (Astral), setuptools src layout
- ASR: sherpa-onnx (SenseVoiceSmall) / Google Web Speech API fallback
- KWS: sherpa-onnx (KeywordSpotter, zipformer-wenetspeech-3.3M)
- Audio: PyAudio, SpeechRecognition (silence segmentation)
- Concurrency: threading (Queue, Lock, Event)
- Linter: ruff

## Commands

```bash
uv sync                                  # Install deps + editable install
uv run examples/demo_cli.py              # Continuous transcription
uv run examples/demo_push_to_talk.py     # Push-to-talk (left Ctrl)
uv run examples/demo_push_to_talk_ui.py  # Push-to-talk with overlay UI
uv run examples/demo_wake_word.py        # Wake-word triggered transcription
uv run python tests/test_import.py       # Smoke test
```

## Project Structure

```
src/voicetyper/           # Installed package: core library + shared reusable logic
├── __init__.py           # Public API exports
├── audio.py              # AudioDeviceResolver (platform-aware host API)
├── recognition.py        # BackgroundSTT (continuous STT with stitch/merge)
├── recorder.py           # PushToTalkRecorder
├── models.py             # SenseVoiceSmallEngine (ASR inference)
├── kws.py                # KwsEngine (wake-word detection, pinyin token encoding)
├── downloads.py          # HTTP download & tar extraction
├── monitor.py            # ResourceMonitor (CPU/RAM/GPU)
├── settings.py           # JSON settings store (~/.voicetyper/settings.json)
├── devices.py            # InputDeviceSelector (UI-agnostic device-picker model)
└── device_watch.py       # DeviceChangeWatcher (hot-plug events; Win32, no-op elsewhere)

UI/                       # Formal UI: GUI code + static assets (not reusable logic)
└── assets/               #   Icons, bundled at package time (--add-data)

examples/                 # Demo scripts — ONLY code the formal app won't reuse
```

## Conventions

- Library-first: all reusable code lives in the installed `voicetyper` package (`src/voicetyper/`); `examples/` holds only demo-specific code the formal app won't reuse. Put reusable modules in the package (not a top-level dir) so they import without `sys.path` hacks and PyInstaller bundles them automatically — including in frozen builds.
- Context managers everywhere (`with` statements)
- Callback-driven API (on_result, on_status, etc.)
- Library code uses `logging` (no `print()`); examples configure `logging.basicConfig()`
- Commit format: `type[scope]: message`
- Line endings: LF (`.gitattributes`)
- Platform-specific code: demo-only platform glue stays in `examples/`; platform-specific code shared with the formal app lives in the package behind a platform guard that degrades to a no-op off-platform (e.g. `device_watch.DeviceChangeWatcher` → Windows WASAPI events, no-op + polling fallback elsewhere)

## Architecture

```
Consumers (application layer — each independently consumes the package):
  examples/   # demo scripts
  UI/         # formal UI + assets
        │
        ▼
voicetyper  (installed package: core library + shared reusable logic)
    ├── recognition.py   → models.py → downloads.py
    ├── kws.py           → downloads.py  [pypinyin]
    ├── recorder.py      [independent]
    ├── monitor.py       [independent]
    ├── audio.py         [independent]
    ├── settings.py      [independent]
    ├── devices.py       → audio.py, settings.py
    └── device_watch.py  [Win32-only, comtypes; no-op + polling fallback elsewhere]
```

## Model Storage

- ASR: `~/.voicetyper/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`
- KWS: `~/.voicetyper/models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/`
- Corrections: `~/.voicetyper/corrections.tsv`
