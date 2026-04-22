# VoiceTyper - Project Instructions

## Overview

VoiceTyper is a Python library for real-time speech-to-text on Windows. It provides continuous background transcription and push-to-talk recording, powered by the SenseVoiceSmall offline ASR engine (with Google Web Speech API as fallback).

## Tech Stack

- **Language**: Python 3.11+
- **Package manager**: uv (Astral)
- **Build system**: setuptools (src layout, editable install)
- **ASR engine**: sherpa-onnx (SenseVoiceSmall) / Google Web Speech API fallback
- **Audio**: PyAudio (PortAudio), SpeechRecognition (silence-based segmentation)
- **Numeric**: numpy (PCM int16 → float32 conversion)
- **Concurrency**: threading (Queue, Lock, Event)
- **Linter**: ruff

## Project Structure

```
src/voicetyper/           # Core library package
├── __init__.py           # Public API exports
├── audio.py              # AudioDeviceResolver - device enumeration (PyAudio/WASAPI)
├── recognition.py        # BackgroundSTT - continuous speech-to-text with stitch/merge
├── recorder.py           # PushToTalkRecorder - push-to-talk recording lifecycle
├── models.py             # SenseVoiceSmallEngine - ASR model management & inference
├── downloads.py          # HTTP download & tar extraction utilities
└── monitor.py            # ResourceMonitor - CPU/RAM/GPU monitoring thread

examples/                 # Demo scripts (consumers of library API, not part of API)
├── demo_cli.py           # Continuous transcription CLI
├── demo_push_to_talk.py  # Left-Ctrl push-to-talk
├── demo_push_to_talk_ui.py  # Shift+Win push-to-talk with overlay UI + tray + text injection + background model loading
└── demo_download.py      # Download utility demo

tests/
└── test_import.py        # Smoke test for public exports

push_to_talk_ui.spec      # PyInstaller spec for packaging demo_push_to_talk_ui as standalone exe
```

## Public API (`voicetyper.__init__`)

| Symbol | Source module | Description |
|---|---|---|
| `AudioDeviceResolver` | `audio.py` | PyAudio/WASAPI device enumeration |
| `BackgroundSTT` | `recognition.py` | Continuous speech-to-text with stitch/merge |
| `AsrEngine` | `recognition.py` | Enum: `SENSEVOICE_SMALL`, `GOOGLE` |
| `PushToTalkRecorder` | `recorder.py` | Push-to-talk recording lifecycle |
| `RecorderConfig` | `recorder.py` | Dataclass: rate, channels, chunk size, sample format |
| `ResourceMonitor` | `monitor.py` | CPU/RAM/GPU monitoring thread |

`SenseVoiceSmallEngine` (`models.py`) is not re-exported but is imported directly by demos as a semi-public API: `from voicetyper.models import SenseVoiceSmallEngine`.

## Architecture

```
examples/  (application layer - consumers of library API)
    │
voicetyper.__init__.py  (public API facade)
    │
    ├── recognition.py  (BackgroundSTT) ──→ models.py ──→ downloads.py
    ├── recorder.py     (PushToTalkRecorder)  [independent, only pyaudio + stdlib]
    ├── monitor.py      (ResourceMonitor)     [independent, only psutil + stdlib]
    └── audio.py        (AudioDeviceResolver) [independent, only pyaudio]
```

## Key Design Patterns

- **Library-first**: Core logic in `src/voicetyper/`, examples are consumers only
- **Context manager everywhere**: All major classes support `with` statements
- **Callback-driven API**: `BackgroundSTT` uses callbacks (`on_result`, `on_status`, etc.) to decouple recognition from UI
- **Engine fallback**: If local SenseVoiceSmall fails, auto-degrades to Google API
- **Thread-based concurrency**: Worker threads + Queue for async audio processing
- **Post-processing pipeline**: ASR output -> strip punctuation -> TSV-based corrections (`~/.voicetyper/corrections.tsv`)
- **Platform-specific code isolated to examples**: Win32 APIs (SendInput, pystray, tkinter overlay) only in demo scripts

## Data Flow

```
Microphone → silence-based segmentation (speech_recognition)
           → RMS silence filter
           → overlap prepend (tail of previous segment, default 200ms)
           → stitch/merge adjacent segments
           → ASR inference (SenseVoiceSmall or Google fallback)
           → post-processing (strip punctuation → TSV corrections)
           → callback to consumer
```

## Common Commands

```bash
uv sync                                    # Install dependencies + editable install
uv run examples/demo_cli.py                # Run continuous transcription
uv run examples/demo_push_to_talk.py       # Run push-to-talk (left Ctrl)
uv run examples/demo_push_to_talk_ui.py    # Run push-to-talk with UI
uv run python tests/test_import.py         # Run smoke test
```

## Model Storage

- Models cached at `~/.voicetyper/models/` (auto-downloaded on first use)
- Corrections file at `~/.voicetyper/corrections.tsv`

## Dependencies Note

Some dependencies in `pyproject.toml` are only used by example scripts, not by the core library:

| Dependency | Used by |
|---|---|
| `pynput` | `demo_push_to_talk.py`, `demo_push_to_talk_ui.py` |
| `pystray` | `demo_push_to_talk_ui.py` |
| `pillow` | `demo_push_to_talk_ui.py` |
| `colorama` | `demo_cli.py`, `demo_push_to_talk.py` |

These are currently declared as project-level dependencies rather than optional extras.

## Conventions

- Commit messages follow `type[scope]: message` format
- Library code uses `logging` (no `print()`); examples configure `logging.basicConfig()`
- Line endings enforced as LF (`.gitattributes`)
