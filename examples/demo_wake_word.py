"""唤醒词触发录音转文字（终端版 MVP）。

后台长时间运行，用 KWS 模型持续监听麦克风：
1. 监听阶段：低功耗运行 KWS，检测唤醒词（默认"转文字"）
2. 录音阶段：唤醒后开始录音，通过 RMS 静音断句 / KWS 结束词 / Enter / Esc 结束
3. 转写阶段：录音结束后用 SenseVoiceSmall 离线转写

用法::

    uv run examples/demo_wake_word.py
    uv run examples/demo_wake_word.py --keywords "你好,开始"
    uv run examples/demo_wake_word.py --silence-duration 2.0 --min-rms 200
"""

import argparse
import logging
import math
import signal
import sys
import threading
import time
import winsound
from enum import Enum, auto
from typing import Optional

import numpy as np
import pyaudio
from colorama import init, Fore, Style
from pynput import keyboard

from voicetyper import AudioDeviceResolver
from voicetyper.kws import KwsEngine
from voicetyper.models import SenseVoiceSmallEngine

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger(__name__)

init(autoreset=True)

# ── 常量 ─────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16
FRAMES_PER_BUFFER = 1024  # ≈ 64ms per chunk at 16kHz


# ── 状态枚举 ─────────────────────────────────────────────────

class State(Enum):
    LISTENING = auto()  # 监听阶段：等待唤醒词
    RECORDING = auto()  # 录音阶段：录音中
    TRANSCRIBING = auto()  # 转写阶段：正在识别


# ── RMS 计算 ─────────────────────────────────────────────────

def rms_int16(data: bytes) -> int:
    """计算 int16 PCM 数据的 RMS 值。"""
    if not data:
        return 0
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    return int(math.sqrt(np.mean(samples * samples)))


# ── 音频反馈（非阻塞） ───────────────────────────────────────

def beep_async(frequency: int, duration_ms: int) -> None:
    """在后台线程播放 Beep，避免阻塞主循环。"""
    t = threading.Thread(
        target=winsound.Beep, args=(frequency, duration_ms), daemon=True
    )
    t.start()


# ── CLI 参数 ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="demo_wake_word",
        description="唤醒词触发录音转文字（终端版）",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="转文字",
        help="唤醒关键词，逗号分隔多个（默认：转文字）",
    )
    parser.add_argument(
        "--end-keywords",
        type=str,
        default="我说完了",
        help="结束关键词（录音阶段），逗号分隔多个（默认：我说完了）",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=1.5,
        help="静音断句阈值（秒），连续静音超过此值视为句子结束（默认：1.5）",
    )
    parser.add_argument(
        "--min-rms",
        type=int,
        default=150,
        help="RMS 静音门限（默认：150）",
    )
    parser.add_argument(
        "--keywords-score",
        type=float,
        default=1.0,
        help="KWS 关键词加分，越大越容易触发（默认：1.0）",
    )
    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.25,
        help="KWS 触发阈值，越大越难触发（默认：0.25）",
    )
    parser.add_argument(
        "--keep-period",
        action="store_true",
        default=False,
        help="保留识别结果末尾的句号/句点（默认自动去除）",
    )
    return parser.parse_args()


# ── 主程序 ────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    wake_keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    end_keywords = [k.strip() for k in args.end_keywords.split(",") if k.strip()]
    all_keywords = list(dict.fromkeys(wake_keywords + end_keywords))  # 去重保序
    wake_set = set(wake_keywords)
    end_set = set(end_keywords)

    silence_duration = args.silence_duration
    min_rms = args.min_rms
    strip_trailing_period = not args.keep_period

    print(f"{Fore.CYAN}=== 唤醒词触发录音转文字 ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}唤醒词: {', '.join(wake_keywords)}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}结束词: {', '.join(end_keywords)}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}静音断句: {silence_duration}s | RMS 门限: {min_rms}{Style.RESET_ALL}")

    # ── 初始化设备 ─────────────────────────────────────────
    print(f"{Fore.YELLOW}正在初始化麦克风...{Style.RESET_ALL}")
    device_index: Optional[int] = None
    try:
        with AudioDeviceResolver() as resolver:
            info = resolver.default_input()
            if info:
                device_index = int(info["index"])
                print(
                    f"{Fore.YELLOW}默认麦克风: [{info['index']}] {info['name']}{Style.RESET_ALL}"
                )
            else:
                print(f"{Fore.YELLOW}默认麦克风: 未知{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}麦克风初始化失败: {e}{Style.RESET_ALL}")
        return

    # ── 初始化 KWS 引擎 ───────────────────────────────────
    print(f"{Fore.YELLOW}正在初始化 KWS 引擎...{Style.RESET_ALL}")
    try:
        kws = KwsEngine(
            keywords=all_keywords,
            keywords_score=args.keywords_score,
            keywords_threshold=args.keywords_threshold,
        )
    except Exception as e:
        print(f"{Fore.RED}KWS 引擎初始化失败: {e}{Style.RESET_ALL}")
        return

    # ── 初始化 ASR 引擎 ───────────────────────────────────
    print(f"{Fore.YELLOW}正在初始化 ASR 引擎...{Style.RESET_ALL}")
    try:
        asr = SenseVoiceSmallEngine(strip_trailing_period=strip_trailing_period)
    except Exception as e:
        print(f"{Fore.RED}ASR 引擎初始化失败: {e}{Style.RESET_ALL}")
        kws.close()
        return

    # ── 打开 PyAudio 流 ───────────────────────────────────
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
    except Exception as e:
        print(f"{Fore.RED}音频流打开失败: {e}{Style.RESET_ALL}")
        kws.close()
        pa.terminate()
        return

    # ── 键盘监听 ──────────────────────────────────────────
    exit_event = threading.Event()
    confirm_event = threading.Event()  # Enter
    cancel_event = threading.Event()  # Esc

    def on_press(key) -> None:
        if key == keyboard.Key.enter:
            confirm_event.set()
        elif key == keyboard.Key.esc:
            cancel_event.set()

    def on_release(key) -> None:
        # Ctrl+C 退出
        pass

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.daemon = True
    kb_listener.start()

    # Ctrl+C 信号处理
    def sigint_handler(signum, frame) -> None:
        exit_event.set()

    try:
        signal.signal(signal.SIGINT, sigint_handler)
    except Exception:
        pass

    # ── 状态机主循环 ──────────────────────────────────────
    state = State.LISTENING
    kws_stream = kws.create_stream()
    recording_frames: list[bytes] = []
    silence_start: float = 0.0
    recording_start: float = 0.0
    # 录音阶段的结束词检测用独立 stream
    end_kws_stream = kws.create_stream() if end_keywords else None

    # 将 float32 samples 喂给 KWS stream 的辅助函数
    def feed_kws(kws_stream_obj, data: bytes) -> None:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        kws_stream_obj.accept_waveform(SAMPLE_RATE, samples.tolist())

    print(
        f"\n{Fore.MAGENTA}>>> 说出唤醒词开始录音，Enter 确认 / Esc 取消 / Ctrl+C 退出 <<<{Style.RESET_ALL}\n"
    )

    try:
        while not exit_event.is_set():
            # 读取一块音频
            try:
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            except Exception:
                continue

            if state == State.LISTENING:
                # ── 监听阶段：KWS 检测唤醒词 ────────────
                sys.stdout.write(
                    f"\r{Fore.BLUE}[监听中] 等待唤醒词...{Style.RESET_ALL}    "
                )
                sys.stdout.flush()

                feed_kws(kws_stream, data)
                while kws.is_ready(kws_stream):
                    kws.decode(kws_stream)

                result = kws.get_result(kws_stream)
                if result:
                    detected = result.strip().strip("/")
                    # 判断是否是唤醒词（而非结束词）
                    if detected in wake_set or any(w in detected for w in wake_set):
                        sys.stdout.write("\033[K")
                        print(
                            f"\r{Fore.GREEN}[唤醒] 检测到: \"{detected}\"{Style.RESET_ALL}"
                        )
                        beep_async(1000, 150)  # 高音短促提示

                        # 切换到录音阶段
                        state = State.RECORDING
                        recording_frames.clear()
                        silence_start = 0.0
                        recording_start = time.time()
                        confirm_event.clear()
                        cancel_event.clear()
                        # 重置结束词 KWS stream
                        if end_kws_stream is not None:
                            end_kws_stream = kws.create_stream()

            elif state == State.RECORDING:
                # ── 录音阶段 ─────────────────────────────
                elapsed = time.time() - recording_start
                sys.stdout.write(
                    f"\r{Fore.MAGENTA}[录音中] {elapsed:.1f}s | "
                    f"Enter=确认 Esc=取消{Style.RESET_ALL}    "
                )
                sys.stdout.flush()

                recording_frames.append(data)

                # 检查 Esc 取消
                if cancel_event.is_set():
                    cancel_event.clear()
                    sys.stdout.write("\033[K")
                    print(f"\r{Fore.RED}[取消] 录音已丢弃{Style.RESET_ALL}")
                    beep_async(400, 100)  # 低音短促提示
                    state = State.LISTENING
                    kws_stream = kws.create_stream()
                    continue

                # 检查 Enter 确认
                if confirm_event.is_set():
                    confirm_event.clear()
                    sys.stdout.write("\033[K")
                    print(f"\r{Fore.YELLOW}[手动确认] 结束录音，开始转写...{Style.RESET_ALL}")
                    beep_async(800, 100)
                    state = State.TRANSCRIBING
                    continue

                # 检查 KWS 结束词
                if end_kws_stream is not None:
                    feed_kws(end_kws_stream, data)
                    while kws.is_ready(end_kws_stream):
                        kws.decode(end_kws_stream)
                    end_result = kws.get_result(end_kws_stream)
                    if end_result:
                        end_detected = end_result.strip().strip("/")
                        if end_detected in end_set or any(
                            e in end_detected for e in end_set
                        ):
                            sys.stdout.write("\033[K")
                            print(
                                f"\r{Fore.YELLOW}[结束词] 检测到: \"{end_detected}\"，"
                                f"开始转写...{Style.RESET_ALL}"
                            )
                            beep_async(800, 100)
                            state = State.TRANSCRIBING
                            continue

                # 检查 RMS 静音断句
                current_rms = rms_int16(data)
                if current_rms < min_rms:
                    if silence_start == 0.0:
                        silence_start = time.time()
                    elif time.time() - silence_start >= silence_duration:
                        sys.stdout.write("\033[K")
                        print(
                            f"\r{Fore.YELLOW}[静音断句] {silence_duration}s 静音，"
                            f"开始转写...{Style.RESET_ALL}"
                        )
                        beep_async(800, 100)
                        state = State.TRANSCRIBING
                        continue
                else:
                    silence_start = 0.0

            if state == State.TRANSCRIBING:
                # ── 转写阶段 ─────────────────────────────
                if not recording_frames:
                    print(f"{Fore.RED}[空录音] 没有录到音频{Style.RESET_ALL}")
                    state = State.LISTENING
                    kws_stream = kws.create_stream()
                    continue

                pcm_data = b"".join(recording_frames)
                duration = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
                print(
                    f"{Fore.YELLOW}识别中... ({duration:.1f}s 音频){Style.RESET_ALL}"
                )

                try:
                    text = asr.transcribe(pcm_data, SAMPLE_RATE)
                    if text.strip():
                        print(
                            f"{Fore.GREEN}结果: {Style.BRIGHT}{text}{Style.RESET_ALL}"
                        )
                    else:
                        print(f"{Fore.RED}[无法识别]{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}识别失败: {e}{Style.RESET_ALL}")

                # 回到监听阶段
                state = State.LISTENING
                kws_stream = kws.create_stream()
                recording_frames.clear()
                print()

    except KeyboardInterrupt:
        pass
    finally:
        # ── 清理资源 ──────────────────────────────────────
        try:
            kb_listener.stop()
        except Exception:
            pass
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
        kws.close()
        print(f"\n{Fore.CYAN}=== 程序已退出 ==={Style.RESET_ALL}")


if __name__ == "__main__":
    main()
