import sys

import speech_recognition as sr
from colorama import init, Fore, Style

import demo_consts
sys.path.append(demo_consts.SRC_DIR)

from voicetyper import AudioDeviceResolver, BackgroundSTT  # noqa: E402

init(autoreset=True)


def on_request_error_print(e: Exception) -> None:
    sys.stdout.write("\033[K")
    print(f"{Fore.RED}无法连接到 Google API: {e}{Style.RESET_ALL}")


def on_result_print(text: str) -> None:
    sys.stdout.write("\033[K")
    print(f"{Fore.GREEN}结果: {Style.BRIGHT}{text}{Style.RESET_ALL}")


def on_status_print(msg: str) -> None:
    if msg == "recognizing":
        sys.stdout.write("\033[K")
        print(f"{Fore.YELLOW}识别中...{Style.RESET_ALL}", end="\r")


def on_unintelligible_print() -> None:
    sys.stdout.write("\033[K")
    print(f"{Fore.RED}[无法识别]{Style.RESET_ALL}")


def main() -> None:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.5
    recognizer.non_speaking_duration = 0.3

    print(f"{Fore.CYAN}=== 实时连续语音转文字（Google API）==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}正在初始化麦克风...{Style.RESET_ALL}")

    try:
        with AudioDeviceResolver() as resolver:
            info = resolver.default_input()
            print('info:', info)
            if info:
                print(f"{Fore.YELLOW}默认麦克风设备: [{info['index']}] {info['name']}{Style.RESET_ALL}")
                mic = sr.Microphone(device_index=info["index"])
            else:
                print(f"{Fore.YELLOW}默认麦克风设备: 未知{Style.RESET_ALL}")
                mic = sr.Microphone()
        print(f"{Fore.GREEN}麦克风就绪!{Style.RESET_ALL}")
        print("正在进行环境噪声校准，请保持安静 1 秒...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"{Fore.GREEN}校准完成!{Style.RESET_ALL}")

        service = BackgroundSTT(recognizer, language="zh-CN", phrase_time_limit=10)
        service.start_listening(mic)
        service.start_worker(
            on_status_print, on_result_print, on_unintelligible_print, on_request_error_print
        )

        print(f"\n{Fore.MAGENTA}>>> 开始讲话（Ctrl+C 退出），持续收听中... <<<{Style.RESET_ALL}\n")

        try:
            while True:
                print(f"{Fore.BLUE}收听中...{Style.RESET_ALL}", end="\r")
                if service.worker_thread:
                    service.worker_thread.join(0.2)
        except KeyboardInterrupt:
            pass

        service.stop()

    except OSError as e:
        print(f"{Fore.RED}错误: 无法访问麦克风。请检查麦克风设置或权限。{Style.RESET_ALL}")
        print(f"详情: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{Fore.CYAN}=== 程序已退出 ==={Style.RESET_ALL}")

if __name__ == "__main__":
    main()
