import argparse
import os
from typing import Optional
from urllib.parse import urlparse

from voicetyper.downloads import (
    download_file,
    extract_tar_bz2,
    make_console_count_progress,
    make_console_download_progress,
)

SAMPLE_URL = "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"
LARGE_SAMPLE_CANDIDATES = [
    "https://fsn1-speed.hetzner.com/100MB.bin",
    "https://ash-speed.hetzner.com/100MB.bin",
    "https://nbg1-speed.hetzner.com/100MB.bin",
    "https://sin-speed.hetzner.com/100MB.bin",
]


def _default_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    return name or "download.bin"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="demo_download")
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument(
        "--sample-large",
        action="store_true",
        help="使用约 100MB 的示例文件下载（用于观察进度条）。若同时提供 --url，则以 --url 为准。",
    )
    parser.add_argument("--timeout", type=float, default=15.0, help="下载超时时间（秒）")
    parser.add_argument("--retries", type=int, default=3, help="下载失败重试次数")
    parser.add_argument("--dest-dir", type=str, default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--extract-tar-bz2", action="store_true")
    parser.add_argument("--remove-archive", action="store_true")
    parser.add_argument("--unsafe", action="store_true")

    args = parser.parse_args(argv)

    if args.url:
        url = args.url
        urls = [url]
    elif args.sample_large:
        urls = LARGE_SAMPLE_CANDIDATES
        url = urls[0]
    else:
        url = SAMPLE_URL
        urls = [url]
    dest_dir = args.dest_dir or os.getcwd()
    filename = args.filename or _default_filename_from_url(url)
    dest_path = os.path.join(dest_dir, filename)

    def try_download(urls: list[str]) -> str:
        last_err: Exception | None = None
        for u in urls:
            print(f"下载: {u}")
            print(f"保存到: {dest_path}")
            try:
                download_file(
                    u,
                    dest_path,
                    on_progress=make_console_download_progress("下载中"),
                    timeout=args.timeout,
                    retries=args.retries,
                )
                print()
                return u
            except Exception as e:
                print(f"下载失败: {e}")
                last_err = e
        if last_err:
            raise last_err
        return urls[-1]

    used_url = try_download(urls)

    if args.extract_tar_bz2:
        if not dest_path.endswith(".tar.bz2"):
            raise ValueError("仅支持 .tar.bz2。请提供 --filename 以指定正确后缀，或关闭 --extract-tar-bz2")
        print(f"解压到: {dest_dir}")
        extract_tar_bz2(
            dest_path,
            dest_dir,
            on_progress=make_console_count_progress("解压中"),
            safe=not args.unsafe,
        )
        print()

        if args.remove_archive and os.path.exists(dest_path):
            os.remove(dest_path)
            print(f"已删除: {dest_path}")

    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
