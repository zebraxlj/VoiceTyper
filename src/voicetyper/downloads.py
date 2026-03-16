import os
import sys
import tarfile
import time
import urllib.request
import urllib.error
import socket
from typing import Callable, Optional

DownloadProgress = Callable[[int, Optional[int]], None]
CountProgress = Callable[[int, int], None]


def download_file(
    url: str,
    dest_path: str,
    *,
    on_progress: Optional[DownloadProgress] = None,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> None:
    dest_dir = os.path.dirname(os.path.abspath(dest_path))
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    tmp_path = dest_path + ".part"

    attempt = 0
    delay = 0.0
    while True:
        if delay > 0:
            time.sleep(delay)
        attempt += 1
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "VoiceTyperDownloader/0.1",
                    "Accept": "*/*",
                    "Connection": "close",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as out:
                total_header = resp.headers.get("Content-Length")
                try:
                    total = int(total_header) if total_header else None
                except ValueError:
                    total = None
                downloaded = 0
                chunk = 1024 * 512  # 512 KiB
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    out.write(data)
                    downloaded += len(data)
                    if on_progress:
                        on_progress(downloaded, total)
            os.replace(tmp_path, dest_path)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, ConnectionError, OSError):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt > retries:
                raise
            delay = max(0.5, delay * backoff + 0.5)


def extract_tar_bz2(
    tar_path: str,
    dest_dir: str,
    *,
    on_progress: Optional[CountProgress] = None,
    safe: bool = True,
) -> None:
    os.makedirs(dest_dir, exist_ok=True)

    with tarfile.open(tar_path, "r:bz2") as tar:
        members = tar.getmembers()
        total = len(members)
        base_real = os.path.realpath(dest_dir)
        done = 0
        for m in members:
            if safe:
                member_path = os.path.realpath(os.path.join(dest_dir, m.name))
                if not member_path.startswith(base_real + os.sep) and member_path != base_real:
                    raise RuntimeError("Unsafe tar contents: path traversal detected")
            tar.extract(m, path=dest_dir)
            done += 1
            if on_progress:
                on_progress(done, total)


def make_console_count_progress(prefix: str = "处理中") -> CountProgress:
    last_print_at = 0.0

    def on_progress(done: int, total: int) -> None:
        nonlocal last_print_at
        now = time.monotonic()
        if now - last_print_at < 0.1:
            return
        sys.stdout.write(f"\r{prefix}: {done}/{total}")
        sys.stdout.flush()
        last_print_at = now

    return on_progress


def make_console_download_progress(prefix: str = "下载中") -> DownloadProgress:
    last_print_at = 0.0

    def on_progress(downloaded: int, total: Optional[int]) -> None:
        nonlocal last_print_at
        now = time.monotonic()
        if now - last_print_at < 0.1:
            return
        if total and total > 0:
            pct = min(downloaded / total * 100.0, 100.0)
            sys.stdout.write(
                f"\r{prefix}: {pct:6.2f}% ({downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)"
            )
        else:
            sys.stdout.write(f"\r{prefix}: {downloaded / (1024 * 1024):.1f} MB")
        sys.stdout.flush()
        last_print_at = now

    return on_progress
