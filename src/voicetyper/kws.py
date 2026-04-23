"""关键词检测（Keyword Spotting）引擎封装。

基于 sherpa-onnx 的 ``KeywordSpotter``，提供：
- KWS 模型的自动下载与缓存管理
- 中文关键词自动转换为拼音 token（基于模型 tokens.txt 的贪心匹配）
- 临时关键词文件生成
- ``KeywordSpotter`` 的初始化与生命周期管理

典型用法::

    engine = KwsEngine(keywords=["转文字", "结束"])
    stream = engine.create_stream()
    # ... 持续喂入音频 ...
    engine.close()
"""

import logging
import os
import tempfile
import unicodedata
from typing import Optional

import sherpa_onnx
from pypinyin import pinyin, Style as PinyinStyle

from .downloads import (
    download_file,
    extract_tar_bz2,
    make_console_count_progress,
    make_console_download_progress,
)

logger = logging.getLogger(__name__)

# ── 拼音 token 编码工具 ─────────────────────────────────────


def _is_cjk(char: str) -> bool:
    """判断单个字符是否属于 CJK 统一表意文字（含扩展区）。"""
    try:
        name = unicodedata.name(char, "")
    except ValueError:
        return False
    return "CJK UNIFIED IDEOGRAPH" in name


def load_token_set(tokens_path: str) -> set[str]:
    """从 tokens.txt 加载所有有效 token（排除特殊 token）。

    Args:
        tokens_path: tokens.txt 文件路径。

    Returns:
        token 字符串的集合。
    """
    token_set: set[str] = set()
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                tok = parts[0]
                if not tok.startswith("<") and not tok.startswith("#"):
                    token_set.add(tok)
    return token_set


def _pinyin_to_tokens(py_str: str, token_set: set[str]) -> list[str]:
    """将单个汉字的完整拼音拆分为模型 token 序列（贪心最长前缀匹配）。

    KWS 模型的 token 是拼音的声母和韵母（带声调），如 ``"zhuǎn"`` → ``["zh", "uǎn"]``。
    本函数从完整拼音字符串中，每次取最长匹配 tokens.txt 中存在的前缀。

    Args:
        py_str: 单个汉字的完整拼音（带声调，如 ``"zhuǎn"``）。
        token_set: 从 tokens.txt 加载的有效 token 集合。

    Returns:
        拆分后的 token 列表。
    """
    result: list[str] = []
    i = 0
    max_token_len = 6  # tokens.txt 中最长 token 的字符数（足够覆盖）
    while i < len(py_str):
        matched = False
        for end in range(min(i + max_token_len, len(py_str)), i, -1):
            candidate = py_str[i:end]
            if candidate in token_set:
                result.append(candidate)
                i = end
                matched = True
                break
        if not matched:
            logger.warning("拼音 token 匹配失败: %r 在位置 %d（拼音: %s）", py_str[i], i, py_str)
            i += 1
    return result


def encode_keyword(keyword: str, token_set: set[str]) -> str:
    """将中文关键词编码为 sherpa-onnx KWS 模型所需的 token 行格式。

    格式：``拼音token1 拼音token2 ... @原始关键词``

    例如 ``"转文字"`` → ``"zh uǎn w én z ì @转文字"``

    非 CJK 字符（英文字母等）直接作为 token 保留。

    Args:
        keyword: 原始关键词字符串。
        token_set: 从 tokens.txt 加载的有效 token 集合。

    Returns:
        编码后的单行字符串（不含换行符）。
    """
    all_tokens: list[str] = []
    for ch in keyword:
        if _is_cjk(ch):
            # 汉字 → 拼音 → token 序列
            py_list = pinyin(ch, style=PinyinStyle.TONE, heteronym=False)
            full_py = py_list[0][0] if py_list else ""
            if full_py:
                toks = _pinyin_to_tokens(full_py, token_set)
                all_tokens.extend(toks)
        elif ch.isascii() and ch.isalpha():
            # 英文字母：大写字母作为独立 token（和 tokens.txt 中一致）
            all_tokens.append(ch.upper() if ch.upper() in token_set else ch)
        # 空格和其他字符忽略

    if not all_tokens:
        logger.warning("关键词 %r 编码后无有效 token", keyword)
        return ""

    return " ".join(all_tokens) + " @" + keyword


def build_keywords_content(keywords: list[str], token_set: set[str]) -> str:
    """将关键词列表转换为 sherpa-onnx keywords 文件内容。

    每行一个关键词，格式为 ``拼音tokens @原始汉字``。

    Args:
        keywords: 关键词字符串列表（如 ``["转文字", "结束"]``）。
        token_set: 从 tokens.txt 加载的有效 token 集合。

    Returns:
        多行字符串，可直接写入关键词文件。
    """
    lines: list[str] = []
    for kw in keywords:
        kw = kw.strip()
        if kw:
            encoded = encode_keyword(kw, token_set)
            if encoded:
                lines.append(encoded)
    return "\n".join(lines) + "\n" if lines else ""


# ── KWS 引擎 ────────────────────────────────────────────────


class KwsEngine:
    """关键词检测引擎，封装 sherpa-onnx ``KeywordSpotter``。

    功能：
    - 自动下载中文 KWS 模型（``sherpa-onnx-kws-zipformer-wenetspeech-3.3M``）
    - 将用户提供的关键词自动拆字并写入临时文件
    - 提供 ``create_stream()`` / ``accept_waveform()`` / ``decode()`` / ``get_result()``
      等流式检测方法

    支持 context manager（``with`` 语句），退出时自动清理临时文件。
    """

    MODEL_URL = (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2"
    )
    MODEL_DIR_NAME = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"

    def __init__(
        self,
        keywords: list[str],
        *,
        model_dir: Optional[str] = None,
        num_threads: int = 2,
        keywords_score: float = 1.0,
        keywords_threshold: float = 0.25,
        num_trailing_blanks: int = 1,
    ) -> None:
        """初始化 KWS 引擎。

        Args:
            keywords: 要检测的关键词列表（如 ``["转文字", "结束"]``）。
            model_dir: 模型存放根目录。为 None 时默认使用 ``~/.voicetyper/models``。
            num_threads: onnxruntime 推理线程数。
            keywords_score: 关键词 token 加分。越大越容易触发，但误触也越多。
            keywords_threshold: 触发概率阈值。越大越难触发。
            num_trailing_blanks: 关键词后需要多少空白帧才确认触发。
        """
        if not keywords:
            raise ValueError("至少需要提供一个关键词")

        self.keywords = keywords
        self.num_threads = num_threads
        self.keywords_score = keywords_score
        self.keywords_threshold = keywords_threshold
        self.num_trailing_blanks = num_trailing_blanks

        if model_dir is None:
            home = os.path.expanduser("~")
            self.base_dir = os.path.join(home, ".voicetyper", "models")
        else:
            self.base_dir = model_dir

        self.model_path = os.path.join(self.base_dir, self.MODEL_DIR_NAME)
        self._spotter: Optional[sherpa_onnx.KeywordSpotter] = None
        self._keywords_tmpfile: Optional[str] = None

        self._ensure_model_exists()
        self._init_spotter()

    # ── context manager ──────────────────────────────────────

    def __enter__(self) -> "KwsEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── 模型下载 ─────────────────────────────────────────────

    def _ensure_model_exists(self) -> None:
        """检查模型目录是否存在，不存在则从 GitHub 下载并解压。"""
        if os.path.exists(self.model_path):
            return

        logger.info("正在下载 KWS 模型到 %s ...", self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        tar_path = os.path.join(self.base_dir, "kws-model.tar.bz2")
        try:
            download_file(
                self.MODEL_URL,
                tar_path,
                on_progress=make_console_download_progress("KWS 模型下载中"),
            )
            logger.info("下载完成，正在解压...")

            extract_tar_bz2(
                tar_path,
                self.base_dir,
                on_progress=make_console_count_progress("解压中"),
                safe=True,
            )
            logger.info("KWS 模型解压完成。")
        except Exception as e:
            logger.error("下载或解压 KWS 模型失败: %s", e)
            raise
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)

    # ── 初始化 ───────────────────────────────────────────────

    def _write_keywords_file(self) -> str:
        """将关键词编码为拼音 token 格式并写入临时文件，返回文件路径。"""
        tokens_path = os.path.join(self.model_path, "tokens.txt")
        token_set = load_token_set(tokens_path)

        content = build_keywords_content(self.keywords, token_set)
        logger.info(
            "KWS 关键词（%d 个）: %s",
            len(self.keywords),
            ", ".join(self.keywords),
        )
        logger.debug("关键词文件内容:\n%s", content.rstrip())

        fd, path = tempfile.mkstemp(suffix=".txt", prefix="kws_keywords_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            os.close(fd)
            raise
        self._keywords_tmpfile = path
        return path

    def _init_spotter(self) -> None:
        """初始化 sherpa-onnx KeywordSpotter。"""
        import time as _time

        keywords_file = self._write_keywords_file()

        encoder = os.path.join(self.model_path, "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        decoder = os.path.join(self.model_path, "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
        joiner = os.path.join(self.model_path, "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")
        tokens = os.path.join(self.model_path, "tokens.txt")

        logger.info("正在初始化 KWS 引擎（%s）...", self.model_path)
        t0 = _time.perf_counter()

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_file,
            num_threads=self.num_threads,
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
            num_trailing_blanks=self.num_trailing_blanks,
        )

        elapsed = _time.perf_counter() - t0
        logger.info("KWS 引擎初始化完成，耗时 %.2fs。", elapsed)

    # ── 流式检测 API ─────────────────────────────────────────

    def create_stream(self) -> "sherpa_onnx.OnlineStream":
        """创建一个新的检测流。

        每次从监听阶段进入或重置时，应创建新 stream。

        Returns:
            sherpa-onnx OnlineStream 实例。
        """
        if self._spotter is None:
            raise RuntimeError("KWS 引擎未初始化")
        return self._spotter.create_stream()

    def is_ready(self, stream: "sherpa_onnx.OnlineStream") -> bool:
        """检查流中是否有足够数据可以解码。"""
        if self._spotter is None:
            raise RuntimeError("KWS 引擎未初始化")
        return self._spotter.is_ready(stream)

    def decode(self, stream: "sherpa_onnx.OnlineStream") -> None:
        """对流执行一步解码。"""
        if self._spotter is None:
            raise RuntimeError("KWS 引擎未初始化")
        self._spotter.decode_stream(stream)

    def get_result(self, stream: "sherpa_onnx.OnlineStream") -> str:
        """获取当前检测结果。

        若返回非空字符串，表示检测到关键词。
        获取结果后，内部状态会自动重置，等待下一次触发。

        Returns:
            检测到的关键词文本，或空字符串。
        """
        if self._spotter is None:
            raise RuntimeError("KWS 引擎未初始化")
        return self._spotter.get_result(stream)

    # ── 资源清理 ─────────────────────────────────────────────

    def close(self) -> None:
        """清理资源（删除临时关键词文件）。可安全重复调用。"""
        if self._keywords_tmpfile and os.path.exists(self._keywords_tmpfile):
            try:
                os.remove(self._keywords_tmpfile)
            except Exception:
                pass
            self._keywords_tmpfile = None
        self._spotter = None
