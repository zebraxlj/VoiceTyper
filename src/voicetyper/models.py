import logging
import os
import re
from typing import Optional

import sherpa_onnx

logger = logging.getLogger(__name__)

from .downloads import (
    download_file,
    extract_tar_bz2,
    make_console_count_progress,
    make_console_download_progress,
)


class SenseVoiceSmallEngine:
    """封装 sherpa-onnx 的 SenseVoiceSmall 离线语音识别模型。

    功能：
    - 自动下载并解压模型文件到本地缓存目录。
    - 支持 int8 量化模型与 fp32 全精度模型切换。
    - 可选去除识别结果末尾的句号/句点。
    - 支持文本纠错：通过 TSV 词表修正专有名词的识别错误。

    纠错词表格式（TSV，默认路径 ``~/.voicetyper/corrections.tsv``）::

        # 每行一条：错误写法<TAB>正确写法（以 # 开头的行为注释）
        open search	OpenSearch

    使用示例::

        engine = SenseVoiceSmallEngine()                      # 量化模型，去除末尾句号
        engine = SenseVoiceSmallEngine(quantized=False)       # 全精度模型，精度更高
        text = engine.transcribe(pcm16_bytes, sample_rate=16000)
    """

    MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
    MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

    DEFAULT_CORRECTIONS_PATH = os.path.join(
        os.path.expanduser("~"), ".voicetyper", "corrections.tsv"
    )

    def __init__(
        self,
        model_dir: Optional[str] = None,
        strip_trailing_period: bool = True,
        quantized: bool = True,
        num_threads: int = 2,
        corrections_file: Optional[str] = None,
    ):
        """初始化引擎。

        Args:
            model_dir: 模型存放目录。为 None 时默认使用 ``~/.voicetyper/models``。
            strip_trailing_period: 是否自动去除识别结果末尾的句号/句点（默认开启）。
            quantized: 是否使用 int8 量化模型（默认开启）。
                设为 False 后使用 fp32 全精度模型，识别精度更高但内存占用约为量化版的 4 倍。
                若指定版本不存在，会自动回退到另一版本。
            num_threads: onnxruntime 推理线程数（默认 2）。增大可降低延迟，但会占用更多 CPU。
            corrections_file: 纠错词表文件路径（TSV 格式，每行 ``错误写法<TAB>正确写法``）。
                为 None 时默认读取 ``~/.voicetyper/corrections.tsv``，文件不存在则跳过。
        """
        self.strip_trailing_period = strip_trailing_period
        self.quantized = quantized
        self.num_threads = num_threads
        if model_dir is None:
            home = os.path.expanduser("~")
            self.base_dir = os.path.join(home, ".voicetyper", "models")
        else:
            self.base_dir = model_dir

        self.model_path = os.path.join(self.base_dir, self.MODEL_DIR_NAME)
        self._recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None
        self._correction_rules: list[tuple[re.Pattern, str]] = []

        self._ensure_model_exists()
        self._init_recognizer()
        self._load_corrections(
            corrections_file
            if corrections_file is not None
            else self.DEFAULT_CORRECTIONS_PATH
        )

    def _ensure_model_exists(self):
        """检查模型目录是否存在，不存在则从 GitHub 下载并解压。"""
        if os.path.exists(self.model_path):
            return

        logger.info("正在下载 SenseVoiceSmall 模型到 %s ...", self.base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        tar_path = os.path.join(self.base_dir, "model.tar.bz2")

        # 下载
        try:
            download_file(
                self.MODEL_URL,
                tar_path,
                on_progress=make_console_download_progress("下载中"),
            )
            logger.info("下载完成，正在解压...")

            extract_tar_bz2(
                tar_path,
                self.base_dir,
                on_progress=make_console_count_progress("解压中"),
                safe=True,
            )
            logger.info("解压完成。")
        except Exception as e:
            logger.error("下载或解压模型失败: %s", e)
            raise
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)

    def _init_recognizer(self):
        """根据 ``self.quantized`` 选择模型文件并初始化 sherpa-onnx 离线识别器。"""
        import time as _time

        logger.info("正在加载 SenseVoiceSmall 模型（%s）...", self.model_path)
        try:
            tokens = os.path.join(self.model_path, "tokens.txt")

            if self.quantized:
                model = os.path.join(self.model_path, "model.int8.onnx")
                if not os.path.exists(model):
                    logger.warning("int8 量化模型不存在，回退到 fp32 全精度模型。")
                    model = os.path.join(self.model_path, "model.onnx")
            else:
                model = os.path.join(self.model_path, "model.onnx")
                if not os.path.exists(model):
                    logger.warning("fp32 全精度模型不存在，回退到 int8 量化模型。")
                    model = os.path.join(self.model_path, "model.int8.onnx")

            t0 = _time.perf_counter()
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model,
                tokens=tokens,
                use_itn=True,  # 启用逆文本标准化（如将"一二三"转为"123"）
                num_threads=self.num_threads,
            )
            elapsed = _time.perf_counter() - t0
            variant = "int8 量化" if self.quantized else "fp32 全精度"
            logger.info("模型加载完成（%s），耗时 %.2fs。", variant, elapsed)
        except Exception as e:
            logger.error("初始化识别器失败: %s", e)
            raise

    def _load_corrections(self, path: str) -> None:
        """从 TSV 文件加载纠错替换规则。

        文件格式为每行一条 ``错误写法<TAB>正确写法``，以 ``#`` 开头的行视为注释。
        匹配时大小写不敏感，替换结果保留词表中指定的大小写。
        """
        self._correction_rules = []
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.rstrip("\n\r")
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) != 2 or not parts[0] or not parts[1]:
                        logger.warning("纠错词表第 %d 行格式错误，已跳过: %r", lineno, line)
                        continue
                    pattern = re.compile(re.escape(parts[0]), re.IGNORECASE)
                    self._correction_rules.append((pattern, parts[1]))
            if self._correction_rules:
                logger.info("已加载 %d 条纠错规则（%s）。", len(self._correction_rules), path)
        except Exception as e:
            logger.error("加载纠错词表失败: %s", e)

    def _apply_corrections(self, text: str) -> str:
        """对识别结果应用纠错替换规则。"""
        for pattern, replacement in self._correction_rules:
            text = pattern.sub(replacement, text)
        return text

    def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        """将 PCM 音频数据转为文本。

        Args:
            audio_data: 原始 PCM 音频字节流（int16 编码）。
            sample_rate: 音频采样率（Hz）。sherpa-onnx 内部会自动重采样到 16000Hz。

        Returns:
            识别出的文本。依次经过末尾句号去除（如开启）和纠错替换（如有词表）。

        Raises:
            RuntimeError: 识别器未初始化时调用。
        """
        if self._recognizer is None:
            raise RuntimeError("Recognizer not initialized")

        stream = self._recognizer.create_stream()

        # 将字节流转换为 float32 数组 (归一化到 -1.0 ~ 1.0)
        # 注意：speech_recognition 的 get_raw_data() 返回的是 int16 (2 bytes)
        import numpy as np

        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)

        result = stream.result
        text = result.text
        if self.strip_trailing_period:
            # 去除模型自动添加的末尾句号/句点
            text = text.rstrip("。.")
        if self._correction_rules:
            text = self._apply_corrections(text)
        return text
