import os
import urllib.request
import tarfile
from typing import Optional

import sherpa_onnx


class SenseVoiceSmallEngine:
    """
    封装 sherpa-onnx 的 SenseVoiceSmall 模型。
    自动处理模型下载与加载。
    """

    MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
    MODEL_DIR_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

    def __init__(self, model_dir: Optional[str] = None):
        """
        初始化引擎。

        Args:
            model_dir: 模型存放目录。如果为 None，则默认存放在用户目录下的 .voicetyper/models 中。
        """
        if model_dir is None:
            home = os.path.expanduser("~")
            self.base_dir = os.path.join(home, ".voicetyper", "models")
        else:
            self.base_dir = model_dir

        self.model_path = os.path.join(self.base_dir, self.MODEL_DIR_NAME)
        self._recognizer: Optional[sherpa_onnx.OfflineRecognizer] = None

        self._ensure_model_exists()
        self._init_recognizer()

    def _ensure_model_exists(self):
        """检查模型是否存在，不存在则下载。"""
        if os.path.exists(self.model_path):
            return

        print(f"正在下载 SenseVoiceSmall 模型到 {self.base_dir} ...")
        os.makedirs(self.base_dir, exist_ok=True)

        tar_path = os.path.join(self.base_dir, "model.tar.bz2")

        # 下载
        try:
            urllib.request.urlretrieve(self.MODEL_URL, tar_path)
            print("下载完成，正在解压...")

            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(path=self.base_dir)

            print("解压完成。")
        except Exception as e:
            print(f"下载或解压模型失败: {e}")
            raise
        finally:
            if os.path.exists(tar_path):
                os.remove(tar_path)

    def _init_recognizer(self):
        """初始化 sherpa-onnx 识别器。"""
        print("正在加载 SenseVoiceSmall 模型...")
        try:
            tokens = os.path.join(self.model_path, "tokens.txt")
            model = os.path.join(self.model_path, "model.int8.onnx") # 使用量化版以降低内存

            if not os.path.exists(model):
                # 兼容性处理：有些 release 可能只包含非量化版
                model = os.path.join(self.model_path, "model.onnx")

            # 修正配置调用方式
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model,
                tokens=tokens,
                use_itn=True, # 启用逆文本标准化（如将“一二三”转为“123”）
            )
            print("模型加载完成。")
        except Exception as e:
            print(f"初始化识别器失败: {e}")
            raise

    def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        """
        识别音频数据。

        Args:
            audio_data: 原始音频字节流 (PCM)
            sample_rate: 采样率 (sherpa-onnx 需要 16000Hz，如果不同内部会自动重采样)

        Returns:
            识别出的文本
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
        return result.text
