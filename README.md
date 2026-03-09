# 简易 Python 语音转文字 (STT) Demo

这是一个使用 Python 实现的简单语音转文字演示程序。
它使用 `SpeechRecognition` 库和 Google Web Speech API 将麦克风输入的语音实时转换为中文文本。

## 前置要求

- Python 3.10 或更高版本
- `uv` 包管理器 (推荐)
- 麦克风设备
- 互联网连接 (用于访问 Google API)

## 安装与运行

本项目使用 `uv` 进行依赖管理和运行。

1. **初始化环境并安装依赖**:
   ```bash
   uv sync
   ```
   或者直接运行（会自动安装依赖）：
   
2. **运行程序**:
   ```bash
   uv run src/demo_speech_recognition.py
   ```

## 依赖说明

- `SpeechRecognition`: 处理语音识别的核心库。
- `pyaudio`: 处理麦克风音频输入 (Windows 下通常需要此库)。
- `colorama`: 用于终端彩色输出，提升交互体验。

## 注意事项

- 本 Demo 使用 Google 免费 API，仅供测试和演示使用。
- 识别速度和准确率取决于网络连接状况。
- 如果遇到麦克风错误，请确保已授予终端访问麦克风的权限。
