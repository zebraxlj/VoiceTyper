# VoiceTyper — 离线语音输入库

基于 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice)（sherpa-onnx）的可复用离线语音转文字库，支持：

- 离线 ASR（默认 SenseVoiceSmall），可回退 Google Web Speech API
- 分段识别 + 拼接修正（短间隔片段自动合并重识）
- Push-to-Talk 录音器（按键说话）
- 连续监听后台转写

## 前置要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 包管理器
- 麦克风设备
- Windows（UI demo 依赖 Win32 API；核心库无平台限制）

## 安装与开发

```bash
# 1. 克隆仓库
git clone <repo-url>
cd VoiceTyper

# 2. 创建虚拟环境、安装依赖，并以开发模式安装 voicetyper 库
uv sync
```

`uv sync` 一条命令完成所有工作：创建 venv、安装第三方依赖、并将 `voicetyper` 以 editable 模式安装到 venv 中。之后即可在 venv 中任意位置 `import voicetyper`。

安装完成后验证：

```bash
uv run python -c "import voicetyper; print(voicetyper.__all__)"
```

## 运行示例

### 离线连续转写（CLI）

```bash
uv run examples/demo_cli.py
```

### Push-to-Talk（左 Ctrl 长按录音，松开识别）

```bash
uv run examples/demo_push_to_talk.py --hold-ms 300
```

- `--hold-ms` 调大可避免误触发（例如 400），调小提升灵敏度（例如 200）
- `--keep-period` 保留识别结果末尾的句号

### Push-to-Talk + UI 悬浮窗（Shift+Win 触发）

```bash
uv run examples/demo_push_to_talk_ui.py
```

- 按住 Shift+Win 超过 300ms 开始录音，松开后识别并输入到当前光标位置
- 系统托盘图标可退出

## 项目结构

```
src/voicetyper/         # 核心库
    __init__.py         # 公共 API 导出
    audio.py            # 音频设备枚举（AudioDeviceResolver）
    recognition.py      # 后台连续转写（BackgroundSTT）
    recorder.py         # Push-to-Talk 录音器（PushToTalkRecorder）
    models.py           # ASR 模型管理（SenseVoiceSmallEngine）
    downloads.py        # 文件下载与解压
    monitor.py          # 资源监控（CPU/RAM/GPU）
examples/               # 示例脚本（非库 API）
tests/                  # 测试
```

## 依赖说明

| 依赖 | 用途 |
|---|---|
| `sherpa-onnx` | SenseVoiceSmall 离线 ASR 推理 |
| `SpeechRecognition` | 静音检测分段 + Google API 回退 |
| `PyAudio` | 麦克风音频采集 |
| `pynput` | 全局热键监听 |
| `numpy` | 音频数据处理 |
| `pystray` + `Pillow` | 系统托盘图标（UI demo） |
| `psutil` | CPU/RAM 监控 |
| `colorama` | 终端彩色输出 |
