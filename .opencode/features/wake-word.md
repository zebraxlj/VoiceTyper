# Feature：唤醒词触发录音转文字（Wake Word → Record → Transcribe）

## 概述

第三种语音交互模式：后台长时间运行，用轻量 KWS（Keyword Spotting）模型持续监听麦克风，当检测到用户喊出触发词（如"转文字"）时，自动开始录音；录音期间通过 RMS 静音检测判断句子结束，随后将录到的音频送入 SenseVoiceSmall 离线转写。

与已有两个 demo 的对比：

| | demo_cli（持续监听） | demo_push_to_talk（按键触发） | **demo_wake_word（唤醒词触发）** |
|---|---|---|---|
| 触发方式 | 始终在听 | 长按快捷键 | 喊出唤醒词 |
| 结束方式 | 静音自动断句 | 松开快捷键 | 静音断句 / 结束关键词 / Enter / Esc |
| 适用场景 | 全量转录 | 精准控制 | **免手操作、按需转写** |

## 设计决策记录

| 决策项 | 选择 | 理由 |
|---|---|---|
| 触发词检测 | sherpa-onnx `KeywordSpotter`（KWS 模型） | 低延迟、低 CPU，适合长时间后台运行；比持续跑完整 ASR 匹配关键词更高效 |
| KWS 模型 | `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`（中文） | sherpa-onnx 官方预训练中文 KWS 模型，仅 3.3MB |
| 默认唤醒词 | "转文字"（可通过 `--keywords` 参数配置） | 用户可自定义，关键词文件中每个汉字用空格分隔 |
| 触发词处理 | **不纳入**最终转写结果 | 唤醒词仅作为信号，触发后开始新录音 |
| 录音模式 | PyAudio 持续录音 + RMS 静音检测断句 | 比 `sr.listen()` 更可控：支持中途取消、无阻塞、可同时跑 KWS 检测结束词 |
| 断句策略 | RMS 静音检测（连续 N 帧低于阈值视为结束），阈值可配置 | 简单可靠 |
| 最大录音时长 | **无硬性上限** | 完全依赖静音断句 + 人工兜底（Esc / Enter） |
| 结束关键词 | **支持**：录音期间同时跑 KWS 检测结束词（如"结束"） | 免手操作的完整体验 |
| 手动结束 | Enter 立即触发转写，Esc 取消本次录音（快捷键可配置） | 人工兜底 |
| 音频反馈 | `winsound.Beep()`（唤醒命中 / 录音结束用不同频率区分） | Windows 内置，无需额外文件 |
| UI | **Phase 1**：终端彩色文字（colorama）；**Phase 2**：tkinter 悬浮窗 + 托盘 | 渐进式开发 |
| 库层变更 | 新建 `src/voicetyper/kws.py`（封装 KWS 模型下载、初始化、关键词文件生成、流式检测） | 可复用，不仅服务于此 demo |

## 技术方案

### 架构：两阶段状态机

```
┌─────────────────────────────────────────────────────────────┐
│                    后台常驻运行                                │
│                                                             │
│  ┌──────────────┐    唤醒词命中     ┌───────────────────┐   │
│  │  监听阶段     │ ──────────────→ │  录音阶段          │   │
│  │  (KWS 检测)   │ ←────────────── │  (PyAudio 录音     │   │
│  │              │  转写完成/取消    │   + RMS 静音断句   │   │
│  │  低功耗运行   │                  │   + KWS 结束词)    │   │
│  └──────────────┘                  └───────────────────┘   │
│                                          │                  │
│                                          ↓                  │
│                                   SenseVoiceSmall           │
│                                   离线转写                   │
└─────────────────────────────────────────────────────────────┘
```

### 监听阶段（Idle → Triggered）

- 用 PyAudio 以 16kHz/mono/int16 持续采集小块音频（如 1024 帧 ≈ 64ms）
- 每块音频喂给 `KeywordSpotter`，调用 `is_ready()` + `decode_stream()` + `get_result()`
- 当 `get_result()` 返回非空字符串 → 唤醒词命中
- 命中后：播放提示音（Beep）、终端输出状态、切换到录音阶段

### 录音阶段（Recording → Transcribing）

- 继续用同一个 PyAudio 流采集音频，缓存到帧列表
- **同时**运行两个检测：
  1. **RMS 静音断句**：每块计算 RMS，若连续 `silence_duration`（可配，默认 1.5s）低于阈值 → 自动结束
  2. **KWS 结束词检测**：同一块音频也喂给另一个 KWS stream，检测结束关键词（如"结束"）→ 命中即结束
- **键盘监听**（pynput，后台线程）：
  - Enter → 立即结束录音并转写
  - Esc → 取消本次录音，丢弃音频，回到监听阶段
- 录音结束后：播放结束提示音、将 PCM 数据送入 `SenseVoiceSmallEngine.transcribe()`、输出结果

### 模块分工

| 模块 | 职责 |
|---|---|
| `src/voicetyper/kws.py`（新建） | KWS 模型下载管理、`KeywordSpotter` 初始化、关键词文件生成（中文自动拆字）、流式检测封装 |
| `src/voicetyper/models.py`（现有） | SenseVoiceSmall ASR 推理（不变） |
| `src/voicetyper/downloads.py`（现有） | HTTP 下载 + tar 解压（复用） |
| `examples/demo_wake_word.py`（新建） | 终端版 demo：状态机主循环、PyAudio 录音、RMS 断句、键盘监听、Beep 反馈 |

### KWS 关键词文件格式

sherpa-onnx 的中文 KWS 模型使用 CJK 字符作为 token，关键词文件每行一个关键词，每个字用空格分隔：

```
转 文 字
结 束
```

`kws.py` 会提供自动拆字的工具函数：输入 `"转文字"` → 输出 `"转 文 字"`。

### 可配置参数（CLI 参数）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--keywords` | `转文字` | 唤醒关键词（逗号分隔多个） |
| `--end-keywords` | `结束` | 结束关键词（录音阶段，逗号分隔多个） |
| `--silence-duration` | `1.5` | 静音断句阈值（秒），连续静音超过此值视为句子结束 |
| `--min-rms` | `150` | RMS 静音门限 |
| `--keywords-score` | `1.0` | KWS 关键词加分（越大越容易触发） |
| `--keywords-threshold` | `0.25` | KWS 触发阈值（越大越难触发） |
| `--confirm-key` | `enter` | 手动确认转写的快捷键 |
| `--cancel-key` | `escape` | 取消录音的快捷键 |
| `--keep-period` | `false` | 是否保留识别结果末尾的句号 |

## 实现步骤

1. **新建 `src/voicetyper/kws.py`**
   - KWS 模型下载逻辑（复用 `downloads.py`）
   - 中文关键词自动拆字
   - 临时关键词文件生成（`tempfile`）
   - `KeywordSpotter` 初始化封装
2. **新建 `examples/demo_wake_word.py`**
   - CLI 参数解析
   - PyAudio 音频流管理
   - 监听阶段主循环（KWS 检测）
   - 录音阶段（RMS 断句 + KWS 结束词 + 键盘监听）
   - SenseVoiceSmall 转写
   - winsound.Beep 音频反馈 + colorama 终端输出
3. **更新 `src/voicetyper/__init__.py`** 导出 KWS 相关 API
4. **更新项目文档**（PLAN.md 待办、instructions.md 项目结构）

## 风险与注意事项

- **KWS 模型精度**：3.3MB 的小模型对口音/语速敏感，可能需要调 `keywords_score` / `keywords_threshold` 来平衡误触和漏触
- **双 KWS 流并行**：录音阶段同时跑唤醒词 KWS 和结束词 KWS，CPU 开销需实测确认是否可接受
- **环境噪声**：RMS 静音断句在嘈杂环境下可能失效，后续可引入 Silero VAD 替代
- **PyAudio 流共享**：监听阶段和录音阶段共用同一个 PyAudio 流，状态切换时需注意线程安全
- **winsound.Beep 阻塞**：`Beep()` 是同步阻塞调用，需要在单独线程播放以免阻塞主循环
