# VoiceTyper 规划（AI 管家语音输入模块）

## 目标

- 打造可复用的语音输入库，作为“AI 管家 / 贾维斯”系统的语音前端模块
- 可在低功耗家用服务器长期运行：低 CPU 占用、低内存、可离线、可控延迟
- 作为库提供稳定 API：可嵌入后续项目（CLI / GUI / 服务端）

## 范围与非目标

### 范围

- 热词/按键触发或持续监听的音频采集
- 语音活动检测（VAD）与分段
- 离线 ASR（默认 SenseVoiceSmall），可切换到在线 ASR（Google）作为兜底
- 结果“回退修正”能力（拼接重识 + 覆盖文本）
- 可观察性与易排错（日志、配置、依赖检查）

### 非目标（先不做或后做）

- 多轮对话管理、工具调用（Home Assistant 控制等）
- TTS（语音合成）与角色化音色
- 端到端流式 ASR（真正边说边出字的 Transducer 流式）

## 当前现状（已完成）

- 已有库结构：`src/voicetyper/`（audio / recognition / models）与 `examples/`
- `BackgroundSTT` 支持分段识别 + 拼接修正
- 默认使用本地 SenseVoiceSmall（sherpa-onnx），失败回退 Google
- ASR 引擎选择已改为 `AsrEngine`（Enum），便于后续扩展更多模型
- 模型下载与解压支持进度展示，并已抽象为可复用下载模块（`voicetyper.downloads`）
- SenseVoiceSmall 识别链路增加：
  - 段边界 overlap（上一段尾部音频叠加到下一段开头，降低漏字）
  - 静音门限（RMS 过滤，降低"没说话也出词"的误触发）
- Push-to-talk 示例已可用：长按左 Ctrl 收音，松开识别（`examples/demo_push_to_talk.py`）
- UI push-to-talk 示例已可用：悬浮窗 + 剪贴板 + 光标输入 + 托盘退出（`examples/demo_push_to_talk_ui.py`）
- `SenseVoiceSmallEngine` 支持多项可配置参数：
  - `strip_trailing_period`：可选去除模型输出末尾的句号/句点
  - `quantized`：可选 int8 量化模型或 fp32 全精度模型（精度更高、内存更大）
  - `num_threads`：onnxruntime 推理线程数（默认 2，可调）
  - `corrections_file`：基于 TSV 词表的识别结果纠错（大小写不敏感替换，默认读取 `~/.voicetyper/corrections.tsv`）
- UI push-to-talk 热键防误触：通过 Win32 `GetAsyncKeyState` 二次校验物理按键状态，修复 Windows 吞掉修饰键 release 事件导致的状态残留问题
- 模型加载计时输出
- 新增轻量资源监控模块 `voicetyper.monitor.ResourceMonitor`：后台线程定时输出当前进程 CPU / RAM，并在可用时输出 GPU 显存占用
- UI push-to-talk 示例支持可选资源监控开关（`SHOW_RESOURCE_USAGE`），便于录音/识别链路调优
- 文本输入改用 Win32 `SendInput`（`KEYEVENTF_UNICODE`）逐字符注入，绕过中文输入法对英文字母的拦截，同时保留逐字输入的视觉效果

## 里程碑

### M1：语音输入库可用（稳定离线）

- 输出：提供稳定 `BackgroundSTT` API；示例可在 Windows 正常离线识别
- 重点：
  - 依赖与平台兼容性（Windows DLL / VC++ / wheel 版本）
  - 模型下载缓存与可配置目录
  - 错误处理与回退策略（离线→在线）

### M2：长期运行与低功耗优化

- 输出：持续监听耗电/占用低；无语音时接近空闲
- 重点：
  - VAD 前置：先用轻量 VAD 决定何时送入 ASR
  - 降低重采样/特征提取开销（尽量 16kHz 输入）
  - 限制拼接最大时长与请求频率

### M3：产品化（可嵌入、可配置、可测试）

- 输出：可被其他项目当依赖引入；提供配置层与测试覆盖
- 重点：
  - 配置体系（YAML/ENV/代码参数）
  - 单元测试与回归样例（固定音频输入、验证输出）
  - 发布到私有/公开索引（可选）

## 技术路线（推荐架构）

### 模块拆分

- `voicetyper.audio`：设备选择、采样率能力探测、录音参数
- `voicetyper.recognition`：录音分段、拼接修正、事件回调（结果/修正/错误）
- `voicetyper.models`：离线模型的下载、校验、加载、推理封装
- `voicetyper.downloads`：通用下载/解压封装（进度、超时/重试、安全解压）
- `examples/`：仅演示，不作为库 API 的一部分

### 数据流

1. 音频采集（麦克风）
2. VAD/切段（当前由 speech_recognition 的静音检测驱动；后续可替换为独立 VAD）
3. ASR 推理（默认 SenseVoiceSmall；失败回退 Google）
4. 后处理：末尾标点去除 → 纠错词表替换 → （规划中）LLM 纠错
5. 拼接修正（短间隔片段拼接后重识并覆盖上一条文本）
6. 上层使用方消费：CLI 打印 / GUI 更新 / 服务端 API

## 风险清单与对策

- Windows 平台二进制依赖复杂（DLL load failed）
  - 对策：固定可用版本范围；提供自检脚本与清晰的故障指引
- 低功耗设备上模型推理/重采样消耗过高
  - 对策：统一 16kHz 输入；VAD 前置；量化模型；限制拼接最大时长
- “修正覆盖”导致 UI 抖动
  - 对策：调优 `stitch_threshold`；增加最小字数/置信度门槛再触发覆盖
- 无语音时“幻觉输出”（标点/短英文词）
  - 对策：增加静音门限（RMS 过滤）或引入独立 VAD

## 待办（与代码 TODO 同步维护）

- [ ] 修复 sherpa-onnx 重采样日志过长问题（强制 16kHz 录音或屏蔽 C++ 日志）
- [x] 增加 Push-to-talk 示例（长按左 Ctrl 收音，松开识别）
- [x] 增加 UI push-to-talk 示例（悬浮窗 + 剪贴板 + 光标输入 + 托盘退出）
- [x] `SenseVoiceSmallEngine` 参数化：量化开关、线程数、末尾标点去除开关
- [x] 基于 TSV 词表的识别结果纠错（`corrections.tsv`）
- [x] UI push-to-talk 热键防误触（Win32 物理按键状态校验）
- [x] 模型加载计时
- [x] 增加轻量资源监控器（`ResourceMonitor`，支持 CPU/RAM 与可选 GPU 显存输出）
- [x] 文本输入绕过输入法（Win32 `SendInput` + `KEYEVENTF_UNICODE`）
- [ ] 将文本输入方式抽象为跨平台接口（当前仅 Windows `SendInput`；需支持 Linux X11/Wayland、macOS Quartz）
- [ ] 引入 LLM 后处理纠错（可选，用于修正专有名词/英文快速语音等场景；支持云端 API 或本地模型）
- [ ] 增加依赖自检命令（打印 Python 位数、VC++ 版本、sherpa-onnx 版本、模型文件完整性）
- [ ] 增加配置层（例如 `VoiceTyperConfig`：模型目录、阈值、引擎选择）
- [ ] 引入可选 VAD（Silero VAD / webrtcvad）；当前已先用 RMS 门限做轻量过滤
- [ ] 添加离线回归测试音频（短句、长句、停顿切段、噪声）

## 代码重构计划

按顺序执行，每步完成后独立提交。

### 第一步：清理遗留代码与废弃 API ✅

- [x] 删除 `src/demo_speech_recognition.py`（已过时的原型，`AudioDeviceResolver` 和 `BackgroundSTT` 已完整重构到库中，无任何文件引用）
- [x] 替换 `recognition.py` 中的 `import audioop`（Python 3.11 deprecated、3.13 removed），改用 `struct` 实现等效 `_rms()` 函数

### 第二步：库代码引入 logging，替换所有 print() ✅

- [x] `models.py`：所有 `print()` 替换为 `logging.getLogger(__name__)` 调用（模型下载进度、加载计时、纠错词表加载等）
- [x] `recognition.py`：引擎初始化失败的 `print()` 警告替换为 `logging.warning()`
- [x] 让使用方（examples）自行配置 `logging.basicConfig()`，库代码不设置日志格式
- [x] `downloads.py` 中的 `make_console_*_progress` 为调用方提供的进度回调（终端 UI 行为），保持 `sys.stdout.write` 不变

### 第三步：拆分 BackgroundSTT.start_worker，加 context manager 和线程安全 ✅

- [x] 提取 `_reset_stitch_state()` 方法（消除 3 处重复的状态重置代码块）
- [x] 将 148 行的 `worker()` 嵌套函数拆分为实例方法：`_worker_loop()`, `_process_audio()`, `_is_silence()`, `_try_stitch()`, `_recognize()`, `_save_overlap()`
- [x] 回调引用存为实例属性（`_on_status` 等），消除嵌套函数闭包捕获
- [x] 为 `BackgroundSTT` 添加 `__enter__` / `__exit__` context manager 支持
- [x] 添加 `threading.Lock`（`_stitch_lock`）保护 `_last_audio` 等拼接状态的跨线程访问

### 第四步：提取 PushToTalkRecorder 到库中，解耦 speech_recognition 依赖

- [ ] 将 `PushToTalkRecorder` 和 `RecorderConfig` 从 examples 提取到 `src/voicetyper/recorder.py`
- [ ] 消除 `demo_push_to_talk.py` 和 `demo_push_to_talk_ui.py` 中的重复代码，改为从库导入
- [ ] 评估 `BackgroundSTT` 对 `speech_recognition` 类型的耦合，考虑接受通用音频源接口

### 第五步：改用 editable install，去除 sys.path hack，补充单元测试

- [ ] 配置 `pip install -e .`（或 `uv pip install -e .`），确保 `import voicetyper` 可直接使用
- [ ] 移除所有 examples 和 tests 中的 `sys.path.append()` hack
- [ ] 为 `_rms()`、`_apply_corrections`、`_load_corrections`、download retry 逻辑补充单元测试
- [ ] 为拼接逻辑（stitching）补充单元测试（固定音频输入，验证输出）

## 已知局限

- SenseVoiceSmall 对英文快速语音识别精度较低（如专有名词被拆散：`SenseVoiceSmall` → `Since voice is small`）
- SenseVoiceSmall 对发音相近的中文字偶有混淆（如"输入"→"叔入"），量化模型（int8）比全精度（fp32）更明显
- 纠错词表（`corrections.tsv`）为静态规则，无法自动适应新词；LLM 后处理纠错可弥补此不足（待实现）
- UI push-to-talk 示例目前仅支持 Windows（依赖 Win32 API：`SendInput`、`GetAsyncKeyState`、`pystray` 托盘）

## 使用建议（团队工作流）

- 以 `PLAN.md` 作为“全局路线图”
- 以 `tasks.md` 或 `TODO.md` 记录更短期、可执行的任务（可从 PLAN.md 派生）
- 每次完成大功能后更新复选框与“当前状态”小节

