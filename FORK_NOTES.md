# Fork 说明：AMD 显卡加速 + 多语言混合识别

本分支基于上游 [WEIFENG2333/VideoCaptioner](https://github.com/WEIFENG2333/VideoCaptioner) 的 **v1.3.3**（提交 `b33b389`），
主要解决两件事：

1. 在 **AMD 显卡** 上用 Vulkan 推理本地 whisper.cpp；
2. 修复「一个视频里中/英/日/俄混说，只能识别出一种语言」的问题。

## 多语言混合识别

`large-v3-turbo` 本身支持多语言，问题出在调用方式：whisper.cpp 的 `-l auto`
**只在每个输入文件开头检测一次语言**，整段视频会被锁定成一种语言。

本分支的做法（`app/core/bk_asr/audio_chunker.py` + `whisper_cpp.py`）：

- FFmpeg `silencedetect` 找句间停顿，把音频切成"一句话一段"（≤20 秒）；
- 每段导出独立 WAV，**一次**进程把所有分段交给 whisper.cpp（模型只加载一次）；
- whisper.cpp 对每个文件单独检测语言；
- 按分段起点平移时间轴，合并、去重、重新编号成一条 SRT。

设置 → 转录 → WhisperCpp → **「多语言混合识别」开关（默认开启）**，开启后忽略"源语言"。

实测：29.4 秒中英各两段音频，逐段语言 = `zh,zh,zh,en,zh,zh,zh,en`；
显卡 10.0 秒 vs 纯 CPU 90.9 秒（RX 7900 XTX + whisper.cpp 1.9.5-dev Vulkan）。

## 其他改动

- 显卡推理失败自动回退 CPU（`-ng` 重试）；
- 临时目录清理不再因 Windows 文件占用报错；
- 修复 Vulkan 目录自带 `whisper.dll` 被误判成 DirectCompute 引擎的问题；
- 修复并发翻译/优化失败时会覆盖最后一块已翻译内容的问题；
- 修复软字幕封装失败仍提示"合成完成"、临时字幕被并发任务覆盖的问题；
- 修复批量合并删掉未选中行、断句对齐丢字、aria2c stderr 卡死、坏样式文件崩进程等问题。

## 注意

- `resource/bin/whisper-cpp-vulkan/`（Vulkan 版 whisper-cli 与 ggml DLL）和 `AppData/models/`
  属于运行时资源，没有纳入仓库；需要自行下载 whisper.cpp 的 Vulkan 构建并放到该目录。
- 转录缓存键包含「多语言」开关，切换后首次会重新识别。
