import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from ...config import MODEL_PATH
from ..utils.logger import setup_logger
from . import audio_chunker, backends
from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR

logger = setup_logger("whisper_asr")

ZH_PROMPT = "你好，我们需要使用简体中文，以下是普通话的句子。"


class WhisperCppASR(BaseASR):
    """whisper.cpp 本地识别（支持 AMD 显卡加速 + 多语言混合视频）。

    自动识别两种引擎：
    * 官方 whisper-cli（Vulkan / HIP / CUDA / Metal，含 CPU 回退）
    * Windows 内置 whisper-cpp.exe（Const-me DirectCompute，A/N/Intel 通用）

    ``multilingual=True`` 时会先按静音把音频切成 10~12 秒的片段，再让
    whisper.cpp 对**每个片段单独做语言检测**。因为 whisper.cpp 的
    ``-l auto`` 只在每个输入文件开头检测一次，整片一次性喂进去时
    中/英/日/俄混说会被锁定成同一种语言。
    """

    def __init__(
        self,
        audio_path,
        language="en",
        whisper_cpp_path=None,
        whisper_model=None,
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
        device: str = "auto",
        gpu_index: int = -1,
        model_path=None,
        model_dir=None,
        program=None,
        multilingual: bool = False,
    ):
        super().__init__(audio_path, use_cache, need_word_time_stamp)
        assert os.path.exists(self.audio_path), f"音频文件 {self.audio_path} 不存在"
        assert self.audio_path.endswith(".wav"), f"音频文件 {self.audio_path} 必须是WAV格式"

        self.multilingual = bool(multilingual)
        # 多语言模式下必须让 whisper 自己判断语言，用户选择的源语言会被忽略
        self.language = "auto" if self.multilingual else (language or "auto")
        self.need_word_time_stamp = need_word_time_stamp
        self.process = None
        self.detected_languages: list[str] = []

        # 引擎探测 -----------------------------------------------------------
        requested_program = program or whisper_cpp_path
        self.engine = backends.find_engine(
            requested_program,
            prefer="cpu" if str(device).lower() == "cpu" else "gpu",
        )
        if self.engine is None:
            raise RuntimeError(
                "未找到可用的 whisper.cpp 程序（whisper-cli / whisper-cpp）。"
                "请把 whisper-cli.exe 放入 resource\\bin\\whisper-cpp-vulkan，"
                "或在设置中填写程序路径。"
            )
        self.whisper_cpp_path = self.engine.path
        self.device_plan = backends.plan_device(self.engine, device, gpu_index)
        logger.info("WhisperCpp 引擎: %s", self.engine.describe())
        logger.info("WhisperCpp 设备: %s", self.device_plan.reason)

        # 模型解析 -----------------------------------------------------------
        self.model_path = self._resolve_model(whisper_model, model_path, model_dir)
        logger.info("找到模型文件: %s", self.model_path)
        if self.multilingual:
            logger.info("已启用多语言混合识别：将按静音分段、逐段检测语言")

    @staticmethod
    def _resolve_model(whisper_model, model_path, model_dir=None):
        if model_path:
            explicit = Path(model_path).expanduser()
            if explicit.is_file():
                return str(explicit)
            raise ValueError(f"指定的 whisper.cpp 模型文件不存在: {model_path}")

        if not whisper_model:
            raise ValueError("whisper_model 不能为空")

        search_dirs = []
        if model_dir:
            custom_dir = Path(model_dir).expanduser()
            if not custom_dir.is_dir():
                raise ValueError(f"whisper.cpp 模型目录不存在: {model_dir}")
            search_dirs.append(custom_dir)

        default_dir = Path(MODEL_PATH)
        if default_dir.is_dir():
            search_dirs.append(default_dir)

        if not search_dirs:
            raise ValueError(f"模型目录不存在: {default_dir}，请先下载模型")

        for directory in search_dirs:
            exact = directory / f"ggml-{whisper_model}.bin"
            if exact.is_file():
                return str(exact)

        for directory in search_dirs:
            model_files = sorted(directory.glob(f"*ggml*{whisper_model}*.bin"))
            if model_files:
                return str(model_files[0])

        searched = "、".join(str(d) for d in search_dirs)
        raise ValueError(
            f"未找到模型 '{whisper_model}'，已搜索: {searched}。"
            "请下载模型，或在设置中指定模型目录/模型文件。"
        )

    # ------------------------------------------------------------------
    # 命令行
    # ------------------------------------------------------------------
    def _engine_args(self, device_args: Sequence[str]) -> list[str]:
        """引擎 + 模型 + 语言 + 输出格式，不含输入文件。"""
        args = [
            str(self.engine.path),
            "-m",
            str(self.model_path),
            "-l",
            self.language or "auto",
            "-osrt",
        ]
        if self.engine.flavor == "upstream":
            args.extend(["-pp", "-t", str(self._thread_count())])
        args.extend(device_args)
        if self.language == "zh" and self.engine.flavor == "upstream":
            args.extend(["--prompt", ZH_PROMPT])
        return args

    @staticmethod
    def _thread_count():
        return max(1, min(8, os.cpu_count() or 4))

    def _expected_output(self, temp_dir: Path, out_base):
        if self.engine.flavor == "upstream" and out_base:
            return temp_dir / f"{out_base}.srt"
        return temp_dir / "audio.wav.srt"

    def _make_segments(self, resp_data: str) -> list[ASRDataSeg]:
        asr_data = ASRData.from_srt(resp_data)
        filtered_segments = []
        for seg in asr_data.segments:
            text = seg.text.strip()
            if not (
                text.startswith("【")
                or text.startswith("[")
                or text.startswith("(")
                or text.startswith("（")
            ):
                filtered_segments.append(seg)
        return filtered_segments

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------
    def _run(self, callback=None) -> str:
        if callback is None:
            callback = lambda x, y: None

        temp_dir = self._new_temp_dir()
        try:
            if self.multilingual:
                if self.engine.flavor == "upstream":
                    return self._run_multilingual(temp_dir, callback)
                logger.warning(
                    "内置 DirectCompute 引擎不支持逐段语言检测，"
                    "本次使用整段自动识别；建议改用 whisper-cli（Vulkan 版）"
                )
            return self._run_single(temp_dir, callback)
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 - 统一包装成可读错误
            logger.exception("处理失败")
            raise RuntimeError(f"生成 SRT 文件失败: {str(e)}") from e
        finally:
            self._stop_process()
            self._cleanup_temp(temp_dir)

    def _run_single(self, temp_dir: Path, callback: Callable) -> str:
        wav_path = temp_dir / "audio.wav"
        shutil.copy2(self.audio_path, wav_path)

        out_base = "out" if self.engine.flavor == "upstream" else None
        output_path = self._expected_output(temp_dir, out_base)

        def build(device_args: Sequence[str]) -> list[str]:
            cmd = self._engine_args(device_args) + ["-f", wav_path.name]
            if out_base:
                cmd.extend(["-of", out_base])
            return cmd

        total_duration = self._audio_seconds() or 600
        logger.info("音频总时长: %d 秒", int(total_duration))

        rc, tail = self._execute_with_fallback(
            build, temp_dir, total_duration, callback
        )
        if rc != 0:
            details = "\n".join(tail[-8:])
            raise RuntimeError(
                f"WhisperCPP 执行失败 (exit code {rc})"
                + (f":\n{details}" if details else "")
            )

        # 等原生程序把 SRT 落盘
        output_path = self._wait_for_srt(temp_dir, output_path)
        callback(100, "转换完成")
        return output_path.read_text(encoding="utf-8", errors="replace")

    def _run_multilingual(self, temp_dir: Path, callback: Callable) -> str:
        duration = self._audio_seconds() or 600
        silences = audio_chunker.detect_silences(self.audio_path)
        chunks = audio_chunker.plan_chunks(duration, silences)
        logger.info("多语言混合识别：音频 %.1f 秒，切分为 %d 段", duration, len(chunks))

        if len(chunks) <= 1:
            # 没有可切的停顿，退回单次识别（仍然是 -l auto）
            logger.info("未找到可用停顿，退回整段自动识别")
            return self._run_single(temp_dir, callback)

        paths = audio_chunker.split_audio(
            self.audio_path, chunks, temp_dir / "parts"
        )
        chunk_seconds = [max(0.1, end - start) for start, end in chunks]

        def build(device_args: Sequence[str]) -> list[str]:
            cmd = self._engine_args(device_args)
            for path in paths:
                cmd.extend(["-f", str(path)])
            return cmd

        rc, tail = self._execute_with_fallback(
            build, temp_dir, sum(chunk_seconds), callback, chunk_seconds
        )
        if rc != 0:
            details = "\n".join(tail[-8:])
            raise RuntimeError(
                f"WhisperCPP 多语言分段识别失败 (exit code {rc})"
                + (f":\n{details}" if details else "")
            )

        self._log_languages()
        merged = self._merge_chunk_srts(chunks, paths)
        if not merged.has_data():
            details = "\n".join(tail[-8:])
            raise RuntimeError(
                "WhisperCPP 未生成有效字幕" + (f":\n{details}" if details else "")
            )
        callback(100, "转换完成")
        return merged.to_srt()

    def _merge_chunk_srts(
        self, chunks: Sequence[tuple[float, float]], paths: Sequence[Path]
    ) -> ASRData:
        """把每段 SRT 的时间轴平移到整片起点后合并。"""
        segments: list[ASRDataSeg] = []
        missing = 0
        for (start, _end), path in zip(chunks, paths):
            srt_path = Path(str(path) + ".srt")
            if not srt_path.is_file():
                missing += 1
                logger.warning("分段字幕缺失: %s", srt_path.name)
                continue
            try:
                data = ASRData.from_srt(
                    srt_path.read_text(encoding="utf-8", errors="replace")
                )
            except Exception:  # noqa: BLE001 - 单个分段坏掉不应拖垮整片
                missing += 1
                logger.exception("解析分段字幕失败: %s", srt_path.name)
                continue
            offset = int(round(start * 1000))
            for seg in data.segments:
                seg.start_time += offset
                seg.end_time += offset
                segments.append(seg)

        if missing:
            logger.warning("有 %d 个分段没有产出字幕", missing)

        # 去重 / 修正相邻分段的重叠
        cleaned: list[ASRDataSeg] = []
        for seg in sorted(segments, key=lambda item: item.start_time):
            if cleaned and seg.start_time < cleaned[-1].end_time:
                seg.start_time = cleaned[-1].end_time
            if seg.end_time > seg.start_time:
                cleaned.append(seg)
        return ASRData(cleaned)

    def _log_languages(self) -> None:
        """记录每段检测到的语言，方便确认混合语言是否识别正确。"""
        if not self.detected_languages:
            return
        summary = "、".join(
            f"{index}:{lang}" for index, lang in enumerate(self.detected_languages)
        )
        logger.info("分段语言检测结果: %s", summary)
        unique = list(dict.fromkeys(self.detected_languages))
        if len(unique) > 1:
            logger.info("检测到多语言混说: %s", ", ".join(unique))

    # ------------------------------------------------------------------
    # 子进程执行 / 进度
    # ------------------------------------------------------------------
    def _execute_with_fallback(
        self,
        build: Callable[[Sequence[str]], list[str]],
        cwd: Path,
        total_seconds: float,
        callback: Callable,
        file_seconds: Optional[Sequence[float]] = None,
    ) -> tuple[int, list[str]]:
        rc, tail = self._execute(
            build(self.device_plan.args), cwd, total_seconds, callback, file_seconds
        )
        # 显卡驱动偶发崩溃 / 显存不足时自动回退 CPU，避免整个任务失败
        if (
            rc != 0
            and self.device_plan.use_gpu
            and self.engine.flavor == "upstream"
        ):
            logger.warning("显卡推理失败 (exit %s)，自动回退 CPU 重试", rc)
            callback(1, "显卡加速失败，正在回退 CPU")
            self._clear_srt_outputs(cwd)
            cpu_rc, cpu_tail = self._execute(
                build(("-ng",)), cwd, total_seconds, callback, file_seconds
            )
            if cpu_rc == 0:
                logger.info("CPU 回退成功")
                return cpu_rc, cpu_tail
            logger.error("CPU 回退仍然失败")
            return rc, tail + ["--- CPU 回退也失败 ---"] + cpu_tail
        return rc, tail

    def _execute(
        self,
        cmd: Sequence[str],
        cwd: Path,
        total_seconds: float,
        callback: Callable,
        file_seconds: Optional[Sequence[float]] = None,
    ) -> tuple[int, list[str]]:
        logger.info("完整命令行参数: %s", subprocess.list2cmdline(list(cmd)))
        self.process = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )

        # 后台读线程：避免阻塞式 readline 卡死
        lines: "queue.Queue" = queue.Queue()
        tail: list[str] = []

        def _pump(stream):
            try:
                for line in iter(stream.readline, ""):
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(
            target=_pump, args=(self.process.stdout,), daemon=True
        )
        reader.start()

        completed = 0
        last_progress = 0
        total_seconds = max(1.0, float(total_seconds))
        segments_total = len(file_seconds) if file_seconds else 0

        def _report(fraction: float):
            nonlocal last_progress
            value = max(0, min(int(fraction * 99), 99))
            if value > last_progress:
                last_progress = value
                callback(value, f"{value}% 正在转换")

        try:
            while True:
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    if self.process.poll() is not None and lines.empty():
                        break
                    continue
                if line is None:
                    break

                text = line.strip()
                if not text:
                    continue
                tail.append(text)
                if len(tail) > 60:
                    tail.pop(0)

                if "auto-detected language" in text:
                    if match := re.search(
                        r"auto-detected language:\s*([A-Za-z_\-]+)", text
                    ):
                        self.detected_languages.append(match.group(1))
                        logger.info(
                            "第 %d 段语言: %s",
                            len(self.detected_languages),
                            match.group(1),
                        )
                    continue

                if "output_srt: saving output" in text:
                    completed += 1
                    if file_seconds and segments_total:
                        done = sum(file_seconds[: completed])
                        _report(min(1.0, done / total_seconds))
                    continue

                # whisper-cli -pp 的输出
                if match := re.search(r"progress\s*=\s*(\d+)%", text):
                    percent = min(100, int(match.group(1))) / 100.0
                    if file_seconds and segments_total:
                        index = min(completed, segments_total - 1)
                        done = sum(file_seconds[: index])
                        current = file_seconds[index] * percent
                        _report(min(1.0, (done + current) / total_seconds))
                    else:
                        _report(percent)
                    continue

                if " --> " in text and "[" in text:
                    try:
                        time_str = text.split("[")[1].split(" -->")[0].strip()
                        current_time = sum(
                            float(x) * y
                            for x, y in zip(
                                reversed(time_str.split(":")), [1, 60, 3600]
                            )
                        )
                        if file_seconds and segments_total:
                            index = min(completed, segments_total - 1)
                            done = sum(file_seconds[: index])
                            _report(
                                min(1.0, (done + current_time) / total_seconds)
                            )
                        else:
                            _report(min(1.0, current_time / total_seconds))
                    except (ValueError, IndexError):
                        continue
        finally:
            try:
                returncode = self.process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self.process.kill()
                returncode = self.process.wait()
            reader.join(timeout=3.0)
            try:
                if self.process.stdout:
                    self.process.stdout.close()
            except Exception:  # noqa: BLE001
                pass
            self.process = None

        return returncode, tail

    def _clear_srt_outputs(self, cwd: Path) -> None:
        # 分段模式下 SRT 写在 parts/ 子目录里，必须递归清理，
        # 否则 CPU 重试会和上次显卡失败留下的半成品混在一起
        for srt in Path(cwd).rglob("*.srt"):
            try:
                srt.unlink()
            except OSError:
                pass

    def _wait_for_srt(self, temp_dir: Path, output_path: Path) -> Path:
        for _ in range(50):
            if output_path.exists():
                return output_path
            time.sleep(0.2)
        produced = sorted(temp_dir.glob("*.srt"))
        if produced:
            if len(produced) > 1:
                produced.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return produced[0]
        raise RuntimeError("WhisperCPP 未生成字幕文件")

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------
    @staticmethod
    def _new_temp_dir() -> Path:
        root = Path(tempfile.gettempdir()) / "bk_asr"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="whisper-cpp-", dir=str(root)))

    def _stop_process(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except Exception:  # noqa: BLE001
                pass
        self.process = None

    @staticmethod
    def _cleanup_temp(temp_dir: Path) -> None:
        """Windows 上子进程/杀毒软件可能短暂占用文件，重试几次且不再抛异常。"""
        for attempt in range(6):
            try:
                shutil.rmtree(temp_dir)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                if attempt == 5:
                    logger.warning("临时目录清理失败（可手动删除）: %s (%s)", temp_dir, exc)
                    return
                time.sleep(0.15 * (attempt + 1))

    # ------------------------------------------------------------------
    # 其它
    # ------------------------------------------------------------------
    def _audio_seconds(self) -> float:
        seconds = audio_chunker.audio_duration(self.audio_path)
        if seconds > 0:
            return seconds
        return float(self.get_audio_duration(self.audio_path) or 0)

    def get_audio_duration(self, filepath: str) -> int:
        try:
            cmd = ["ffmpeg", "-i", filepath]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            info = result.stderr
            if duration_match := re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info):
                hours, minutes, seconds = map(float, duration_match.groups())
                return int(hours * 3600 + minutes * 60 + seconds)
            return 600
        except Exception as e:
            logger.exception("获取音频时长时出错: %s", str(e))
            return 600

    def _get_key(self):
        return (
            f"{self.crc32_hex}-{self.need_word_time_stamp}-{self.model_path}-"
            f"{self.language}-{self.engine.path}-{self.engine.display_backend}-"
            f"{self.device_plan.use_gpu}-{self.device_plan.gpu_index}-"
            f"ml{int(self.multilingual)}"
        )


if __name__ == "__main__":
    asr = WhisperCppASR(
        audio_path="audio.mp3",
        whisper_model="tiny",
        whisper_cpp_path="bin/whisper-cpp.exe",
        language="en",
        need_word_time_stamp=True,
    )
    asr_data = asr._run(callback=print)
