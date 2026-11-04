import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Literal, Optional

from ..utils.logger import setup_logger
from ..utils.ass_auto_wrap import auto_wrap_ass_file
from .ffmpeg_utils import run_ffmpeg_command
from .video_encoder import select_encoder, software_encoder

logger = setup_logger("video_utils")


def video2audio(input_file: str, output: str = "") -> bool:
    """使用ffmpeg将视频转换为音频"""
    # 创建output目录
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output = str(output)
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-map",
        "0:a",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "aresample=async=1",  # 处理音频同步问题
        "-y",
        output,
    ]
    logger.info(f"转换为音频执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            ),
        )
        if result.returncode == 0 and Path(output).is_file():
            return True
        else:
            logger.error("音频转换失败")
            return False
    except Exception as e:
        logger.exception(f"音频转换出错: {str(e)}")
        return False


def check_cuda_available() -> bool:
    """检查CUDA是否可用"""
    logger.info("检查CUDA是否可用")
    try:
        # 首先检查ffmpeg是否支持cuda
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            capture_output=True,
            text=True,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            ),
        )
        if "cuda" not in result.stdout.lower():
            logger.info("CUDA不在支持的硬件加速器列表中")
            return False

        # 进一步检查CUDA设备信息
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-init_hw_device", "cuda"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        # 如果stderr中包含"Cannot load cuda" 或 "Failed to load"等错误信息，说明CUDA不可用
        if any(
            error in result.stderr.lower()
            for error in ["cannot load cuda", "failed to load", "error"]
        ):
            logger.info("CUDA设备初始化失败")
            return False

        logger.info("CUDA可用")
        return True

    except Exception as e:
        logger.exception(f"检查CUDA出错: {str(e)}")
        return False


def add_subtitles(
    input_file: str,
    subtitle_file: str,
    output: str,
    quality: Literal[
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ] = "medium",
    vcodec: str = "libx264",
    soft_subtitle: bool = False,
    progress_callback: callable = None,
    prefer: str = "auto",
) -> None:
    assert Path(input_file).is_file(), "输入文件不存在"
    assert Path(subtitle_file).is_file(), "字幕文件不存在"

    # 移动到临时文件  Fix: 路径错误
    # 每次合成都用独立目录：固定 temp_subtitle.ass 在批量任务和手动任务
    # 同时运行时会互相覆盖，导致烧进错误的字幕。
    suffix = Path(subtitle_file).suffix.lower()
    temp_root = Path(tempfile.gettempdir()) / "VideoCaptioner"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="subtitle-", dir=str(temp_root)))
    try:
        temp_subtitle = temp_dir / f"temp_subtitle{suffix}"
        shutil.copy2(subtitle_file, temp_subtitle)
        subtitle_file = str(temp_subtitle)

        # video_info = get_video_info(input_file)
        if suffix == ".ass":
            subtitle_file = auto_wrap_ass_file(
                subtitle_file,
                # video_width=video_info["width"],
                # video_height=video_info["height"],
            )

        # 如果是WebM格式，强制使用硬字幕
        if Path(output).suffix.lower() == ".webm":
            soft_subtitle = False
            logger.info("WebM格式视频，强制使用硬字幕")

        if soft_subtitle:
            # 添加软字幕
            cmd = [
                "ffmpeg",
                "-i",
                input_file,
                "-i",
                subtitle_file,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-c:s",
                "mov_text",
                output,
                "-y",
            ]
            logger.info(f"添加软字幕执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, "CREATE_NO_WINDOW")
                    else 0
                ),
            )
            # 之前不检查返回值：封装失败也会被当成「合成完成」，输出文件根本不存在
            if result.returncode != 0 or not Path(output).is_file():
                logger.error("添加软字幕失败: %s", (result.stderr or "")[-800:])
                raise RuntimeError(
                    f"添加软字幕失败 (exit code {result.returncode})"
                )
        else:
            logger.info("使用硬字幕")
            subtitle_file = Path(subtitle_file).as_posix().replace(":", r"\:")
            # 根据输出文件后缀决定vf参数
            if Path(output).suffix.lower() == ".ass":
                vf = f"ass='{subtitle_file}'"
            else:
                # 其他格式使用默认的vf参数
                vf = f"subtitles='{subtitle_file}'"

            if Path(output).suffix.lower() == ".webm":
                vcodec = "libvpx-vp9"
                logger.info("WebM格式视频，使用libvpx-vp9编码器")

            # 选择编码器：有可用显卡时用 AMF/NVENC/QSV，否则 libx264
            choice = select_encoder(
                prefer=prefer,
                preset=quality,
                output_suffix=Path(output).suffix,
            )

            def _build_command(encoder):
                return [
                    "ffmpeg",
                    "-i",
                    input_file,
                    "-acodec",
                    "copy",
                    *encoder.args,
                    "-vf",
                    vf,
                    "-y",  # 覆盖输出文件
                    output,
                ]

            try:
                run_ffmpeg_command(_build_command(choice), progress_callback)
            except Exception:
                if not choice.is_hardware:
                    raise
                logger.warning("硬件编码 %s 失败，回退 libx264", choice.name)
                if Path(output).exists():
                    Path(output).unlink()
                run_ffmpeg_command(
                    _build_command(software_encoder(quality)), progress_callback
                )
    finally:
        # 无论成功失败都清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_video_info(file_path: str) -> Optional[Dict]:
    """获取视频信息"""
    try:
        cmd = ["ffmpeg", "-i", file_path]

        # logger.info(f"获取视频信息执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            ),
        )
        info = result.stderr

        video_info_dict = {
            "file_name": Path(file_path).stem,
            "file_path": file_path,
            "duration_seconds": 0,
            "bitrate_kbps": 0,
            "video_codec": "",
            "width": 0,
            "height": 0,
            "fps": 0,
            "audio_codec": "",
            "audio_sampling_rate": 0,
            "thumbnail_path": "",
        }

        # 提取时长
        if duration_match := re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", info):
            hours, minutes, seconds = map(float, duration_match.groups())
            video_info_dict["duration_seconds"] = hours * 3600 + minutes * 60 + seconds

        # 提取比特率
        if bitrate_match := re.search(r"bitrate: (\d+) kb/s", info):
            video_info_dict["bitrate_kbps"] = int(bitrate_match.group(1))

        # 提取视频流信息
        if video_stream_match := re.search(
            r"Stream #.*?Video: (\w+)(?:\s*\([^)]*\))?.* (\d+)x(\d+).*?(?:(\d+(?:\.\d+)?)\s*(?:fps|tb[rn]))",
            info,
            re.DOTALL,
        ):
            video_info_dict.update(
                {
                    "video_codec": video_stream_match.group(1),
                    "width": int(video_stream_match.group(2)),
                    "height": int(video_stream_match.group(3)),
                    "fps": float(video_stream_match.group(4)),
                }
            )
        else:
            logger.warning("未找到视频流信息")

        return video_info_dict
    except Exception as e:
        logger.exception(f"获取视频信息时出错: {str(e)}")
        return None
