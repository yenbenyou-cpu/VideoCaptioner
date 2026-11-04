"""Shared FFmpeg command runner (1.3.3 build)."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Callable, Optional

from .logger import setup_logger

logger = setup_logger("ffmpeg_utils")


def run_ffmpeg_command(
    cmd: list,
    progress_callback: Optional[Callable] = None,
    start_message: str = "正在合成",
    done_message: str = "合成完成",
) -> None:
    cmd_str = subprocess.list2cmdline(cmd)
    logger.info("FFmpeg 命令: %s", cmd_str)

    process = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=(
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        ),
    )
    assert process.stderr is not None

    total_duration = None
    current_time = 0.0
    tail: list = []

    while True:
        line = process.stderr.readline()
        if not line:
            break
        line = line.rstrip()
        if line:
            tail.append(line)
            if len(tail) > 30:
                tail.pop(0)

        if not progress_callback:
            continue

        if total_duration is None:
            duration_match = re.search(
                r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", line
            )
            if duration_match:
                h, m, s = map(float, duration_match.groups())
                total_duration = h * 3600 + m * 60 + s

        time_match = re.search(r"time=(\d{2}):(\d{2}):(\d{2}\.\d{2})", line)
        if time_match:
            h, m, s = map(float, time_match.groups())
            current_time = h * 3600 + m * 60 + s

        if total_duration:
            progress_callback(
                f"{round(current_time / total_duration * 100)}", start_message
            )

    process.wait()

    if process.returncode != 0:
        logger.error("FFmpeg 失败 (exit %s): %s", process.returncode, cmd_str)
        for entry in tail[-10:]:
            logger.error("  %s", entry)
        raise RuntimeError(f"FFmpeg 编码失败 (exit code {process.returncode})")

    if progress_callback:
        progress_callback("100", done_message)
    logger.info("FFmpeg 命令完成")
