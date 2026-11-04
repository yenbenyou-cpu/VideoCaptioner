"""把一段音频按静音切成若干小段，供 whisper.cpp 逐段识别语言。

whisper.cpp 的 ``-l auto`` 只在**每个输入文件**开头做一次语言检测，
所以「一个视频里中/英/日/俄混说」时，整片会被锁定成同一种语言，
其余语言要么被翻译成英文、要么整段乱码。

这里用 FFmpeg 的 ``silencedetect`` 找到句间停顿，把长音频切成
10~12 秒左右的语音片段；再把所有片段作为多个输入文件一次交给
whisper.cpp，它就会对每个片段单独检测语言。

纯标准库实现，不依赖 numpy。
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Iterable, Sequence

from ..utils.logger import setup_logger

logger = setup_logger("asr.chunker")

_CREATE_NO_WINDOW = (
    getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
)

# 分段太长会让两种语言落进同一段（语言检测按段进行），
# 太短又会让 whisper 的语言检测不稳定；一句话一段最稳妥。
DEFAULT_MAX_SECONDS = 20.0
DEFAULT_MIN_SECONDS = 1.0
DEFAULT_PAD_SECONDS = 0.12
# 只有换气级的短停顿才合并；句间停顿直接作为分段边界
DEFAULT_MERGE_GAP = 0.40
DEFAULT_SILENCE_DB = -35
DEFAULT_SILENCE_SECONDS = 0.35

_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def audio_duration(wav_path: str | Path) -> float:
    """返回 WAV 的时长（秒）；读取失败时返回 0。"""
    try:
        with wave.open(str(wav_path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except (wave.Error, OSError, EOFError):
        return 0.0


def detect_silences(
    wav_path: str | Path,
    noise_db: int = DEFAULT_SILENCE_DB,
    min_silence: float = DEFAULT_SILENCE_SECONDS,
    ffmpeg: str = "ffmpeg",
) -> list[tuple[float, float]]:
    """用 FFmpeg silencedetect 找出静音区间，返回 ``[(start, end), ...]``（秒）。"""
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-nostdin",
        "-i",
        str(wav_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("静音检测失败，将按固定长度分段: %s", exc)
        return []

    output = f"{proc.stderr or ''}\n{proc.stdout or ''}"
    pending: float | None = None
    silences: list[tuple[float, float]] = []
    for line in output.splitlines():
        if match := _SILENCE_START.search(line):
            pending = max(0.0, float(match.group(1)))
        elif match := _SILENCE_END.search(line):
            end = max(0.0, float(match.group(1)))
            if pending is not None and end > pending:
                silences.append((pending, end))
            pending = None
    return silences


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _speech_regions(
    duration: float, silences: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in _merge_intervals(silences):
        start = min(start, duration)
        end = min(end, duration)
        if start > cursor + 0.05:
            regions.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 0.05:
        regions.append((cursor, duration))
    return regions or [(0.0, duration)]


def _split_long_region(
    start: float, end: float, max_seconds: float
) -> list[tuple[float, float]]:
    """连续说话没有停顿时，按等长切开，避免单段超过 whisper 窗口。"""
    length = end - start
    if length <= max_seconds:
        return [(start, end)]
    parts = max(1, math.ceil(length / max_seconds))
    step = length / parts
    return [(start + i * step, start + (i + 1) * step) for i in range(parts)]


def plan_chunks(
    duration: float,
    silences: Sequence[tuple[float, float]],
    max_seconds: float = DEFAULT_MAX_SECONDS,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    pad_seconds: float = DEFAULT_PAD_SECONDS,
    merge_gap: float = DEFAULT_MERGE_GAP,
) -> list[tuple[float, float]]:
    """把语音区间打包成若干个 ``(start, end)`` 分段。

    规则：
    * 每段不超过 ``max_seconds``（whisper 窗口内）；
    * 只跨越很短的换气停顿合并，遇到较长的句间停顿就断开——
      语言通常在同一说话人/同一段落内切换，长停顿是最可靠的分界；
    * 边界向两侧静音各扩 ``pad_seconds``，避免切掉辅音。
    """
    if duration <= 0:
        return [(0.0, 0.0)]

    pieces: list[tuple[float, float]] = []
    for start, end in _speech_regions(duration, silences):
        pieces.extend(_split_long_region(start, end, max_seconds))

    packed: list[tuple[float, float]] = []
    for start, end in pieces:
        if packed:
            gap = start - packed[-1][1]
            if gap <= merge_gap and end - packed[-1][0] <= max_seconds:
                packed[-1] = (packed[-1][0], end)
                continue
        packed.append((start, end))

    # 太短的碎片并入相邻段（优先并到前一段，放不下再并到后一段）
    merged: list[tuple[float, float]] = []
    for start, end in packed:
        if merged and (end - start) < min_seconds:
            prev_start, prev_end = merged[-1]
            if end - prev_start <= max_seconds:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    if len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_seconds:
        last_end = merged[-1][1]
        prev_start = merged[-2][0]
        if last_end - prev_start <= max_seconds:
            merged[-2] = (prev_start, last_end)
            merged.pop()

    padded: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(merged):
        prev_end = merged[index - 1][1] if index > 0 else 0.0
        next_start = merged[index + 1][0] if index + 1 < len(merged) else duration
        left = min(pad_seconds, max(0.0, (start - prev_end) / 2))
        right = min(pad_seconds, max(0.0, (next_start - end) / 2))
        padded.append((max(0.0, start - left), min(duration, end + right)))

    # 去掉空段
    return [(max(0.0, s), min(duration, e)) for s, e in padded if e - s > 0.05]


def split_audio(
    wav_path: str | Path,
    chunks: Sequence[tuple[float, float]],
    output_dir: str | Path,
    ffmpeg: str = "ffmpeg",
) -> list[Path]:
    """把每个分段导出成独立 WAV（16k 单声道），返回文件路径列表。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, (start, end) in enumerate(chunks):
        target = output_dir / f"part_{index:03d}.wav"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(wav_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            creationflags=_CREATE_NO_WINDOW,
        )
        if proc.returncode != 0 or not target.is_file():
            raise RuntimeError(
                f"音频分段失败（{start:.2f}s-{end:.2f}s）: "
                f"{(proc.stderr or '').strip()[-300:]}"
            )
        paths.append(target)
    return paths
