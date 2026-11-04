"""Pick the best H.264 encoder for hard-subtitle burn-in (1.3.3 build)."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .logger import setup_logger

logger = setup_logger("video_encoder")

_ENCODER_LINE = re.compile(r"^\s*[VAS][\w.]*\s+(\S+)", re.MULTILINE)

# x264 preset -> hardware QP.  Hardware encoders use constant QP instead of CRF;
# these values keep the output size close to the software path.
_PRESET_QP = {
    "ultrafast": 32,
    "superfast": 31,
    "veryfast": 30,
    "faster": 29,
    "fast": 29,
    "medium": 28,
    "slow": 26,
    "slower": 25,
    "veryslow": 24,
}

_VENDOR_ENCODERS = {
    "amd": ("h264_amf",),
    "nvidia": ("h264_nvenc",),
    "intel": ("h264_qsv",),
}


@dataclass(frozen=True)
class EncoderChoice:
    name: str
    kind: str
    args: tuple

    @property
    def is_hardware(self) -> bool:
        return self.kind not in ("libx264", "libvpx-vp9")


def _creation_flags() -> int:
    import os

    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


@lru_cache(maxsize=1)
def available_encoders() -> frozenset:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=_creation_flags(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to list FFmpeg encoders: %s", exc)
        return frozenset()
    return frozenset(_ENCODER_LINE.findall(result.stdout or ""))


def _vendor_order() -> list:
    try:
        from ..bk_asr.backends import gpu_vendors

        vendors = gpu_vendors()
    except Exception:  # noqa: BLE001
        vendors = set()

    order: list = []
    for vendor in ("amd", "nvidia", "intel"):
        if vendor in vendors:
            order.extend(_VENDOR_ENCODERS[vendor])
    for vendor in ("amd", "nvidia", "intel"):
        for name in _VENDOR_ENCODERS[vendor]:
            if name not in order:
                order.append(name)
    return order


def _hardware_args(name: str, qp: int) -> tuple:
    if name == "h264_amf":
        return (
            "-c:v", "h264_amf",
            "-quality", "quality",
            "-rc", "cqp",
            "-qp_i", str(qp),
            "-qp_p", str(qp + 2),
            "-qp_b", str(qp + 4),
            "-pix_fmt", "yuv420p",
        )
    if name == "h264_nvenc":
        return (
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-tune", "hq",
            "-rc", "constqp",
            "-qp", str(qp + 2),
            "-pix_fmt", "yuv420p",
        )
    if name == "h264_qsv":
        return (
            "-c:v", "h264_qsv",
            "-preset", "medium",
            "-global_quality", str(qp + 2),
            "-pix_fmt", "nv12",
        )
    return ("-c:v", name, "-pix_fmt", "yuv420p")


def _software_args(preset: str) -> tuple:
    return ("-c:v", "libx264", "-preset", preset)


def select_encoder(
    *,
    prefer: str = "auto",
    preset: str = "medium",
    output_suffix: str = ".mp4",
) -> EncoderChoice:
    suffix = (output_suffix or "").lower()
    if suffix == ".webm":
        return EncoderChoice(
            name="libvpx-vp9", kind="libvpx-vp9", args=("-c:v", "libvpx-vp9")
        )

    if (prefer or "auto").lower() != "cpu":
        encoders = available_encoders()
        qp = _PRESET_QP.get(preset, 28)
        for name in _vendor_order():
            if name in encoders:
                logger.info("Hardware encoder: %s (QP %d)", name, qp)
                return EncoderChoice(name=name, kind=name, args=_hardware_args(name, qp))
        logger.debug("No hardware encoder available; using libx264")

    return EncoderChoice(
        name="libx264", kind="libx264", args=_software_args(preset)
    )


def software_encoder(preset: str) -> EncoderChoice:
    return EncoderChoice(
        name="libx264", kind="libx264", args=_software_args(preset)
    )


def hardware_encoders() -> list:
    encoders = available_encoders()
    return [name for name in _vendor_order() if name in encoders]
