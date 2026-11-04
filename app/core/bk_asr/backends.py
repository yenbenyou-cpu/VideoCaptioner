"""Local whisper.cpp engine discovery and GPU backend detection.

VideoCaptioner can drive two different flavours of whisper.cpp binaries:

``upstream``
    The official CLI built from `ggml-org/whisper.cpp <https://github.com/ggml-org/whisper.cpp>`_
    (``whisper-cli``). GPU acceleration depends on the backends compiled into the
    binary: Vulkan (works on AMD/Intel/NVIDIA), CUDA, HIP/ROCm, Metal or SYCL.
    The CPU path is always available through ``--no-gpu``.

``directcompute``
    Const-me's DirectCompute port that ships with the Windows build
    (``whisper-cpp.exe`` + ``Whisper.dll``). It runs on any Direct3D 11 capable
    GPU, including AMD Radeon cards, but has no CPU mode.

This module keeps all platform probing here so that the ASR implementation and
the UI can stay declarative.
"""

from __future__ import annotations

import ctypes
import glob
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Optional

from ..utils.logger import setup_logger

logger = setup_logger("asr.backends")

Flavor = Literal["upstream", "directcompute"]
DeviceMode = Literal["auto", "gpu", "cpu"]

_GPU_BACKEND_ORDER = ("vulkan", "hip", "cuda", "metal", "sycl")
_SIBLING_BACKEND_DLLS = {
    "vulkan": ("ggml-vulkan*.dll", "ggml_vulkan*.dll"),
    "cuda": ("ggml-cuda*.dll", "ggml_cuda*.dll", "cudart64_*.dll", "cublas64_*.dll"),
    "hip": ("ggml-hip*.dll", "ggml_hip*.dll", "amdhip64*.dll", "rocblas*.dll"),
    "sycl": ("ggml-sycl*.dll", "ggml_sycl*.dll"),
    "metal": ("ggml-metal*",),
}
_BINARY_BACKEND_MARKERS = {
    "vulkan": b"ggml_vulkan",
    "cuda": b"ggml_cuda",
    "hip": b"ggml_hip",
    "sycl": b"ggml_sycl",
    "metal": b"ggml_metal",
}
_VENDOR_NAMES = {
    0x1002: "amd",
    0x1022: "amd",
    0x10DE: "nvidia",
    0x8086: "intel",
    0x13B5: "arm",
    0x5143: "qualcomm",
    0x1010: "imagination",
}
_VK_PHYSICAL_DEVICE_TYPES = {
    0: "other",
    1: "integrated",
    2: "discrete",
    3: "virtual",
    4: "cpu",
}

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


@dataclass(frozen=True)
class GpuAdapter:
    """A graphics adapter usable for inference."""

    index: int
    name: str
    vendor: str = "unknown"
    kind: str = "other"

    @property
    def is_discrete(self) -> bool:
        return self.kind == "discrete"

    def label(self) -> str:
        return f"{self.index}: {self.name}"


@dataclass(frozen=True)
class EngineInfo:
    """A discovered whisper.cpp executable and the backends it was built with."""

    path: Path
    flavor: Flavor
    backends: tuple[str, ...]
    version: str = ""

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def gpu_backend(self) -> Optional[str]:
        """First GPU backend in preference order, or ``None`` for CPU-only builds."""
        for backend in _GPU_BACKEND_ORDER:
            if backend in self.backends:
                return backend
        if "directcompute" in self.backends:
            return "directcompute"
        return None

    @property
    def has_gpu(self) -> bool:
        return self.gpu_backend is not None

    @property
    def supports_cpu(self) -> bool:
        return self.flavor == "upstream"

    @property
    def display_backend(self) -> str:
        if self.flavor == "directcompute":
            return "DirectCompute"
        backend = self.gpu_backend
        return backend.upper() if backend else "CPU"

    def describe(self) -> str:
        if self.flavor == "directcompute":
            return f"{self.name} (DirectCompute)"
        return f"{self.name} (upstream, {self.display_backend})"


@dataclass(frozen=True)
class DevicePlan:
    """How a transcription run should be launched."""

    use_gpu: bool
    gpu_index: int
    args: tuple[str, ...] = ()
    reason: str = ""


def _run_help(path: Path) -> str:
    """Return the CLI help text of an engine, or an empty string on failure."""
    try:
        proc = subprocess.run(
            [str(path), "-h"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
            creationflags=_CREATE_NO_WINDOW,
        )
        return f"{proc.stdout or ''}\n{proc.stderr or ''}"
    except Exception as exc:  # noqa: BLE001 - probing must never crash the app
        logger.debug("Failed to read help from %s: %s", path, exc)
        return ""


def _sibling_backends(path: Path) -> list[str]:
    found: list[str] = []
    directory = path.parent
    for backend, patterns in _SIBLING_BACKEND_DLLS.items():
        for pattern in patterns:
            if any(Path(p).is_file() for p in glob.glob(str(directory / pattern))):
                found.append(backend)
                break
    # Const-me's engine exposes the API through Whisper.dll
    if (directory / "Whisper.dll").is_file() or (directory / "whisper.dll").is_file():
        found.append("directcompute")
    return found


def _binary_backends(path: Path) -> list[str]:
    """Detect statically linked backends by scanning the executable."""
    try:
        if path.stat().st_size > 400 * 1024 * 1024:
            return []
        data = path.read_bytes()
    except OSError:
        return []
    return [name for name, marker in _BINARY_BACKEND_MARKERS.items() if marker in data]


@lru_cache(maxsize=64)
def _detect_engine_cached(path_str: str, size: int, mtime_ns: int) -> Optional[EngineInfo]:
    path = Path(path_str)
    help_text = _run_help(path)
    lower_help = help_text.lower()

    sibling = _sibling_backends(path)
    # Vulkan 版 whisper-cli 同样自带 whisper.dll（小写），不能只凭 DLL 名字
    # 就判定成 Const-me 的 DirectCompute 版，否则会传错 -gpu 参数。
    has_ggml_backend = any(name != "directcompute" for name in sibling)
    if (
        "--list-adapters" in lower_help
        or "directcompute" in lower_help
        or ("directcompute" in sibling and not has_ggml_backend)
    ):
        return EngineInfo(
            path=path,
            flavor="directcompute",
            backends=("directcompute",),
        )

    if "--no-gpu" not in lower_help and not sibling and not _binary_backends(path):
        # Unknown helper; refuse to pretend it is a whisper.cpp CLI.
        return None

    backends = _sibling_backends(path) or _binary_backends(path)
    backends = [b for b in backends if b != "directcompute"]
    backends.append("cpu")
    return EngineInfo(path=path, flavor="upstream", backends=tuple(dict.fromkeys(backends)))


def detect_engine(path: str | Path) -> Optional[EngineInfo]:
    """Inspect a concrete executable and return its capabilities."""
    resolved = Path(path).expanduser()
    try:
        stat = resolved.stat()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    return _detect_engine_cached(str(resolved.resolve()), stat.st_size, stat.st_mtime_ns)


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    for name in ("whisper-cli", "whisper-cpp", "whisper"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    try:
        from app.config import (
            BIN_PATH,
            BUNDLED_BIN_PATH,
            BUNDLED_WHISPER_VULKAN_PATH,
            WHISPER_VULKAN_PATH,
        )
    except Exception:  # pragma: no cover - config import should not fail in practice
        BIN_PATH = BUNDLED_BIN_PATH = None  # type: ignore[assignment]
        WHISPER_VULKAN_PATH = BUNDLED_WHISPER_VULKAN_PATH = None  # type: ignore[assignment]

    for directory in (
        WHISPER_VULKAN_PATH,
        BUNDLED_WHISPER_VULKAN_PATH,
        BIN_PATH,
        BUNDLED_BIN_PATH,
    ):
        if directory is None:
            continue
        for name in ("whisper-cli.exe", "whisper-cpp.exe", "whisper-cli", "whisper-cpp"):
            candidate = Path(directory) / name
            if candidate.is_file():
                candidates.append(candidate)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve()).lower()
        except OSError:
            key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def list_engines() -> list[EngineInfo]:
    """Return every usable whisper.cpp engine found on this machine."""
    engines: list[EngineInfo] = []
    for candidate in _candidate_paths():
        info = detect_engine(candidate)
        if info is not None:
            engines.append(info)
    return engines


def _engine_score(engine: EngineInfo, prefer: str) -> int:
    if prefer == "cpu":
        return 100 if engine.supports_cpu else 0
    if engine.flavor == "upstream" and engine.has_gpu:
        return 100
    if engine.flavor == "directcompute":
        return 90
    if engine.supports_cpu:
        return 40
    return 0


def find_engine(explicit: Optional[str] = None, *, prefer: str = "gpu") -> Optional[EngineInfo]:
    """Pick the best engine.

    Args:
        explicit: User supplied executable path. When given it is used as-is.
        prefer: ``"gpu"`` for hardware acceleration, ``"cpu"`` for the CPU path.
    """
    if explicit:
        info = detect_engine(explicit)
        if info is None:
            raise RuntimeError(
                f"指定的 WhisperCpp 程序不可用: {explicit}. "
                "请选择一个带 --no-gpu 的 whisper-cli，或 Windows 内置的 whisper-cpp.exe。"
            )
        return info

    engines = list_engines()
    if not engines:
        return None
    engines.sort(key=lambda e: _engine_score(e, prefer), reverse=True)
    # Return the best candidate even when it cannot satisfy the preference:
    # plan_device() then raises a precise error instead of "engine not found".
    return engines[0]


def directcompute_adapters(engine: EngineInfo) -> list[str]:
    """List adapters exposed by Const-me's engine (``-la``)."""
    if engine.flavor != "directcompute":
        return []
    try:
        proc = subprocess.run(
            [str(engine.path), "-la"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=15,
            creationflags=_CREATE_NO_WINDOW,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to list DirectCompute adapters: %s", exc)
        return []
    output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    return [m.strip() for m in re.findall(r'^\s*"(.*)"\s*$', output, flags=re.MULTILINE)]


# ---------------------------------------------------------------------------
# Vulkan device enumeration (ctypes, no external dependency)
# ---------------------------------------------------------------------------


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def _vulkan_loader() -> Optional[ctypes.CDLL]:
    for name in ("vulkan-1.dll", "vulkan-1", "libvulkan.so.1", "libvulkan.so", "libvulkan.dylib", "libvulkan.1.dylib"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _vulkan_adapters_ctypes() -> list[GpuAdapter]:
    lib = _vulkan_loader()
    if lib is None:
        return []

    app_info = _VkApplicationInfo(
        sType=0,
        pNext=None,
        pApplicationName=b"VideoCaptioner",
        applicationVersion=1,
        pEngineName=b"VideoCaptioner",
        engineVersion=1,
        apiVersion=(1 << 22),  # Vulkan 1.0
    )
    create_info = _VkInstanceCreateInfo(
        sType=1,
        pNext=None,
        flags=0,
        pApplicationInfo=ctypes.pointer(app_info),
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None,
    )

    instance = ctypes.c_void_p()
    try:
        create = lib.vkCreateInstance
        create.argtypes = [
            ctypes.POINTER(_VkInstanceCreateInfo),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create.restype = ctypes.c_int
        if create(ctypes.byref(create_info), None, ctypes.byref(instance)) != 0:
            return []

        enumerate_fn = lib.vkEnumeratePhysicalDevices
        enumerate_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        enumerate_fn.restype = ctypes.c_int

        count = ctypes.c_uint32(0)
        if enumerate_fn(instance, ctypes.byref(count), None) not in (0, 5) or count.value == 0:
            return []
        devices = (ctypes.c_void_p * count.value)()
        if enumerate_fn(instance, ctypes.byref(count), devices) not in (0, 5):
            return []

        get_props = lib.vkGetPhysicalDeviceProperties
        get_props.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        get_props.restype = None

        adapters: list[GpuAdapter] = []
        for index in range(count.value):
            buffer = ctypes.create_string_buffer(1024)
            get_props(devices[index], buffer)
            vendor_id = ctypes.c_uint32.from_buffer(buffer, 8).value
            device_type = ctypes.c_uint32.from_buffer(buffer, 16).value
            raw_name = buffer.raw[20:276].split(b"\x00", 1)[0]
            name = raw_name.decode("utf-8", "replace") or f"Vulkan device {index}"
            adapters.append(
                GpuAdapter(
                    index=index,
                    name=name,
                    vendor=_VENDOR_NAMES.get(vendor_id, "unknown"),
                    kind=_VK_PHYSICAL_DEVICE_TYPES.get(device_type, "other"),
                )
            )
        return adapters
    except Exception as exc:  # noqa: BLE001 - driver DLLs may misbehave
        logger.debug("Vulkan enumeration failed: %s", exc)
        return []
    finally:
        if instance:
            try:
                destroy = lib.vkDestroyInstance
                destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                destroy.restype = None
                destroy(instance, None)
            except Exception:  # noqa: BLE001
                pass


def _vulkan_adapters_vulkaninfo() -> list[GpuAdapter]:
    executable = shutil.which("vulkaninfo") or shutil.which("vulkaninfoSDK")
    if not executable:
        return []
    try:
        proc = subprocess.run(
            [executable, "--summary"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=20,
        )
    except Exception:  # noqa: BLE001
        return []
    adapters: list[GpuAdapter] = []
    for line in f"{proc.stdout}\n{proc.stderr}".splitlines():
        match = re.match(r"\s*deviceName\s*=\s*(.+)", line)
        if match:
            name = match.group(1).strip()
            adapters.append(
                GpuAdapter(
                    index=len(adapters),
                    name=name,
                    vendor=_guess_vendor(name),
                    kind="discrete" if re.search(r"RX|Radeon|GeForce|RTX|Arc", name, re.I) else "other",
                )
            )
    return adapters


@lru_cache(maxsize=1)
def _cached_vulkan_adapters() -> tuple[GpuAdapter, ...]:
    adapters = _vulkan_adapters_ctypes() if sys.platform in ("win32", "linux") else []
    if not adapters:
        adapters = _vulkan_adapters_vulkaninfo()
    return tuple(adapters)


def vulkan_adapters(refresh: bool = False) -> list[GpuAdapter]:
    """Enumerate adapters through the Vulkan loader.

    The order matches ggml's device indices, so the values can be passed to
    ``whisper-cli --device``.
    """
    if refresh:
        _cached_vulkan_adapters.cache_clear()
    return list(_cached_vulkan_adapters())


def _guess_vendor(name: str) -> str:
    lower = name.lower()
    if "amd" in lower or "radeon" in lower:
        return "amd"
    if "nvidia" in lower or "geforce" in lower or "rtx" in lower:
        return "nvidia"
    if "intel" in lower or "arc" in lower:
        return "intel"
    return "unknown"


def gpu_vendors() -> set[str]:
    """Vendors of the adapters that can actually run inference."""
    vendors = {adapter.vendor for adapter in vulkan_adapters()}
    vendors.discard("unknown")
    return vendors


def best_adapter_index(adapters: Iterable[GpuAdapter]) -> int:
    """Pick the adapter that is most likely to give the best throughput."""
    ranked = sorted(
        adapters,
        key=lambda a: (
            a.is_discrete,
            a.vendor in ("amd", "nvidia"),
            -a.index,
        ),
        reverse=True,
    )
    return ranked[0].index if ranked else -1


def _pick_directcompute_index(engine: EngineInfo, requested: int) -> int:
    if requested >= 0:
        return requested
    adapters = directcompute_adapters(engine)
    if not adapters:
        return -1
    vulkan_discrete = [a.name for a in vulkan_adapters() if a.is_discrete]
    for name in vulkan_discrete:
        if name in adapters:
            return adapters.index(name)
    return 0


def plan_device(
    engine: EngineInfo,
    device: str = "auto",
    gpu_index: int = -1,
) -> DevicePlan:
    """Translate the user's device preference into CLI arguments.

    Raises:
        ValueError: if ``device`` is not one of auto/gpu/cpu.
        RuntimeError: if GPU was requested but the engine cannot use it.
    """
    device = (device or "auto").strip().lower()
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError(f"未知的运行设备: {device!r} (可选 auto/gpu/cpu)")

    if engine.flavor == "directcompute":
        if device == "cpu":
            raise RuntimeError(
                "内置的 DirectCompute 引擎没有 CPU 模式。"
                "如需纯 CPU 识别，请改用设备选项 auto/gpu，"
                "或安装官方 whisper.cpp（whisper-cli）后把程序路径指过去。"
            )
        index = _pick_directcompute_index(engine, gpu_index)
        args = ("-gpu", str(index)) if index >= 0 else ()
        return DevicePlan(
            use_gpu=True,
            gpu_index=index,
            args=args,
            reason="DirectCompute 显卡加速（AMD/NVIDIA/Intel 通用）",
        )

    if device == "cpu" or (device == "auto" and not engine.has_gpu):
        return DevicePlan(
            use_gpu=False,
            gpu_index=-1,
            args=("-ng",),
            reason="CPU 模式（--no-gpu）",
        )

    if not engine.has_gpu:
        raise RuntimeError(
            f"{engine.name} 是 CPU 版 whisper.cpp，没有编译任何 GPU 后端。"
            "请改用带 Vulkan/CUDA/HIP 后端的 whisper-cli，或在设备选项中选择 cpu。"
        )

    index = gpu_index
    if index < 0 and engine.gpu_backend == "vulkan":
        index = best_adapter_index(vulkan_adapters())
    args = ("-dev", str(index)) if index >= 0 else ()
    backend = (engine.gpu_backend or "gpu").upper()
    suffix = f"，设备 {index}" if index >= 0 else ""
    return DevicePlan(
        use_gpu=True,
        gpu_index=index,
        args=args,
        reason=f"{backend} 显卡加速{suffix}",
    )


def reset_cache() -> None:
    """Clear memoized probes (used by the settings dialog refresh button and tests)."""
    _detect_engine_cached.cache_clear()
    _cached_vulkan_adapters.cache_clear()
