from __future__ import annotations

import contextlib
from typing import Any

try:  # pragma: no cover - availability depends on deployment extras.
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def get_hardware_snapshot() -> dict[str, Any]:
    """Return a best-effort hardware snapshot without requiring GPU libraries."""
    warnings: list[str] = []
    ram_total_bytes = None
    ram_available_bytes = None
    cpu_count = None

    if psutil is None:
        warnings.append("psutil_unavailable")
    else:
        try:
            memory = psutil.virtual_memory()
            ram_total_bytes = int(getattr(memory, "total", 0))
            ram_available_bytes = int(getattr(memory, "available", 0))
        except Exception:
            warnings.append("psutil_memory_unavailable")
        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_count = int(cpu_count) if cpu_count is not None else None
        except Exception:
            warnings.append("psutil_cpu_unavailable")

    gpus, gpu_warnings = load_nvml_snapshot()
    warnings.extend(gpu_warnings)

    return {
        "ram_total_bytes": ram_total_bytes,
        "ram_available_bytes": ram_available_bytes,
        "cpu_count": cpu_count,
        "gpus": gpus,
        "warnings": warnings,
    }


def load_nvml_snapshot() -> tuple[list[dict[str, Any]], list[str]]:
    """Return NVIDIA GPU data when NVML is available, otherwise a structured warning."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        return [], ["nvml_unavailable"]

    try:
        pynvml.nvmlInit()
        device_count = int(pynvml.nvmlDeviceGetCount())
        gpus: list[dict[str, Any]] = []
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "index": index,
                    "name": str(name),
                    "memory_total_bytes": int(getattr(memory, "total", 0)),
                    "memory_free_bytes": int(getattr(memory, "free", 0)),
                    "memory_used_bytes": int(getattr(memory, "used", 0)),
                }
            )
        return gpus, []
    except Exception:
        return [], ["nvml_probe_failed"]
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
