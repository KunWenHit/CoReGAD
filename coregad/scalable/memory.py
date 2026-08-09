from __future__ import annotations

import os
from typing import Any

import torch


def memory_snapshot(device: torch.device | str | None = None) -> dict[str, Any]:
    resident_bytes: int | None = None
    try:
        import psutil

        resident_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        try:
            import resource

            resident_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if os.name != "nt":
                resident_bytes *= 1024
        except Exception:
            pass
    result: dict[str, Any] = {"cpu_resident_bytes": resident_bytes}
    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and torch.cuda.is_available():
            result.update(
                {
                    "gpu_allocated_bytes": int(torch.cuda.memory_allocated(resolved)),
                    "gpu_reserved_bytes": int(torch.cuda.memory_reserved(resolved)),
                    "gpu_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(resolved)
                    ),
                }
            )
    return result
