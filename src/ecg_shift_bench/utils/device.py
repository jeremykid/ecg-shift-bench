"""CUDA device selection helpers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CUDAGPUInfo:
    """A single GPU entry reported by nvidia-smi."""

    index: int
    uuid: str
    memory_free_mib: int
    memory_total_mib: int
    utilization_gpu: int
    name: str


def _visible_device_tokens() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return tokens or None


def _query_nvidia_smi() -> list[CUDAGPUInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to query nvidia-smi for GPU selection") from error

    devices: list[CUDAGPUInfo] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        devices.append(
            CUDAGPUInfo(
                index=int(parts[0]),
                uuid=parts[1],
                memory_free_mib=int(parts[2]),
                memory_total_mib=int(parts[3]),
                utilization_gpu=int(parts[4]),
                name=",".join(parts[5:]).strip(),
            )
        )
    if not devices:
        raise RuntimeError("nvidia-smi returned no GPU entries")
    return devices


def _select_best_visible_device(devices: list[CUDAGPUInfo]) -> tuple[CUDAGPUInfo, int]:
    """Select the device with the most free memory and map it to the torch ordinal."""
    visible_tokens = _visible_device_tokens()
    if visible_tokens is None:
        chosen = max(
            devices,
            key=lambda device: (device.memory_free_mib, -device.utilization_gpu, -device.index),
        )
        return chosen, chosen.index

    visible_devices: list[tuple[int, CUDAGPUInfo]] = []
    for ordinal, token in enumerate(visible_tokens):
        for device in devices:
            if token == str(device.index) or token == device.uuid:
                visible_devices.append((ordinal, device))
                break
    if not visible_devices:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is set, but none of the visible devices matched nvidia-smi"
        )
    chosen_ordinal, chosen_device = max(
        visible_devices,
        key=lambda item: (
            item[1].memory_free_mib,
            -item[1].utilization_gpu,
            -item[1].index,
        ),
    )
    return chosen_device, chosen_ordinal


def resolve_cuda_device(requested: str) -> torch.device:
    """Resolve a GPU-only device specification.

    Accepts explicit ordinals like ``cuda:0`` or ``cuda:3`` and the special
    values ``auto`` and ``cuda``. Automatic selection uses nvidia-smi and picks
    the visible GPU with the most free memory.
    """
    spec = requested.strip().lower()
    auto_requested = spec in {"auto", "cuda", "cuda:auto"}
    if not auto_requested and not spec.startswith("cuda:"):
        raise RuntimeError(f"GPU is required; got device {requested!r}")

    if auto_requested:
        try:
            devices = _query_nvidia_smi()
            chosen_device, ordinal = _select_best_visible_device(devices)
            resolved = torch.device(f"cuda:{ordinal}")
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA appears unavailable after selecting {chosen_device.name} "
                    f"(index {chosen_device.index}, {chosen_device.memory_free_mib}/{chosen_device.memory_total_mib} MiB free)"
                )
            if ordinal >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Selected CUDA ordinal {ordinal} is outside the visible device range "
                    f"(count={torch.cuda.device_count()})"
                )
            return resolved
        except Exception:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            raise

    resolved = torch.device(requested)
    if resolved.type != "cuda":
        raise RuntimeError(f"GPU is required; got device {requested!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for {requested!r}")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device {requested!r} but only {torch.cuda.device_count()} GPU(s) are visible"
        )
    return resolved
