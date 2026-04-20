"""RunPod serverless diagnostics handler.

Answers the node-caching questions §10 of FINDINGS.md needs, without
requiring the 90 GB LTX-2 image. Handler ops:

    {"input": {"op": "probe"}}  -> node/host identity + marker r/w
    {"input": {"op": "gpu"}}    -> cuBLAS/HBM/PCIe/SDPA/Conv3D micro-bench

Guaranteed-cold driving: `drive_diagnostics.sh` scales the endpoint to
0 between probes via PATCH /v1/endpoints/<id>/update and polls until
no non-terminated workers remain -- only then fires the next probe.

Marker files in 8 candidate paths. A readable marker after a cold
start proves the path survived container recreation; comparing
hostnames/boot_ids across those probes tells us whether storage is
node-local or region-local.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any


MARKER_PATHS: dict[str, Path | None] = {
    "hf_cache_default": Path(
        os.path.expanduser("~/.cache/huggingface/runpod-diag-marker.json")
    ),
    "hf_home_env": (
        Path(os.environ["HF_HOME"]) / "runpod-diag-marker.json"
        if os.environ.get("HF_HOME")
        else None
    ),
    "runpod_volume": Path("/runpod-volume/runpod-diag-marker.json"),
    "runpod_hf_cache": Path(
        "/runpod-volume/huggingface-cache/runpod-diag-marker.json"
    ),
    "root_home": Path("/root/runpod-diag-marker.json"),
    "tmp": Path("/tmp/runpod-diag-marker.json"),
    "workspace": Path("/workspace/runpod-diag-marker.json"),
    "opt_models": Path("/opt/models/runpod-diag-marker.json"),
}


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001
        return f"ERR: {type(e).__name__}: {e}"


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()[:4000]
    except Exception as e:  # noqa: BLE001
        return f"ERR: {type(e).__name__}: {e}"


def _env_of_interest() -> dict[str, str]:
    keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
        "RUNPOD_POD_ID",
        "RUNPOD_ENDPOINT_ID",
        "RUNPOD_WORKER_ID",
        "RUNPOD_DC_ID",
        "RUNPOD_GPU_COUNT",
        "RUNPOD_CPU_COUNT",
        "RUNPOD_PUBLIC_IP",
        "RUNPOD_TCP_PORT_22",
        "HOSTNAME",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}


def _list_dir(p: Path | None, n: int = 25) -> dict[str, Any]:
    if not p or not p.exists():
        return {"path": str(p), "exists": False}
    try:
        entries = sorted(e.name for e in p.iterdir())
        du = _safe(shutil.disk_usage, str(p))
        out: dict[str, Any] = {
            "path": str(p),
            "exists": True,
            "n_entries": len(entries),
            "sample": entries[:n],
        }
        if hasattr(du, "total"):
            out["disk_total_gb"] = round(du.total / 1e9, 2)
            out["disk_free_gb"] = round(du.free / 1e9, 2)
        return out
    except Exception as e:  # noqa: BLE001
        return {"path": str(p), "exists": True, "error": str(e)}


def _read_marker(p: Path | None) -> dict[str, Any] | None:
    if p is None or not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        data["_marker_age_s"] = round(time.time() - p.stat().st_mtime, 1)
        data["_marker_mtime"] = p.stat().st_mtime
        return data
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}


def _write_marker(p: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if p is None:
        return {"skipped": True, "reason": "path unset"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))
        return {"wrote": str(p), "bytes": p.stat().st_size}
    except Exception as e:  # noqa: BLE001
        return {"path": str(p), "error": f"{type(e).__name__}: {e}"}


def _mount_info() -> dict[str, Any]:
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    interesting: list[str] = []
    needles = (
        "/runpod-volume",
        "/data",
        "huggingface",
        "/.cache",
        "/root",
        "/workspace",
        "/model",
        "/opt/models",
    )
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        mountpoint = parts[4]
        if any(needle in mountpoint for needle in needles):
            interesting.append(line)
    return {"n_total_mounts": len(lines), "interesting": interesting}


def _snapshot(phase: str) -> dict[str, Any]:
    now = time.time()
    uname = _safe(
        lambda: dict(
            zip(
                ("sysname", "nodename", "release", "version", "machine"),
                os.uname(),
            )
        )
    )
    return {
        "phase": phase,
        "t": now,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "pid": os.getpid(),
        "hostname": _safe(socket.gethostname),
        "uname": uname,
        "machine_id": _safe(lambda: Path("/etc/machine-id").read_text().strip()),
        "boot_id": _safe(
            lambda: Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        ),
        "container_uptime_s": _safe(
            lambda: float(Path("/proc/uptime").read_text().split()[0])
        ),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,serial,name,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
        "env": _env_of_interest(),
        "hf_default_cache": _list_dir(
            Path(os.path.expanduser("~/.cache/huggingface/hub"))
        ),
        "hf_home_cache": (
            _list_dir(Path(os.environ["HF_HOME"]) / "hub")
            if os.environ.get("HF_HOME")
            else {"note": "HF_HOME unset"}
        ),
        "runpod_volume": _list_dir(Path("/runpod-volume")),
        "runpod_hf_cache": _list_dir(Path("/runpod-volume/huggingface-cache/hub")),
        "mountinfo": _mount_info(),
        "marker_reads": {name: _read_marker(p) for name, p in MARKER_PATHS.items()},
    }


# --------------------------------------------------------------------------- #
# GPU microbench                                                              #
# --------------------------------------------------------------------------- #


def _cuda_time(fn, iters: int, warmup: int = 2) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ms[len(ms) // 2]


def _bench_matmul(size: int, dtype_name: str, iters: int = 20) -> dict[str, Any]:
    import torch

    dtype = {
        "fp32": torch.float32,
        "tf32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = dtype_name == "tf32"
    try:
        a = torch.randn(size, size, device="cuda", dtype=dtype)
        b = torch.randn(size, size, device="cuda", dtype=dtype)
        out = torch.empty_like(a)

        def step():
            torch.matmul(a, b, out=out)

        ms = _cuda_time(step, iters=iters)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        del a, b, out
        torch.cuda.empty_cache()

    flops = 2.0 * size * size * size
    return {
        "size": size,
        "dtype": dtype_name,
        "ms_median": ms,
        "tflops": flops / (ms / 1000.0) / 1e12,
    }


def _bench_hbm(n_bytes: int = 2 * 1024**3, iters: int = 20) -> dict[str, Any]:
    import torch

    n_fp32 = n_bytes // 4
    a = torch.empty(n_fp32, dtype=torch.float32, device="cuda")
    b = torch.randn(n_fp32, dtype=torch.float32, device="cuda")

    def step():
        a.copy_(b)

    try:
        ms = _cuda_time(step, iters=iters)
    finally:
        del a, b
        torch.cuda.empty_cache()
    bytes_moved = 2 * n_bytes
    return {
        "bytes_per_iter": bytes_moved,
        "ms_median": ms,
        "gbps": bytes_moved / (ms / 1000.0) / 1e9,
    }


def _bench_pcie(n_bytes: int = 512 * 1024**2, iters: int = 5) -> dict[str, Any]:
    import torch

    n_fp32 = n_bytes // 4
    h = torch.empty(n_fp32, dtype=torch.float32, pin_memory=True)
    h.uniform_()
    d = torch.empty(n_fp32, dtype=torch.float32, device="cuda")

    def step_h2d():
        d.copy_(h, non_blocking=True)

    def step_d2h():
        h.copy_(d, non_blocking=True)

    try:
        ms_h2d = _cuda_time(step_h2d, iters=iters, warmup=2)
        ms_d2h = _cuda_time(step_d2h, iters=iters, warmup=2)
    finally:
        del h, d
        torch.cuda.empty_cache()
    return {
        "bytes": n_bytes,
        "h2d_ms_median": ms_h2d,
        "d2h_ms_median": ms_d2h,
        "h2d_gbps": n_bytes / (ms_h2d / 1000.0) / 1e9,
        "d2h_gbps": n_bytes / (ms_d2h / 1000.0) / 1e9,
    }


def _bench_sdpa(
    B: int = 1, H: int = 24, S: int = 4096, D: int = 128, iters: int = 10
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    def step():
        F.scaled_dot_product_attention(q, k, v, is_causal=False)

    try:
        ms = _cuda_time(step, iters=iters)
    finally:
        del q, k, v
        torch.cuda.empty_cache()
    flops = 4.0 * B * H * S * S * D
    return {"shape": [B, H, S, D], "ms_median": ms, "tflops": flops / (ms / 1000.0) / 1e12}


def _bench_conv3d(iters: int = 10) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    C = 256
    T, Hh, Ww = 16, 64, 64
    net = nn.Conv3d(C, C, kernel_size=3, padding=1).cuda().bfloat16()
    x = torch.randn(1, C, T, Hh, Ww, device="cuda", dtype=torch.bfloat16)

    def step():
        net(x)

    try:
        ms = _cuda_time(step, iters=iters)
    finally:
        del net, x
        torch.cuda.empty_cache()
    flops = 2.0 * C * C * 3 * 3 * 3 * T * Hh * Ww
    return {"hidden_shape": [1, C, T, Hh, Ww], "ms_median": ms, "tflops": flops / (ms / 1000.0) / 1e12}


def _gpu_benchmark(request: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch import failed: {type(e).__name__}: {e}"}
    if not torch.cuda.is_available():
        return {"error": "torch.cuda.is_available() is False"}

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    info = {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": props.name,
        "device_uuid": _run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]),
        "sm_count": props.multi_processor_count,
        "sm_version": f"{props.major}.{props.minor}",
        "vram_total_gb": round(total / 1e9, 2),
        "vram_free_gb_at_start": round(free / 1e9, 2),
    }

    matmul_size = int(request.get("matmul_size", 8192))
    results: dict[str, Any] = {
        "matmul": [
            _bench_matmul(matmul_size, "fp32", iters=10),
            _bench_matmul(matmul_size, "tf32", iters=20),
            _bench_matmul(matmul_size, "fp16", iters=30),
            _bench_matmul(matmul_size, "bf16", iters=30),
        ],
        "hbm_memcpy": _bench_hbm(),
        "pcie": _bench_pcie(),
        "sdpa": [
            _bench_sdpa(B=1, H=24, S=2048, D=128),
            _bench_sdpa(B=1, H=24, S=4096, D=128),
            _bench_sdpa(B=1, H=24, S=8192, D=128),
        ],
        "conv3d_bf16": _bench_conv3d(),
    }

    free2, _ = torch.cuda.mem_get_info()
    info["vram_free_gb_at_end"] = round(free2 / 1e9, 2)
    info["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return {"info": info, "results": results, "wall_s": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- #
# Handler state + ops                                                         #
# --------------------------------------------------------------------------- #


class DiagState:
    def __init__(self) -> None:
        self.setup_snapshot: dict[str, Any] = {}
        self.setup_id: str = ""
        self.ready: bool = False


DIAG = DiagState()


def _diag_setup() -> None:
    DIAG.setup_id = uuid.uuid4().hex
    print(f"[DIAG] setup() enter at {time.time()} id={DIAG.setup_id}", flush=True)
    DIAG.setup_snapshot = _snapshot("setup")
    DIAG.setup_snapshot["setup_id"] = DIAG.setup_id
    print(
        f"[DIAG] setup() done id={DIAG.setup_id} "
        f"hostname={DIAG.setup_snapshot.get('hostname')!r} "
        f"boot_id={DIAG.setup_snapshot.get('boot_id')!r}",
        flush=True,
    )
    DIAG.ready = True


_diag_thread = threading.Thread(target=_diag_setup, name="diag-setup", daemon=True)
_diag_thread.start()


def _op_probe(_: dict[str, Any]) -> dict[str, Any]:
    _diag_thread.join(timeout=60)
    snap = _snapshot("probe")
    payload = {
        "written_iso": snap["iso"],
        "written_t": snap["t"],
        "written_hostname": snap["hostname"],
        "written_boot_id": snap["boot_id"],
        "written_machine_id": snap["machine_id"],
        "written_pid": snap["pid"],
        "written_runpod_worker_id": os.environ.get("RUNPOD_WORKER_ID"),
        "written_runpod_endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID"),
    }
    writes = {name: _write_marker(p, payload) for name, p in MARKER_PATHS.items()}
    return {
        "setup_snapshot": DIAG.setup_snapshot,
        "probe_snapshot": snap,
        "marker_writes": writes,
    }


def _op_gpu(job_input: dict[str, Any]) -> dict[str, Any]:
    return _gpu_benchmark(job_input)


def _op_info(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "diagnostics",
        "server_t": time.time(),
        "ready": DIAG.ready,
        "setup_id": DIAG.setup_id,
        "pid": os.getpid(),
        "hostname": _safe(socket.gethostname),
    }


_OPS = {
    "info": _op_info,
    "probe": _op_probe,
    "gpu": _op_gpu,
}


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_input = (event or {}).get("input") or {}
    op = job_input.get("op", "probe")
    fn = _OPS.get(op)
    if fn is None:
        return {"error": f"unknown op {op!r}; expected one of {list(_OPS)}"}
    try:
        return fn(job_input)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "op": op}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    if args.local:
        print(json.dumps(_snapshot("local"), indent=2, default=str))
    else:
        import runpod
        runpod.serverless.start({"handler": handler})
