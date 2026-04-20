"""RunPod serverless handler for the LTX-2 benchmark workload.

Queue-based endpoint (the only type the RunPod REST API supports at
creation time — Load Balancing is a Web-Console-only toggle). Client
side calls:

    POST https://api.runpod.ai/v2/<endpoint_id>/run
    POST https://api.runpod.ai/v2/<endpoint_id>/runsync
    GET  https://api.runpod.ai/v2/<endpoint_id>/status/<job_id>

with body `{"input": {"op": "info" | "warmup" | "generate", ...}}`.

Cold-start instrumentation:
- `mark(event)` emits JSON lines to stdout. RunPod captures them in the
  worker logs (viewable in the console or via its LOGS API).
- Each job response includes `delayTime` (ms from submission to first
  handler run) and `executionTime` (ms inside `handler`). `delayTime`
  on a scale-0 → first-request call is the cold-start wall as seen by
  the provider; the benchmark client records it explicitly.
- Setup runs in a background thread so the handler can accept a
  follow-up job while weights are loading -- but in practice RunPod
  routes the first job into `handler` only once the handler is
  ready-to-be-called, so we also expose `op=warmup` that simply
  returns the current setup-timing list, used for measurement.

Four storage modes, picked by the `MODEL_SOURCE` and `MODEL_PATH` env
vars baked into the endpoint config:

  MODEL_SOURCE=bake   MODEL_PATH=/opt/models/ltx-2                  (bake variant)
  MODEL_SOURCE=live   MODEL_PATH=Lightricks/LTX-2                   (live HF variant)
  MODEL_SOURCE=cache  MODEL_PATH=auto (resolves under /runpod-volume/huggingface-cache/hub)
  MODEL_SOURCE=volume MODEL_PATH=/runpod-volume/ltx-2               (self-seeded volume)
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

MILESTONES: list[dict[str, Any]] = []
_STATE_LOCK = threading.Lock()


def mark(event: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "event": event,
        "t": time.time(),
        "t_ns": time.time_ns(),
    }
    if extra:
        entry.update(extra)
    with _STATE_LOCK:
        MILESTONES.append(entry)
    print(f"[TIMING] {json.dumps(entry, default=str)}", flush=True)
    return entry


mark("module_import")


class WorkerState:
    def __init__(self) -> None:
        self.ready: bool = False
        self.load_error: str | None = None
        self.pipe: Any = None
        self.torch: Any = None
        self.vram_total: int = 0
        self.gpu_name: str = "unknown"
        self.model_source: str = os.environ.get("MODEL_SOURCE", "live")
        self.model_path: str = os.environ.get("MODEL_PATH", "Lightricks/LTX-2")
        self.bytes_on_disk: int = 0
        self.setup_timings: list[dict[str, Any]] = []

    def resolve_model_path(self) -> str:
        src = self.model_source
        path = self.model_path
        if src == "cache":
            repo = (
                os.environ.get("HF_REPO", "Lightricks/LTX-2")
                if path in ("", "auto")
                else path
            )
            org, name = repo.split("/", 1)
            root = Path("/runpod-volume/huggingface-cache/hub") / f"models--{org}--{name}"
            refs_main = root / "refs" / "main"
            snapshots_dir = root / "snapshots"
            if refs_main.is_file():
                snapshot = refs_main.read_text().strip()
                candidate = snapshots_dir / snapshot
                if candidate.is_dir():
                    return str(candidate)
            if snapshots_dir.is_dir():
                versions = sorted(
                    d.name for d in snapshots_dir.iterdir() if d.is_dir()
                )
                if versions:
                    return str(snapshots_dir / versions[0])
            raise RuntimeError(
                f"MODEL_SOURCE=cache but no snapshot under {snapshots_dir}; "
                f"endpoint's Model setting should be '{repo}'."
            )
        return path


STATE = WorkerState()


def _du_bytes(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    try:
        out = subprocess.check_output(["du", "-sb", path], stderr=subprocess.DEVNULL)
        return int(out.decode().split()[0])
    except Exception:
        return 0


def _setup() -> None:
    mark("setup_enter", {"model_source": STATE.model_source, "model_path_env": STATE.model_path})
    try:
        import torch
        from diffusers.pipelines.ltx2 import LTX2Pipeline

        mark("after_torch_import")

        resolved = STATE.resolve_model_path()
        if resolved != STATE.model_path:
            mark("resolved_model_path", {"resolved": resolved})
            STATE.model_path = resolved

        if STATE.model_source in ("bake", "volume", "cache"):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        mark("before_from_pretrained", {"model_path": STATE.model_path})
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if STATE.model_source in ("bake", "volume", "cache"):
            kwargs["local_files_only"] = True
        pipe = LTX2Pipeline.from_pretrained(STATE.model_path, **kwargs)
        STATE.bytes_on_disk = _du_bytes(STATE.model_path)
        mark("after_from_pretrained", {"bytes_on_disk": STATE.bytes_on_disk})

        pipe.to("cuda")
        mark(
            "after_to_cuda",
            {
                "vram_alloc_bytes": int(torch.cuda.memory_allocated()),
                "vram_reserved_bytes": int(torch.cuda.memory_reserved()),
            },
        )

        try:
            pipe.vae.enable_tiling()
            mark("vae_tiling_enabled")
        except Exception as e:  # noqa: BLE001
            mark("vae_tiling_skipped", {"error": repr(e)})

        STATE.pipe = pipe
        STATE.torch = torch
        STATE.vram_total = int(torch.cuda.get_device_properties(0).total_memory)
        STATE.gpu_name = torch.cuda.get_device_name(0)

        mark("setup_ready")
        STATE.setup_timings = list(MILESTONES)
        STATE.ready = True
    except Exception as e:  # noqa: BLE001
        STATE.load_error = f"{type(e).__name__}: {e}"
        mark("setup_failed", {"error": STATE.load_error})
        raise


_setup_thread = threading.Thread(target=_setup, name="ltx2-setup", daemon=True)
_setup_thread.start()


# --------------------------------------------------------------------------- #
# Handler ops                                                                 #
# --------------------------------------------------------------------------- #


def _ensure_ready(timeout: float = 1800.0) -> None:
    if STATE.ready:
        return
    _setup_thread.join(timeout=timeout)
    if not STATE.ready:
        raise RuntimeError(
            f"worker not ready after {timeout}s; setup error: {STATE.load_error}"
        )


def _op_info(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "setup_timings": STATE.setup_timings or list(MILESTONES),
        "gpu_name": STATE.gpu_name,
        "vram_total_bytes": STATE.vram_total,
        "model_source": STATE.model_source,
        "model_path": STATE.model_path,
        "bytes_on_disk": STATE.bytes_on_disk,
        "server_t": time.time(),
        "ready": STATE.ready,
        "load_error": STATE.load_error,
    }


def _op_warmup(_: dict[str, Any]) -> dict[str, Any]:
    entry = mark("warmup_called", {"ready": STATE.ready})
    _ensure_ready()
    return {
        "ok": True,
        "server_t": entry["t"],
        "ready": STATE.ready,
        "milestones": STATE.setup_timings or list(MILESTONES),
    }


def _op_generate(job_input: dict[str, Any]) -> dict[str, Any]:
    _ensure_ready()
    torch = STATE.torch
    pipe = STATE.pipe

    prompt = job_input.get(
        "prompt", "A cat sitting on a windowsill at sunset, cinematic lighting"
    )
    negative_prompt = job_input.get(
        "negative_prompt",
        "shaky, glitchy, low quality, deformed, distorted, motion smear, fused fingers",
    )
    width = int(job_input.get("width", 1920))
    height = int(job_input.get("height", 1088))
    num_frames = int(job_input.get("num_frames", 121))
    num_inference_steps = int(job_input.get("num_inference_steps", 8))
    guidance_scale = float(job_input.get("guidance_scale", 1.0))
    seed = int(job_input.get("seed", 42))
    frame_rate = float(job_input.get("frame_rate", 24.0))
    upload_result = bool(job_input.get("upload_result", False))
    enable_vae_tiling = bool(job_input.get("enable_vae_tiling", True))
    enable_model_cpu_offload = bool(job_input.get("enable_model_cpu_offload", False))
    enable_sequential_cpu_offload = bool(
        job_input.get("enable_sequential_cpu_offload", False)
    )

    mark(
        "generate_enter",
        {
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "steps": num_inference_steps,
            "upload_result": upload_result,
        },
    )

    if enable_model_cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            mark("model_cpu_offload_enabled")
        except Exception as e:  # noqa: BLE001
            mark("model_cpu_offload_skipped", {"error": repr(e)})
    if enable_sequential_cpu_offload:
        try:
            pipe.enable_sequential_cpu_offload()
            mark("sequential_cpu_offload_enabled")
        except Exception as e:  # noqa: BLE001
            mark("sequential_cpu_offload_skipped", {"error": repr(e)})

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
        return_dict=False,
    )
    dt = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated())
    mark("generate_done", {"wall_s": dt, "vram_peak_bytes": peak})

    video_b64: str | None = None
    video_size: int | None = None
    if upload_result:
        video_b64, video_size = _encode_to_base64(result, width, height, frame_rate)
        mark("encode_done", {"video_size_bytes": video_size})

    return {
        "wall_s": dt,
        "s_per_frame": dt / max(num_frames, 1),
        "ms_per_step": 1000.0 * dt / max(num_inference_steps, 1),
        "vram_peak_bytes": peak,
        "vram_total_bytes": STATE.vram_total,
        "shape": [num_frames, height, width, 3],
        "server_t": time.time(),
        "video_base64": video_b64,
        "video_size_bytes": video_size,
        "milestones": MILESTONES[-20:],
    }


def _encode_to_base64(
    pipe_result: Any, width: int, height: int, frame_rate: float
) -> tuple[str | None, int | None]:
    try:
        import tempfile

        import imageio_ffmpeg
        import numpy as np

        video = pipe_result[0] if isinstance(pipe_result, tuple) else pipe_result
        frames = np.asarray(video)
        while frames.ndim > 4:
            frames = frames[0]
        if frames.dtype != np.uint8:
            frames = (frames.clip(0.0, 1.0) * 255).astype(np.uint8)
        frames = np.ascontiguousarray(frames)

        h, w = frames.shape[1], frames.shape[2]
        tmp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = imageio_ffmpeg.write_frames(
            tmp_path,
            size=(w, h),
            fps=frame_rate,
            codec="libx264",
            quality=7,
            macro_block_size=1,
        )
        writer.send(None)
        for frame in frames:
            writer.send(frame)
        writer.close()
        with open(tmp_path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("ascii"), len(raw)
    except Exception as e:  # noqa: BLE001
        mark("encode_failed", {"error": repr(e)})
        return None, None


# --------------------------------------------------------------------------- #
# RunPod handler                                                              #
# --------------------------------------------------------------------------- #


_OPS = {
    "info": _op_info,
    "warmup": _op_warmup,
    "generate": _op_generate,
}


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_input = (event or {}).get("input") or {}
    op = job_input.get("op", "generate")
    fn = _OPS.get(op)
    if fn is None:
        return {"error": f"unknown op {op!r}; expected one of {list(_OPS)}"}
    try:
        return fn(job_input)
    except Exception as e:  # noqa: BLE001
        mark("handler_error", {"op": op, "error": repr(e)})
        return {"error": f"{type(e).__name__}: {e}", "op": op}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
