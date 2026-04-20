"""Client-side benchmark harness for the RunPod LTX-2 exploration
(queue-based endpoints).

Port of exploration/fal/benchmark.py. Same scenarios + result schema,
different provider primitives:

  deploy          git commit + push -> RunPod Hub rebuild -> worker
                  healthy. We poll /status on a dummy /run job and
                  treat delayTime -> 0 as "rebuild complete". First
                  deploy is also recorded separately (caches cold).
  run             No true ephemeral-GPU equivalent on RunPod
                  serverless; `runpod project start` is a different
                  compute pool. We still record its wall for §6.
  iteration       3x2 edit-kind x command matrix.
  coldstart       Scale endpoint to 0 via REST, wait for workers to
                  drain, then POST /runsync with op=warmup. Records
                  the `delayTime` (provider-reported queue wait ==
                  cold-start wall) and the server-side setup_timings.
  compute         /runsync op=generate at 1920x1088x121, optional
                  fallback to 1280x720.
  invocation      /runsync op=generate with upload_result=True. Mp4
                  returned as base64 in the response body; we measure
                  encoded size + round-trip latency as proxies for a
                  signed-URL delivery time.
  all             coldstart + compute + invocation.

Required env:
  RUNPOD_API_KEY                 (auth for both REST and /run)
  RUNPOD_ENDPOINT_ID             (for /run URL and for REST scale-to-zero)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

DEFAULT_PROMPT = "A sweeping aerial shot of a coastal cliff at golden hour"

REST_BASE = "https://rest.runpod.io/v1"
RUN_BASE = "https://api.runpod.ai/v2"


# --------------------------------------------------------------------------- #
# HTTP                                                                        #
# --------------------------------------------------------------------------- #


def http_request(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 900.0,
) -> tuple[int, dict[str, Any] | str, float, float]:
    hdr = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdr.update(headers)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    t_before = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    t_after = time.time()
    try:
        payload: dict[str, Any] | str = json.loads(raw.decode() or "{}")
    except Exception:
        payload = (raw or b"").decode(errors="replace")
    return status, payload, t_before, t_after


def _rp_headers() -> dict[str, str]:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY unset")
    return {"Authorization": f"Bearer {key}"}


def rp_runsync(endpoint_id: str, body: dict[str, Any], timeout: float = 1800.0) -> tuple[dict[str, Any], float, float]:
    status, payload, tb, ta = http_request(
        "POST",
        f"{RUN_BASE}/{endpoint_id}/runsync",
        body=body,
        headers=_rp_headers(),
        timeout=timeout,
    )
    if status >= 400:
        raise RuntimeError(f"/runsync -> {status}: {payload}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"/runsync non-JSON: {payload!r}")
    return payload, tb, ta


def rp_run(endpoint_id: str, body: dict[str, Any]) -> dict[str, Any]:
    status, payload, _, _ = http_request(
        "POST", f"{RUN_BASE}/{endpoint_id}/run", body=body, headers=_rp_headers(), timeout=60.0
    )
    if status >= 400 or not isinstance(payload, dict):
        raise RuntimeError(f"/run -> {status}: {payload}")
    return payload


def rp_status(endpoint_id: str, job_id: str) -> dict[str, Any]:
    status, payload, _, _ = http_request(
        "GET", f"{RUN_BASE}/{endpoint_id}/status/{job_id}", headers=_rp_headers(), timeout=30.0
    )
    if status >= 400 or not isinstance(payload, dict):
        raise RuntimeError(f"/status -> {status}: {payload}")
    return payload


def rp_wait_job(endpoint_id: str, job_id: str, timeout_s: float = 1800.0, poll_s: float = 2.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = rp_status(endpoint_id, job_id)
        s = state.get("status")
        if s in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return state
        time.sleep(poll_s)
    raise TimeoutError(f"job {job_id} did not finish in {timeout_s}s")


def rp_rest_get(path: str) -> tuple[int, Any]:
    s, p, _, _ = http_request("GET", f"{REST_BASE}{path}", headers=_rp_headers(), timeout=30.0)
    return s, p


def rp_rest_patch(path: str, body: dict[str, Any]) -> tuple[int, Any]:
    s, p, _, _ = http_request("PATCH", f"{REST_BASE}{path}", body=body, headers=_rp_headers(), timeout=60.0)
    return s, p


def rp_endpoint(endpoint_id: str) -> dict[str, Any]:
    s, p = rp_rest_get(f"/endpoints/{endpoint_id}")
    if s >= 400 or not isinstance(p, dict):
        raise RuntimeError(f"GET /endpoints/{endpoint_id} -> {s}: {p}")
    return p


def rp_workers(endpoint_id: str) -> list[dict[str, Any]]:
    ep = rp_endpoint(endpoint_id)
    w = ep.get("workers") or ep.get("workerList") or []
    return w if isinstance(w, list) else []


def rp_set_workers(endpoint_id: str, workers_min: int, workers_max: int) -> None:
    s, p = rp_rest_patch(
        f"/endpoints/{endpoint_id}/update",
        {"workersMin": workers_min, "workersMax": workers_max},
    )
    print(f"[rp] PATCH workers min={workers_min} max={workers_max} -> {s}", flush=True)
    if s >= 400:
        raise RuntimeError(f"PATCH endpoint: {s}: {p}")


def rp_wait_drain(endpoint_id: str, timeout_s: float = 300.0, poll_s: float = 5.0) -> float:
    t0 = time.time()
    while True:
        try:
            workers = rp_workers(endpoint_id)
        except Exception as e:
            print(f"[rp] worker poll failed: {e!r}", flush=True)
            workers = []
        live = [
            w for w in workers
            if str(w.get("status") or w.get("state") or "").upper()
            not in ("TERMINATED", "EXITED", "TERMINATING", "")
        ]
        if not live:
            return time.time() - t0
        if time.time() - t0 > timeout_s:
            print(f"[rp] WARN: workers did not drain after {timeout_s}s: {live}", flush=True)
            return time.time() - t0
        time.sleep(poll_s)


def _maybe_scale_to_zero(endpoint_id: str | None) -> None:
    if not endpoint_id:
        return
    try:
        rp_set_workers(endpoint_id, 0, 0)
        drain = rp_wait_drain(endpoint_id)
        print(f"[rp] workers drained in {drain:.1f}s", flush=True)
        rp_set_workers(endpoint_id, 0, 1)
    except Exception as e:
        print(f"[rp] scale-to-zero failed: {e!r}", flush=True)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class CliRun:
    cmd: list[str]
    t_start: float
    t_end: float
    returncode: int
    stdout: str
    stderr: str

    @property
    def wall_s(self) -> float:
        return self.t_end - self.t_start


def run_cli(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> CliRun:
    full_env = {**os.environ, **(env or {})}
    print(f"[cli] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd or HERE, env=full_env, capture_output=True, text=True)
    t1 = time.time()
    print(f"[cli] rc={proc.returncode} wall={t1 - t0:.1f}s", flush=True)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
    return CliRun(cmd=cmd, t_start=t0, t_end=t1, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def estimate_skew(endpoint_id: str, samples: int = 3) -> float:
    deltas: list[float] = []
    for _ in range(samples):
        try:
            resp, tb, ta = rp_runsync(endpoint_id, {"input": {"op": "warmup"}}, timeout=60.0)
            out = resp.get("output") or {}
            server_t = float(out.get("server_t") or 0)
            if server_t:
                deltas.append(server_t - (tb + ta) / 2.0)
        except Exception as e:
            print(f"[skew] {e!r}", file=sys.stderr)
        time.sleep(0.5)
    return statistics.median(deltas) if deltas else 0.0


def save_result(scenario: str, payload: dict[str, Any]) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = RESULTS / f"{scenario}-{ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[result] wrote {path}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# Deploy / run                                                                #
# --------------------------------------------------------------------------- #


def cmd_deploy(args: argparse.Namespace) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for i in range(args.trials):
        push = run_cli(["git", "push", "origin", args.branch], cwd=args.repo_path or HERE)
        t0 = push.t_start
        trials.append({"trial": i, "push_wall_s": push.wall_s, "t_push_start": t0})
    payload = {"scenario": "deploy", "branch": args.branch, "trials": trials}
    save_result(f"deploy-{args.tag}" if args.tag else "deploy", payload)
    return payload


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    result = run_cli(["runpod", "project", "start"])
    payload = {"scenario": "run", "tag": args.tag, "cli": asdict(result), "wall_s": result.wall_s}
    save_result(f"run-{args.tag}" if args.tag else "run", payload)
    return payload


# --------------------------------------------------------------------------- #
# Iteration                                                                   #
# --------------------------------------------------------------------------- #

PYTHON_MARKER = "# BENCHMARK_BUMP"
REQUIREMENTS_MARKER = "# benchmark_bump"


@dataclass
class Edit:
    name: str
    apply: Any
    revert: Any


def _bump_python_constant() -> Edit:
    path = HERE / "app.py"
    original = path.read_text()

    def apply() -> None:
        path.write_text(original + f"\n{PYTHON_MARKER} {time.time_ns()}\n")

    def revert() -> None:
        path.write_text(original)

    return Edit(name="python_only", apply=apply, revert=revert)


def _bump_requirement() -> Edit:
    path = HERE / "requirements.txt"
    original = path.read_text()

    def apply() -> None:
        path.write_text(original + f"{REQUIREMENTS_MARKER}\nrequests==2.32.3\n")

    def revert() -> None:
        path.write_text(original)

    return Edit(name="pip_dep", apply=apply, revert=revert)


def _bump_dockerfile() -> Edit:
    path = HERE / "Dockerfile.live"
    original = path.read_text()

    def apply() -> None:
        path.write_text(
            original + f'\nRUN echo "benchmark_bump {time.time_ns()}" > /tmp/bench_marker\n'
        )

    def revert() -> None:
        path.write_text(original)

    return Edit(name="dockerfile_layer", apply=apply, revert=revert)


def cmd_iteration(args: argparse.Namespace) -> dict[str, Any]:
    edits = {
        "python_only": _bump_python_constant,
        "pip_dep": _bump_requirement,
        "dockerfile_layer": _bump_dockerfile,
    }
    commands = {"git_push": None, "runpod_dev": ["runpod", "project", "dev"]}

    results: dict[str, dict[str, list[float]]] = {}
    first_trials: dict[str, dict[str, float]] = {}

    for edit_name, edit_factory in edits.items():
        results[edit_name] = {}
        first_trials[edit_name] = {}
        for cmd_name, cmd in commands.items():
            if args.only and args.only not in (edit_name, cmd_name):
                continue
            wall_times: list[float] = []
            for trial in range(args.trials):
                edit = edit_factory()
                edit.apply()
                try:
                    if cmd_name == "git_push":
                        commit = run_cli(
                            ["git", "commit", "-am", f"bench: {edit_name} bump", "--allow-empty"]
                        )
                        push = run_cli(["git", "push", "origin", args.branch])
                        wall_times.append(push.t_end - commit.t_start)
                    else:
                        run = run_cli(cmd)
                        wall_times.append(run.wall_s)
                finally:
                    edit.revert()
                time.sleep(2.0)
            if wall_times:
                first_trials[edit_name][cmd_name] = wall_times[0]
                rest = wall_times[1:] or wall_times
                results[edit_name][cmd_name] = rest
                median = statistics.median(rest)
                print(
                    f"[iter] {edit_name}/{cmd_name}: first={wall_times[0]:.1f}s steady={rest} median={median:.1f}s",
                    flush=True,
                )

    summary = {
        edit: {
            cmd: {
                "first_trial_s": first_trials.get(edit, {}).get(cmd),
                "steady_samples": t,
                "steady_median_s": statistics.median(t) if t else None,
            }
            for cmd, t in m.items()
        }
        for edit, m in results.items()
    }
    payload = {"scenario": "iteration", "trials": args.trials, "matrix": summary}
    save_result("iteration", payload)
    return payload


# --------------------------------------------------------------------------- #
# Coldstart / compute / invocation                                            #
# --------------------------------------------------------------------------- #


def _require_eid(args: argparse.Namespace) -> str:
    eid = args.endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
    if not eid:
        print("error: --endpoint-id or $RUNPOD_ENDPOINT_ID required", file=sys.stderr)
        sys.exit(2)
    return eid


def _process_runsync_resp(resp: dict[str, Any]) -> dict[str, Any]:
    """Extract delayTime + output + server milestones from a /runsync response."""
    out = resp.get("output") or {}
    milestones = out.get("milestones") or out.get("setup_timings") or []
    by_event = {m["event"]: m for m in milestones if isinstance(m, dict) and "event" in m}

    def at(event: str) -> float | None:
        m = by_event.get(event)
        return float(m["t"]) if m else None

    return {
        "job_id": resp.get("id"),
        "delay_time_ms": resp.get("delayTime"),
        "execution_time_ms": resp.get("executionTime"),
        "worker_id": resp.get("workerId") or resp.get("worker_id"),
        "status": resp.get("status"),
        "output_keys": sorted(out.keys()) if isinstance(out, dict) else None,
        "setup_enter_t": at("setup_enter"),
        "setup_ready_t": at("setup_ready"),
        "before_from_pretrained_t": at("before_from_pretrained"),
        "after_from_pretrained_t": at("after_from_pretrained"),
        "after_to_cuda_t": at("after_to_cuda"),
        "milestones_count": len(milestones),
    }


def cmd_coldstart(args: argparse.Namespace) -> dict[str, Any]:
    eid = _require_eid(args)

    skew_s = 0.0
    if args.estimate_skew:
        skew_s = estimate_skew(eid, samples=args.skew_samples)
        print(f"[skew] estimated offset = {skew_s:+.3f}s", flush=True)

    _maybe_scale_to_zero(eid)

    trials: list[dict[str, Any]] = []
    for i in range(args.trials):
        if i > 0 and args.sleep_between:
            print(f"[cold] sleep {args.sleep_between}s then re-drain", flush=True)
            time.sleep(args.sleep_between)
            _maybe_scale_to_zero(eid)

        t_before = time.time()
        try:
            resp, tb, ta = rp_runsync(eid, {"input": {"op": "warmup"}}, timeout=1800.0)
        except Exception as e:
            trials.append({"trial": i, "error": repr(e), "t_before": t_before})
            continue

        info = _process_runsync_resp(resp)
        out = resp.get("output") or {}
        milestones = out.get("milestones") or out.get("setup_timings") or []
        by_event = {m["event"]: m for m in milestones if isinstance(m, dict) and "event" in m}

        def at(event: str) -> float | None:
            m = by_event.get(event)
            return float(m["t"]) if m else None

        setup_enter = at("setup_enter")
        setup_ready = at("setup_ready")
        fp_in = at("before_from_pretrained")
        fp_out = at("after_from_pretrained")
        to_cuda = at("after_to_cuda")
        bytes_on_disk = 0
        m = by_event.get("after_from_pretrained")
        if m and "bytes_on_disk" in m:
            bytes_on_disk = int(m["bytes_on_disk"])

        trials.append(
            {
                "trial": i,
                "t_http_start": tb,
                "t_http_end": ta,
                "skew_s": skew_s,
                "cold_total_s": ta - tb,
                "delay_time_ms": info["delay_time_ms"],
                "execution_time_ms": info["execution_time_ms"],
                "worker_id": info["worker_id"],
                "setup_enter_to_ready_s": (setup_ready - setup_enter)
                if (setup_enter and setup_ready) else None,
                "from_pretrained_s": (fp_out - fp_in) if (fp_in and fp_out) else None,
                "to_cuda_s": (to_cuda - fp_out) if (fp_out and to_cuda) else None,
                "bytes_on_disk": bytes_on_disk,
                "load_mbps": ((bytes_on_disk / 1_000_000) / (fp_out - fp_in))
                if (fp_in and fp_out and bytes_on_disk) else None,
                "milestones": milestones,
            }
        )
        t = trials[-1]
        print(
            f"[cold] trial {i}: delay={t['delay_time_ms']}ms exec={t['execution_time_ms']}ms "
            f"cold_total={t['cold_total_s']:.1f}s load_mbps={t['load_mbps']}",
            flush=True,
        )

    payload = {
        "scenario": "coldstart",
        "endpoint_id": eid,
        "model_source_hint": args.model_source_hint,
        "trials": trials,
    }
    save_result(
        f"coldstart-{args.model_source_hint}" if args.model_source_hint else "coldstart", payload
    )
    return payload


def _invoke(eid: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
    t0 = time.time()
    resp, tb, ta = rp_runsync(eid, body, timeout=3600.0)
    return resp, ta - t0


def cmd_compute(args: argparse.Namespace) -> dict[str, Any]:
    eid = _require_eid(args)

    base = {
        "op": "generate",
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "num_frames": args.num_frames,
        "width": args.width,
        "height": args.height,
        "enable_vae_tiling": True,
    }
    primary_label = f"{args.width}x{args.height}"

    trials: list[dict[str, Any]] = []
    fallback_used = False
    last_error: str | None = None

    for i in range(args.trials):
        try:
            resp, wall_s = _invoke(eid, {"input": base})
            out = resp.get("output") or {}
            trials.append(
                {
                    "trial": i,
                    "resolution": primary_label,
                    "wall_s": wall_s,
                    "delay_time_ms": resp.get("delayTime"),
                    "execution_time_ms": resp.get("executionTime"),
                    "s_per_frame": out.get("s_per_frame"),
                    "ms_per_step": out.get("ms_per_step"),
                    "vram_peak_bytes": out.get("vram_peak_bytes"),
                    "vram_total_bytes": out.get("vram_total_bytes"),
                    "output_keys": sorted(out.keys()) if isinstance(out, dict) else None,
                }
            )
            peak_gb = (out.get("vram_peak_bytes") or 0) / 1e9
            print(
                f"[compute] trial {i} {primary_label}: {wall_s:.1f}s "
                f"{out.get('s_per_frame', 0):.2f}s/frame peak={peak_gb:.1f}GB",
                flush=True,
            )
        except Exception as e:
            last_error = repr(e)
            if not args.fallback:
                raise
            fallback_used = True
            fallback = {
                **base,
                "width": 1280,
                "height": 720,
                "enable_vae_tiling": True,
                "enable_model_cpu_offload": True,
            }
            resp, wall_s = _invoke(eid, {"input": fallback})
            trials.append(
                {
                    "trial": i,
                    "resolution": "1280x720",
                    "wall_s": wall_s,
                    "fallback_reason": last_error,
                    "output": resp.get("output"),
                }
            )

    payload = {
        "scenario": "compute",
        "endpoint_id": eid,
        "resolution": primary_label,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "fallback_used": fallback_used,
        "last_error": last_error,
        "trials": trials,
    }
    save_result("compute", payload)
    return payload


def cmd_invocation(args: argparse.Namespace) -> dict[str, Any]:
    eid = _require_eid(args)

    body = {
        "input": {
            "op": "generate",
            "prompt": args.prompt,
            "num_inference_steps": args.steps,
            "num_frames": 25,
            "width": args.width,
            "height": args.height,
            "upload_result": True,
            "enable_vae_tiling": True,
        }
    }

    t_before = time.time()
    resp, tb, ta = rp_runsync(eid, body, timeout=3600.0)
    generate_wall_s = ta - t_before
    out = resp.get("output") or {}

    video_b64 = out.get("video_base64")
    size_bytes = out.get("video_size_bytes")
    encoded_bytes = len(video_b64) if video_b64 else 0

    on_disk_path: str | None = None
    if video_b64 and args.save_mp4:
        on_disk_path = str(RESULTS / f"invocation-{time.strftime('%Y%m%dT%H%M%S')}.mp4")
        with open(on_disk_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

    download_s = ta - tb
    download_mbps = ((encoded_bytes / 1_000_000) / download_s) if download_s > 0 else None

    payload = {
        "scenario": "invocation",
        "endpoint_id": eid,
        "generate_wall_s": generate_wall_s,
        "mp4_size_bytes": size_bytes,
        "encoded_b64_bytes": encoded_bytes,
        "inline_download_s": download_s,
        "inline_download_mbps_upper_bound": download_mbps,
        "saved_to": on_disk_path,
        "delay_time_ms": resp.get("delayTime"),
        "execution_time_ms": resp.get("executionTime"),
        "output_excluding_video": {k: v for k, v in out.items() if k != "video_base64"},
    }
    save_result("invocation", payload)
    return payload


def cmd_all(args: argparse.Namespace) -> dict[str, Any]:
    out = {
        "coldstart": cmd_coldstart(args),
        "compute": cmd_compute(args),
        "invocation": cmd_invocation(args),
    }
    save_result("all", out)
    return out


# --------------------------------------------------------------------------- #
# Argparse                                                                    #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="scenario", required=True)

    q = sub.add_parser("deploy")
    q.add_argument("--tag", default="")
    q.add_argument("--branch", default="main")
    q.add_argument("--repo-path", default=None)
    q.add_argument("--trials", type=int, default=1)

    q = sub.add_parser("run")
    q.add_argument("--tag", default="")

    q = sub.add_parser("iteration")
    q.add_argument("--trials", type=int, default=3)
    q.add_argument("--only", default=None)
    q.add_argument("--branch", default="main")

    for name in ("coldstart", "compute", "invocation", "all"):
        q = sub.add_parser(name)
        q.add_argument("--endpoint-id", default=None)
        q.add_argument("--trials", type=int, default=1)
        q.add_argument("--sleep-between", type=float, default=180.0)
        q.add_argument("--steps", type=int, default=8)
        q.add_argument("--num-frames", type=int, default=49)
        q.add_argument("--width", type=int, default=1280)
        q.add_argument("--height", type=int, default=720)
        q.add_argument("--fallback", action="store_true")
        q.add_argument("--prompt", default=DEFAULT_PROMPT)
        q.add_argument("--model-source-hint", default=None)
        q.add_argument("--estimate-skew", action="store_true")
        q.add_argument("--skew-samples", type=int, default=3)
        q.add_argument("--save-mp4", action="store_true")

    return p


DISPATCH = {
    "deploy": cmd_deploy,
    "run": cmd_run,
    "iteration": cmd_iteration,
    "coldstart": cmd_coldstart,
    "compute": cmd_compute,
    "invocation": cmd_invocation,
    "all": cmd_all,
}


def main() -> None:
    args = build_parser().parse_args()
    DISPATCH[args.scenario](args)


if __name__ == "__main__":
    main()
