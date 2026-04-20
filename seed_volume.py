"""One-shot: seed a RunPod network volume with LTX-2 weights via S3 API.

RunPod exposes network volumes through an S3-compatible endpoint:
  https://s3api-<dc-id>.runpod.io

This script:

  1. huggingface-hub snapshot_download Lightricks/LTX-2 (pipeline-only)
     into ./ltx2-weights/. Records HF -> local MB/s.
  2. boto3 put_object each file into s3://<volume-id>/ltx-2/<rel-path>.
     Records local -> volume MB/s.

Both halves are timed separately so §1 of FINDINGS.md has:
  - HF CDN throughput seen from an ISP/laptop
  - RunPod S3 ingest throughput from that same ISP/laptop

For in-DC timing (which is what a first cold start sees), the runpod/
diag endpoint Dockerfile can be reused with a brief snapshot_download
shim; that number is captured separately by the `coldstart-live`
benchmark. This script is *not* intended to be run from a RunPod pod.

Env needed:
  RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY  (create in the RunPod console -> Settings -> S3 API Keys)
  RUNPOD_NETWORK_VOLUME_ID                    (e.g. 'a1b2c3d4e5')
  RUNPOD_S3_DATACENTER                        (e.g. 'EUR-IS-2'; must match the volume's DC)

Also needed: HF_TOKEN for rate-limited pulls (set by HF_TOKEN= or via
`huggingface-cli login` before running).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

HF_REPO = "Lightricks/LTX-2"
DEFAULT_LOCAL = Path("./ltx2-weights")
DEFAULT_REMOTE_PREFIX = "ltx-2"

PIPELINE_ONLY_PATTERNS = [
    "model_index.json",
    "*.json",
    "LICENSE",
    "README.md",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "transformer/*",
    "vae/*",
    "audio_vae/*",
    "connectors/*",
    "vocoder/*",
]


def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    out = subprocess.check_output(["du", "-sb", str(path)]).decode().split()[0]
    return int(out)


def _fmt_rate(nbytes: int, seconds: float) -> str:
    if seconds <= 0 or nbytes <= 0:
        return "n/a"
    mb_s = (nbytes / 1_000_000) / seconds
    return f"{mb_s:.1f} MB/s ({mb_s * 8:.1f} Mb/s)"


def download_from_hf(
    local_dir: Path, repo: str, allow_patterns: list[str] | None
) -> tuple[int, float]:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[hf] snapshot_download {repo} -> {local_dir} patterns={allow_patterns}", flush=True)
    t0 = time.perf_counter()
    snapshot_download(
        repo_id=repo,
        local_dir=str(local_dir),
        max_workers=16,
        allow_patterns=allow_patterns,
    )
    dt = time.perf_counter() - t0
    nbytes = _du_bytes(local_dir)
    print(f"[hf] {nbytes:,} bytes in {dt:.1f}s -> {_fmt_rate(nbytes, dt)}", flush=True)
    return nbytes, dt


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def upload_to_runpod(
    local_dir: Path,
    volume_id: str,
    dc_id: str,
    remote_prefix: str,
    access_key: str,
    secret_key: str,
) -> tuple[int, float, int]:
    """Upload via boto3 to RunPod's S3-compatible API.

    Returns (bytes, seconds, n_files).
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        print("[rp] ERROR: pip install boto3", file=sys.stderr)
        sys.exit(2)

    endpoint_url = f"https://s3api-{dc_id.lower()}.runpod.io"
    # RunPod's S3 endpoint expects virtual-hosted-style OFF.
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    files = list(_iter_files(local_dir))
    n = len(files)
    total = sum(p.stat().st_size for p in files)
    print(
        f"[rp] uploading {n} files ({total:,} bytes) to "
        f"s3://{volume_id}/{remote_prefix}/ via {endpoint_url}",
        flush=True,
    )

    t0 = time.perf_counter()
    for i, src in enumerate(files, 1):
        rel = src.relative_to(local_dir).as_posix()
        key = f"{remote_prefix.rstrip('/')}/{rel}"
        t_f = time.perf_counter()
        with src.open("rb") as f:
            s3.put_object(Bucket=volume_id, Key=key, Body=f)
        dt = time.perf_counter() - t_f
        sz = src.stat().st_size
        if i <= 5 or i % 25 == 0 or i == n:
            print(
                f"[rp] [{i}/{n}] {rel}: {sz:,} B in {dt:.1f}s "
                f"({_fmt_rate(sz, dt)})",
                flush=True,
            )
    total_dt = time.perf_counter() - t0
    print(f"[rp] total upload: {total:,} B in {total_dt:.1f}s ({_fmt_rate(total, total_dt)})", flush=True)
    return total, total_dt, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--remote-prefix", default=DEFAULT_REMOTE_PREFIX)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--pipeline-only", action="store_true", default=True)
    parser.add_argument(
        "--volume-id", default=os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    )
    parser.add_argument("--dc-id", default=os.environ.get("RUNPOD_S3_DATACENTER"))
    parser.add_argument(
        "--access-key", default=os.environ.get("RUNPOD_S3_ACCESS_KEY")
    )
    parser.add_argument(
        "--secret-key", default=os.environ.get("RUNPOD_S3_SECRET_KEY")
    )
    args = parser.parse_args()

    allow_patterns = PIPELINE_ONLY_PATTERNS if args.pipeline_only else None

    hf_bytes, hf_dt = (0, 0.0)
    up_bytes, up_dt, up_n = (0, 0.0, 0)

    if not args.skip_download:
        hf_bytes, hf_dt = download_from_hf(args.local, args.repo, allow_patterns)
    else:
        hf_bytes = _du_bytes(args.local)

    if not args.skip_upload:
        missing = [
            k for k, v in (
                ("--volume-id", args.volume_id),
                ("--dc-id", args.dc_id),
                ("--access-key", args.access_key),
                ("--secret-key", args.secret_key),
            ) if not v
        ]
        if missing:
            print(f"[rp] ERROR: need {missing}", file=sys.stderr)
            sys.exit(2)
        up_bytes, up_dt, up_n = upload_to_runpod(
            args.local,
            args.volume_id,
            args.dc_id,
            args.remote_prefix,
            args.access_key,
            args.secret_key,
        )

    print("\n=== seed_volume summary ===")
    print(f"local:   {args.local}  ({hf_bytes:,} bytes)")
    print(f"remote:  s3://{args.volume_id}/{args.remote_prefix}")
    if hf_dt:
        print(f"HF download:  {hf_dt:.1f}s  ({_fmt_rate(hf_bytes, hf_dt)})")
    if up_dt:
        print(f"S3 upload:    {up_dt:.1f}s  ({_fmt_rate(up_bytes, up_dt)})  n={up_n}")


if __name__ == "__main__":
    main()
