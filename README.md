# RunPod serverless LTX-2 evaluation harness

This repo is the **worker side** of the RunPod leg of the
`sma-smmls-demo/exploration/GUIDE.md` provider evaluation. RunPod
pulls this repo, builds the Docker images in its own datacenter (no
local network constraints), and runs the resulting workers as
Serverless queue endpoints. The benchmark harness in
`benchmark.py` drives those endpoints from a laptop.

## Workspace layout

| File | What it is |
| ---- | ---------- |
| `app.py` | RunPod handler for the LTX-2 workload. Exposes `op=info \| warmup \| generate`. Picks the storage variant at startup via `MODEL_SOURCE` / `MODEL_PATH` env. |
| `diagnostics.py` | RunPod handler for Phase 1/2 probing. Exposes `op=probe \| gpu`. Stamps marker files to 8 candidate paths + runs a cuBLAS/HBM/PCIe/SDPA/Conv3D microbench. |
| `Dockerfile.diag` | Tiny cu124 + torch + runpod-sdk image for diagnostics. Base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. |
| `Dockerfile.live` | LTX-2 image without weights. Used for the `live`, `cache`, `volume` variants; the variant is picked at runtime from endpoint env. |
| `Dockerfile.bake` | LTX-2 image with the `--pipeline-only` weight set baked to `/opt/models/ltx-2`. Image is ~45 GB (under RunPod Hub's 80 GB cap). |
| `benchmark.py` | Client-side harness. Covers `deploy \| run \| iteration \| coldstart \| compute \| invocation \| all`. Uses RunPod's REST API for scale-to-zero + worker polling, and `POST /v2/<eid>/runsync` for job submission. |
| `drive_diagnostics.sh` | Guaranteed cold-start driver: scales the diag endpoint to 0 → waits for workers to drain → scales to 1 → fires one `op=probe` (or `op=gpu`) via `/runsync`. Repeat N times to collect hostname/boot_id/GPU UUID across trials. |
| `seed_volume.py` | Seeds the 200 GB Network Volume with `Lightricks/LTX-2` (pipeline-only) weights via the volume's S3-compatible API. Prints HF→local and local→volume throughput separately. |
| `FINDINGS.md` | Rubric-shaped write-up. TBM placeholders until data lands. |
| `results/` | `benchmark.py` and `drive_diagnostics.sh` write `*.json` artefacts here. |

## How this maps to `fal` concepts

| fal concept | RunPod equivalent |
| ----------- | ----------------- |
| `fal.App` | Handler class + `runpod.serverless.start({"handler": handler})` |
| `fal.endpoint("/generate")` | `op=generate` dispatch inside the handler |
| `setup()` background task | Python `threading.Thread` that loads the pipeline; handler `join()`s before first real job |
| `ContainerImage.from_dockerfile` | **RunPod Hub GitHub integration** — RunPod builds the Dockerfile from this repo in their own datacenter |
| `fal files upload` (signed URL) | RunPod Network Volume + S3-compatible API for seeding; inline base64 mp4 for result delivery (LB supports signed URLs but queue does not) |
| `fal apps runners` polling | `GET https://rest.runpod.io/v1/endpoints/<id>` for worker state, plus `delayTime`/`executionTime` on job responses |
| `fal run` (ephemeral GPU job) | No direct equivalent. `runpod project start` is the nearest; recorded for §6 only. |

## The four LTX-2 storage variants

1. **bake** — `Dockerfile.bake` copies the pipeline-only weight set into `/opt/models/ltx-2` at build time. Cold start = image pull + torch import + `to(cuda)`, with zero HF/volume I/O.
2. **live** — `Dockerfile.live` + `MODEL_SOURCE=live MODEL_PATH=Lightricks/LTX-2`. Weights fetched from HF on every cold start; worst-case baseline.
3. **cache** — `Dockerfile.live` + `MODEL_SOURCE=cache` + RunPod Endpoint's "Cached Model" config set to `Lightricks/LTX-2`. Worker gets scheduled on a host that has the HF cache pre-staged at `/runpod-volume/huggingface-cache/hub/`. §10 of `FINDINGS.md` dissects whether this actually works.
4. **volume** — `Dockerfile.live` + `MODEL_SOURCE=volume MODEL_PATH=/runpod-volume/ltx-2` + a 200 GB Network Volume attached at `/runpod-volume/`. `seed_volume.py` populates the volume out-of-band.

## One-time setup (manual, in the RunPod console)

Two steps in the console are unavoidable; the REST API does not
expose GitHub integration or the "Cached Model" field.

1. **Connect GitHub**
   Console → user settings → Connections → GitHub → Connect → grant
   `Runpod Inc.` access to `patrickhulce/serverless-gpu-exploration`
   (or "All repositories").

2. **Create a GitHub release**
   RunPod indexes releases, not commits. After pushing scaffold,
   create release `v0.1.0` on `main`.

3. **Create 5 endpoints**, all Queue type, all H100 80GB HBM3:

   | Endpoint name | Dockerfile Path | Container disk | Env | Network volume | Cached Model |
   | ------------- | --------------- | -------------- | --- | -------------- | ------------ |
   | `ltx2-eval-diag` | `Dockerfile.diag` | 20 GB | (none) | — | — |
   | `ltx2-bake` | `Dockerfile.bake` | 60 GB | `MODEL_SOURCE=bake, MODEL_PATH=/opt/models/ltx-2` | — | — |
   | `ltx2-live` | `Dockerfile.live` | 40 GB | `MODEL_SOURCE=live, MODEL_PATH=Lightricks/LTX-2, HF_TOKEN=…` | — | — |
   | `ltx2-cache` | `Dockerfile.live` | 40 GB | `MODEL_SOURCE=cache, MODEL_PATH=auto` | — | `Lightricks/LTX-2` |
   | `ltx2-volume` | `Dockerfile.live` | 40 GB | `MODEL_SOURCE=volume, MODEL_PATH=/runpod-volume/ltx-2` | `ltx2-eval-vol` (200 GB, EU-RO-1) | — |

   Common settings on every endpoint:
   - Workers: Min 0, Max 1
   - Idle timeout: 5 s
   - Flashboot: **off** (we need genuine cold starts)
   - Execution timeout: 1800000 ms
   - Allowed CUDA: 12.4+

   Grab each endpoint ID and export it:

   ```bash
   export RUNPOD_API_KEY=rpa_...
   export DIAG_ENDPOINT_ID=...
   export LTX2_BAKE_EID=...
   export LTX2_LIVE_EID=...
   export LTX2_CACHE_EID=...
   export LTX2_VOLUME_EID=...
   ```

4. **Seed the network volume** (one-time, does not need the console):

   ```bash
   python seed_volume.py --volume-id $VOLUME_ID --datacenter EU-RO-1 \
     --s3-access-key ... --s3-secret-key ...
   ```

   S3 credentials for the volume are shown once in the console under
   User Settings → Data Center Access (and only there).

## Phase 1 — caching probe (§10)

```bash
RUNPOD_API_KEY=$RUNPOD_API_KEY \
DIAG_ENDPOINT_ID=$DIAG_ENDPOINT_ID \
DIAG_MODE=probe TRIALS=3 \
./drive_diagnostics.sh
```

Compare `hostname`, `boot_id`, and the GPU UUID across the 3 trials.

## Phase 2 — GPU microbench (§11)

```bash
RUNPOD_API_KEY=$RUNPOD_API_KEY \
DIAG_ENDPOINT_ID=$DIAG_ENDPOINT_ID \
DIAG_MODE=gpu TRIALS=3 \
./drive_diagnostics.sh
```

## Phase 3 — LTX-2 cold start / compute / invocation (§1, §3, §4)

For each of the four variants:

```bash
EID=$LTX2_BAKE_EID            # or live/cache/volume
VARIANT=bake
python benchmark.py coldstart --endpoint-id $EID --trials 3 \
  --sleep-between 180 --model-source-hint $VARIANT --estimate-skew
python benchmark.py compute   --endpoint-id $EID --trials 3 \
  --width 1920 --height 1088 --num-frames 121 --steps 8 --fallback
python benchmark.py invocation --endpoint-id $EID \
  --width 1280 --height 720 --steps 4 --save-mp4
```

## Phase 4 — iteration velocity (§6)

```bash
python benchmark.py iteration --trials 3 --branch main
```

This covers the 3 edit kinds × 2 commands matrix and records
first-build vs median wall clock separately, because the first build
after a release is always longer.

## RunPod + fal differences to keep in mind

- Queue endpoints return `delayTime` (ms) and `executionTime` (ms) on every job. `delayTime` on a scale-0 → first-request call is the cold-start wall as seen by the provider. LB endpoints use `/ping` instead, but LB is not exposed in the REST API (console-only toggle).
- RunPod has a 30 MB per-response payload cap on LB, and queue responses are also bounded. `seed_volume.py` covers the ingest side; `benchmark.py invocation` base64-encodes a short 25-frame clip into the response body as a delivery proxy.
- Flashboot skews cold-start measurements: on by default, off for the eval endpoints so we can see worker boot + image pull time.
- Builds are keyed on GitHub **releases**, not commits. The `iteration` scenario in `benchmark.py` creates lightweight tags.
