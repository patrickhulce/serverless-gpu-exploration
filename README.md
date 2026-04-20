# RunPod LTX-2 exploration

Port of `exploration/fal/` onto RunPod Serverless. Produces
`FINDINGS.md` using the same rubric format so the two providers can be
compared row-by-row.

Reference workload: `Lightricks/LTX-2` via `diffusers.LTX2Pipeline`,
1920×1088×121 fr bf16 on an H100 80GB HBM3 worker (A100-80GB fallback
priced in). VAE tiling on, no CPU offload.

## Layout

```
exploration/runpod/
├── app.py                    # LTX-2 FastAPI server (ports fal app.py)
├── diagnostics.py            # node/caching probe + GPU microbench
├── benchmark.py              # client-side driver (ports fal benchmark.py)
├── seed_volume.py            # HF -> local -> RunPod S3 volume seeder
├── drive_diagnostics.sh      # guaranteed-cold probe driver
├── create_endpoints.sh       # one-shot: create templates + endpoints via REST
├── Dockerfile.diag           # tiny diagnostics image (no diffusers)
├── Dockerfile.live           # LTX-2 image, NO weights baked (live/cache/volume)
├── Dockerfile.bake           # LTX-2 image with --pipeline-only weights baked
├── requirements.txt          # LTX-2 runtime deps
├── requirements.diag.txt     # tiny diagnostics deps (torch + fastapi)
├── results/                  # benchmark JSON output, diag probes, mp4s
└── FINDINGS.md               # the rubric deliverable; grown during Phases 1-4
```

## Architecture mapping (fal → RunPod)

| fal                                          | RunPod                                            |
|----------------------------------------------|---------------------------------------------------|
| `fal.App` + `@fal.endpoint("/x")`            | `FastAPI()` + `@app.post("/x")` + `/ping` LB probe|
| `ContainerImage.from_dockerfile`             | RunPod Hub GitHub build OR `docker push ghcr.io/...` |
| `fal files upload /data/ltx-2`               | RunPod Network Volume + S3 API (`s3api-<dc>.runpod.io`) |
| `fal apps runners ... --json`                | `GET /v1/endpoints/{id}` -> `.workers[]`          |
| `fal run` (ephemeral GPU dev)                | `runpod project start` (different compute pool)   |
| `fal.toolkit.File.from_path` (signed URL)    | **no equivalent**; base64-in-response-body (30 MB LB cap) |

## Four LTX-2 storage variants

All four share `app.py`; the variant is picked by the `MODEL_SOURCE`
and `MODEL_PATH` env on the endpoint:

| Variant       | Dockerfile        | Env (set on endpoint)                                    | Storage path at runtime                         |
|---------------|-------------------|----------------------------------------------------------|-------------------------------------------------|
| `bake`        | `Dockerfile.bake` | `MODEL_SOURCE=bake  MODEL_PATH=/opt/models/ltx-2`        | Container rootfs, weights in image (~95 GB)    |
| `live`        | `Dockerfile.live` | `MODEL_SOURCE=live  MODEL_PATH=Lightricks/LTX-2`         | `from_pretrained` hits HF CDN every cold start |
| `cache`       | `Dockerfile.live` | `MODEL_SOURCE=cache MODEL_PATH=auto`                     | `/runpod-volume/huggingface-cache/hub/...` (RunPod Model Caching feature) |
| `volume`      | `Dockerfile.live` | `MODEL_SOURCE=volume MODEL_PATH=/runpod-volume/ltx-2`    | User-seeded network volume                      |

## One-time setup

```bash
# 0. env
export RUNPOD_API_KEY=...     # global key from runpod.io/console
export HF_TOKEN=...            # optional, for rate-limited HF pulls

# 1. push this tree to github.com/patrickhulce/serverless-gpu-exploration
#    and connect the repo in the RunPod web console: Hub -> Github Integration.

# 2. (optional) if you want a fully-scripted path without Hub, build images
#    locally and push them to Docker Hub / GHCR:
#      IMAGE_DIAG=patrickhulce/runpod-eval-diag:1
#      IMAGE_LIVE=patrickhulce/runpod-eval-live:1
#      IMAGE_BAKE=patrickhulce/runpod-eval-bake:1
#    Then run create_endpoints.sh which creates templates+endpoints via REST.

IMAGE_DIAG=<registry>/runpod-eval-diag:1 \
IMAGE_LIVE=<registry>/runpod-eval-live:1 \
IMAGE_BAKE=<registry>/runpod-eval-bake:1 \
./create_endpoints.sh

# 3. Manual UI steps flagged by create_endpoints.sh:
#    - Toggle endpoints to Load Balancing (REST API creates queue-based).
#    - Set Cached Model = Lightricks/LTX-2 on the ltx2-cache endpoint.
#    - Verify the attached network volume ID matches on ltx2-volume.
```

## Phase 1: diagnostics

```bash
export RUNPOD_ENDPOINT_ID=<diag-endpoint-id>
export RUNPOD_ENDPOINT_URL=https://<diag-endpoint-id>.api.runpod.ai
./drive_diagnostics.sh 3
# inspect results/diag-1.json, diag-2.json, diag-3.json, diag-summary.json
```

## Phase 2: GPU microbenchmark

```bash
# Drive /gpu across three distinct cold starts (re-uses the
# drive_diagnostics harness to force cold starts, then POSTs /gpu).
for i in 1 2 3; do
  curl -sS -X POST \
       -H "Authorization: Bearer $RUNPOD_API_KEY" \
       -H 'content-type: application/json' -d '{}' \
       "$RUNPOD_ENDPOINT_URL/gpu" > results/diag-gpu-${i}.json
  # force a cold start before the next probe
  curl -sS -X PATCH \
       -H "Authorization: Bearer $RUNPOD_API_KEY" \
       -H 'content-type: application/json' \
       -d '{"workersMin":0,"workersMax":0}' \
       "https://rest.runpod.io/v1/endpoints/${RUNPOD_ENDPOINT_ID}/update" >/dev/null
  sleep 120
  curl -sS -X PATCH \
       -H "Authorization: Bearer $RUNPOD_API_KEY" \
       -H 'content-type: application/json' \
       -d '{"workersMin":0,"workersMax":1}' \
       "https://rest.runpod.io/v1/endpoints/${RUNPOD_ENDPOINT_ID}/update" >/dev/null
done
```

## Phase 3: reference workload

```bash
# 3a: cold-start, per variant
for variant in bake live cache volume; do
  export RUNPOD_ENDPOINT_ID=<ltx2-${variant}-endpoint-id>
  export RUNPOD_ENDPOINT_URL=https://<...>.api.runpod.ai
  python benchmark.py coldstart \
      --trials 3 --sleep-between 180 --estimate-skew \
      --endpoint-id $RUNPOD_ENDPOINT_ID \
      --model-source-hint $variant
done

# 3b: compute (1920x1088x121)
for variant in bake live cache volume; do
  python benchmark.py compute --width 1920 --height 1088 --num-frames 121 --fallback \
      --url https://<...>.api.runpod.ai
done

# 3c: invocation round-trip (inline mp4 b64)
python benchmark.py invocation --url https://<...>.api.runpod.ai --save-mp4

# 3d: iteration velocity
python benchmark.py iteration --trials 3 --url https://<...>.api.runpod.ai
```

## Authentication

- REST API calls use `Authorization: Bearer $RUNPOD_API_KEY`.
- LB endpoint calls (`/ping`, `/info`, `/warmup`, `/generate`) also
  accept the same Bearer key; without it you'll get 401 unless the
  endpoint is set to public.
