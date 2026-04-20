# RunPod LTX-2 benchmark findings

> **Status: partial.** The code harness is complete and reproducible via
> `create_endpoints.sh` + `benchmark.py`, but no on-cloud measurements
> have been captured yet. Rows below are placeholders marked **TBM**
> ("to be measured") until the Phase 1-4 runs are driven on a RunPod
> account with capacity and a connected GitHub/Hub repo. See
> `README.md` and the top-level plan for execution steps.

Raw observations from `exploration/runpod/` on `patrickhulce/serverless-gpu-exploration`
(auth=private, GPU-H100 HBM3 80 GB, `idleTimeout=5 s`, `flashboot=false`,
`Dockerfile.{live,bake,diag}` = `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
+ `python3` system + `requirements.txt`). Model = `Lightricks/LTX-2`
via `diffusers.LTX2Pipeline` in bf16, VAE tiling enabled.

Four `MODEL_SOURCE` variants exercised:
- `bake`  — weights baked into the image at `/opt/models/ltx-2`,
- `live`  — `from_pretrained("Lightricks/LTX-2")` at each cold start,
- `cache` — RunPod Model Caching feature; `/runpod-volume/huggingface-cache/hub/...`,
- `volume` — self-seeded network volume at `/runpod-volume/ltx-2`.

All server-side timestamps come from JSON `[TIMING]` lines emitted by
`mark()` in `app.py`, captured from `/info`, `/warmup`, `/generate`
response bodies (RunPod does not expose `fal runners logs`-style
streaming setup() output on LB endpoints; see §8).

Executive summary (to be filled in after Phase 3):
- **TBM** – comparative cold-start story between bake/live/cache/volume.
- **TBM** – whether RunPod Model Caching actually schedules workers onto
  hosts that already have the weights, or whether it falls back to
  per-worker download.
- **TBM** – whether RunPod's 200-400 MB/s documented volume throughput
  holds from inside a worker calling `from_pretrained` on the volume
  (many small files → NFS-style per-file overhead is plausible).

---

## 1. Model storage throughput

| Source                                                   | Throughput (effective) | Method |
|----------------------------------------------------------|-----------------------:|--------|
| HuggingFace → Workbench (`snapshot_download`)            | **TBM**                | `seed_volume.py --pipeline-only` timing HF side |
| Workbench → RunPod S3 volume (`boto3.put_object`)        | **TBM**                | `seed_volume.py` S3 side, against `s3api-<dc>.runpod.io` |
| HuggingFace → RunPod runner, cold (`from_pretrained`)    | **TBM**                | `coldstart-live` trials, `after_from_pretrained` Δ |
| HuggingFace → RunPod runner, Model-Caching hit           | **TBM**                | `coldstart-cache` trials |
| RunPod network volume → runner (`from_pretrained`)       | **TBM**                | `coldstart-volume` trials |
| Image-baked weights → GPU (`from_pretrained` local-only) | **TBM**                | `coldstart-bake` trials |

_Pending_: (a) HF-vs-S3 first-time throughput (usually worse than docs;
we expect 200-400 MB/s claimed, see if reality is closer to 50-100);
(b) whether `snapshot_download` uses a single-connection pull from HF
or parallelises — if single-connection we get 40-80 MB/s.

---

## 2. Compute / accelerator

All values are from the **warm** worker (pipeline already on GPU). Model
is LTX-2 distilled in bf16, 8 inference steps, VAE tiling on, no CPU offload.

| Resolution × frames                                | wall_s | s/frame | ms/step | VRAM peak / total |
|----------------------------------------------------|-------:|--------:|--------:|-------------------|
| 1280 × 704 × 49 fr  (≈720p × 2 s @24 fps)          | **TBM** | **TBM** | **TBM** | **TBM**           |
| 1920 × 1088 × 121 fr (≈1080p × 5 s @24 fps)        | **TBM** | **TBM** | **TBM** | **TBM**           |

Expectation (from fal on the same H100 HBM3): ~46 s for 1080p × 121 fr
× 8 steps, 73 GB VRAM peak. If RunPod's H100 reports the same
device-info block (SM count, HBM bandwidth) we expect the compute rows
to land within ±5%.

---

## 3. VRAM headroom

Expected: 1080p × 121 fr × 8 steps bf16 fits on H100 80 GB with VAE
tiling and no CPU offload. Measured peak **TBM**. (fal measured 73.4 GB
/ 80 GB = 92% utilised on the same workload.)

| state                          | allocated | reserved | of 80 GB |
|--------------------------------|----------:|---------:|---------:|
| Post-setup, idle               | **TBM**   | **TBM**  | **TBM**  |
| Peak during 720p × 49 gen      | **TBM**   | n/a      | **TBM**  |
| Peak during 1080p × 121 gen    | **TBM**   | n/a      | **TBM**  |

---

## 4. Cold start — per-variant timeline

Four separate regimes, one per storage strategy. All drive the same
benchmark (`benchmark.py coldstart --trials 3 --sleep-between 180
--estimate-skew`) with the endpoint scaled to 0 between each trial.

### (a) `bake` — weights in image, no external storage touched

| event                          | wall UTC      | Δ from prior |
|--------------------------------|---------------|-------------:|
| Client POST /warmup            | **TBM**       | 0            |
| Worker container started (/ping 204) | **TBM** | +Δ image pull |
| setup_enter                    | **TBM**       | +Δ           |
| before_from_pretrained         | **TBM**       | +Δ           |
| after_from_pretrained          | **TBM**       | **TBM (pure disk read)** |
| after_to_cuda                  | **TBM**       | +Δ           |
| setup_ready (/ping 200)        | **TBM**       | +Δ           |

**Client → ready: TBM.** The `bake` image is ~95 GB; Docker pull
wall is the single largest term.

### (b) `live` — from_pretrained hits HF CDN on every cold start

| event                          | wall UTC      | Δ from prior |
|--------------------------------|---------------|-------------:|
| Client POST /warmup            | **TBM**       | 0            |
| Worker container started       | **TBM**       | +Δ image pull |
| setup_enter                    | **TBM**       | +Δ           |
| after_from_pretrained          | **TBM**       | **TBM (HF download)** |
| after_to_cuda                  | **TBM**       | +Δ           |
| setup_ready                    | **TBM**       | +Δ           |

### (c) `cache` — RunPod Model Caching feature hit

Does RunPod's Model Caching pre-scheduling actually place workers on
already-cached hosts? If yes, cold start should be close to (b) with the
HF download swapped for a local disk read. If no, the cache is a one-per-
host download that happens the first time a worker lands there —
effectively (b) for probe #1 and (c) for probes #2+ on the same host.

| event                          | wall UTC      | Δ from prior |
|--------------------------------|---------------|-------------:|
| Client POST /warmup            | **TBM**       | 0            |
| Worker container started       | **TBM**       | +Δ image pull |
| setup_enter                    | **TBM**       | +Δ           |
| after_from_pretrained          | **TBM**       | **TBM (local cache read)** |
| after_to_cuda                  | **TBM**       | +Δ           |

### (d) `volume` — self-seeded network volume

| event                          | wall UTC      | Δ from prior |
|--------------------------------|---------------|-------------:|
| Client POST /warmup            | **TBM**       | 0            |
| Worker container started       | **TBM**       | +Δ image pull |
| setup_enter                    | **TBM**       | +Δ           |
| after_from_pretrained          | **TBM**       | **TBM (volume read)** |
| after_to_cuda                  | **TBM**       | +Δ           |

### 4-way cold-start comparison (expected shape)

| Mode                | Image pull | `from_pretrained` | `to_cuda` | setup() total | Client → ready |
|---------------------|-----------:|------------------:|----------:|--------------:|---------------:|
| `bake`              | **TBM** (big) | **TBM (small)** | **TBM**   | **TBM**       | **TBM**        |
| `live` (worst case) | **TBM**    | **TBM (big)**     | **TBM**   | **TBM**       | **TBM**        |
| `cache` (best)      | **TBM**    | **TBM (small)**   | **TBM**   | **TBM**       | **TBM**        |
| `volume`            | **TBM**    | **TBM (medium)**  | **TBM**   | **TBM**       | **TBM**        |

---

## 5. Iteration velocity — `git push` → RunPod Hub rebuild → ready

Measured with the 3×2 edit matrix (`python_only`, `pip_dep`,
`dockerfile_layer`) × (`git_push`, `runpod_dev`). First trial for each
cell is recorded separately from the steady-state median because
RunPod's Hub builder has no node-local layer cache for the very first
build.

| edit kind        | first trial | steady median |
|------------------|------------:|--------------:|
| `python_only` → `git push` | **TBM** | **TBM** |
| `pip_dep` → `git push`     | **TBM** | **TBM** |
| `dockerfile_layer` → `git push` | **TBM** | **TBM** |

---

## 6. Iteration velocity — ephemeral dev (`runpod project start`)

| scenario                                   | wall     |
|--------------------------------------------|---------:|
| First `runpod project start` of the day    | **TBM**  |
| Subsequent start on warm pod               | **TBM**  |

`runpod project start` runs on a **different compute pool** (dev pods,
not serverless workers). It is the closest analogue to `fal run` but
it is not the production path — rebuilds through `git push` + Hub are.
This will likely hurt RunPod's iteration-velocity score vs fal's.

---

## 7. Invocation asset latency

Measured on a warm worker, same request body (1280 × 704 × 49 fr × 8
steps) with `upload_result` flipped.

| scenario                                  | server-side wall | delivery time |
|-------------------------------------------|-----------------:|--------------:|
| `upload_result=false` (inline null)       | **TBM**          | 0             |
| `upload_result=true` (base64 in response) | **TBM**          | **TBM**       |

**Limitation**: RunPod load-balancing endpoints have no provider-native
signed URL. We return the mp4 bytes inline as base64 in the response
body; the client writes them to disk. This caps output resolution at the
30 MB LB payload limit. A 1080p × 121 fr H.264 mp4 at `quality=7`
measured **TBM MB** — **TBM** whether it fits.

For workloads that need signed URLs, the workaround is to have the
worker push to an external S3 bucket at upload-time; RunPod doesn't
provide a first-party equivalent to `fal.toolkit.File.from_path`.

---

## 8. Debug experience — qualitative, empirical

**TBM** after Phase 3 runs. The incidents log will capture each
failure mode and the wall-clock cost of diagnosing it. Expected rough
shape based on the RunPod docs and console UI:

- LB endpoint setup() stdout/stderr is visible in the console's
  **Live Logs** tab; CLI does not stream it. Equivalent to
  `fal runners logs` reliability = ~partial.
- Health-probe `/ping` status transitions are logged. That alone beats
  fal's silent "setup still running" wait.
- REST API PATCH of workersMin/Max is synchronous and returns the new
  state immediately — no eventual-consistency wait like fal's apps-scale
  commands.

---

## 10. How RunPod actually caches (node/storage probe)

The answers that `diagnostics.py` is designed to pin down:

1. **Does the Model Caching feature pre-schedule workers onto hosts
   that already have the weights?** — compare marker-file reads on
   `/runpod-volume/huggingface-cache/hub/...` across three cold
   probes. If the directory is populated on probe #1 on a **new
   host** (`boot_id` unseen before), the feature prefetches at host
   level. If only probes on a host previously seen populate it, it
   downloads-on-first-hit.
2. **Is `/runpod-volume` node-local or a networked volume?** — compare
   marker-file reads across probes that landed on **different hosts**.
   RunPod claims network volumes are region-scoped persistent storage
   attached via JuiceFS; the `/proc/self/mountinfo` filter will show
   the actual fs type.
3. **Is there a node-local Docker layer cache?** — compare /ping
   204→200 wall on probe #1 (new-to-host image pull) vs probe #2 on the
   same host (should be seconds).
4. **What does the network volume look like under load?** — same marker
   methodology but on the volume endpoint; `du -sb /runpod-volume/ltx-2`
   reports logical bytes, so any delta from upload bytes is real cost.

### Method

Three guaranteed cold starts on `diag-runpod-eval` (tiny image, no
LTX-2), driven by `drive_diagnostics.sh 3` which:

- PATCH /v1/endpoints/<id>/update `{workersMin:0,workersMax:0}` after each
  probe,
- polls GET /v1/endpoints/<id> until `workers[].status` has no
  non-terminated entries,
- PATCH back to `{workersMin:0,workersMax:1}`,
- POSTs /probe which stamps markers into 8 candidate paths and reads
  markers left by previous probes,
- each probe also returns container `hostname`, `/proc/sys/kernel/random/boot_id`,
  GPU UUID, `/proc/self/mountinfo` (filtered), all HF env vars.

Artifacts: `results/diag-1.json`, `diag-2.json`, `diag-3.json`,
`diag-summary.json`.

### Result

**TBM.** Template once the probes run:

| probe | hostname (container) | boot_id (host) | GPU UUID suffix | mountinfo `/runpod-volume` fs | marker_reads non-null paths |
|------:|----------------------|----------------|-----------------|-------------------------------|-----------------------------|
| 1     | **TBM**              | **TBM**        | **TBM**         | **TBM**                       | (none, first probe)         |
| 2     | **TBM**              | **TBM**        | **TBM**         | **TBM**                       | **TBM**                     |
| 3     | **TBM**              | **TBM**        | **TBM**         | **TBM**                       | **TBM**                     |

### Interpretation

Questions the table must answer (to be written after data is in):
- `$HF_HOME` injection: does RunPod set it, and where does it point?
- Is `/runpod-volume` the same mount for the Model Caching feature and
  for a user-attached volume, or distinct bind mounts?
- Do markers on `/root`, `/tmp`, `/workspace` survive scale-to-zero on
  the **same host**, or are they always container-ephemeral as on fal?
- Is the image layer cache host-local (seconds for #2 on same host) or
  always cold (minutes for every probe)?

---

## 11. GPU microbenchmark (cuBLAS / cuDNN / SDPA / HBM / PCIe)

Drive `/gpu` on the diagnostics endpoint three times across three
distinct cold starts (forced via `drive_diagnostics.sh` style
scale-to-zero between probes). Goal: confirm RunPod's H100 is a
full-fat SXM5 at advertised performance, not a PCIe variant, a MIG
slice, or a throttled shared tenant.

### Device info (to be captured)

| field              | value  |
|--------------------|--------|
| device             | **TBM** (expected `NVIDIA H100 80GB HBM3`) |
| SM count           | **TBM** (expected 132 for SXM5, 114 for PCIe) |
| compute capability | **TBM** |
| VRAM reported      | **TBM** |
| CUDA runtime       | **TBM** (should be ≥12.4) |
| GPU UUIDs observed | **TBM** (3 distinct cards wanted) |

### Kernel-level throughput

Medians over 10-30 iterations after warmup, timed with CUDA events.
Raw: `results/diag-gpu-{1,2,3}.json`.

| kernel                                          | run #1   | run #2   | run #3   | vs published peak |
|-------------------------------------------------|---------:|---------:|---------:|-------------------|
| cuBLAS matmul 8192² fp32                        | **TBM**  | **TBM**  | **TBM**  | **TBM**           |
| cuBLAS matmul 8192² tf32                        | **TBM**  | **TBM**  | **TBM**  | **TBM**           |
| cuBLAS matmul 8192² fp16                        | **TBM**  | **TBM**  | **TBM**  | **TBM**           |
| cuBLAS matmul 8192² bf16                        | **TBM**  | **TBM**  | **TBM**  | **TBM**           |
| HBM3 D→D memcpy (2 GB, read+write)              | **TBM**  | **TBM**  | **TBM**  | **TBM**           |
| PCIe Gen5 x16 H→D (512 MB, pinned)              | **TBM**  | **TBM**  | **TBM**  | **TBM**           |
| PCIe Gen5 x16 D→H                               | **TBM**  | **TBM**  | **TBM**  | —                 |
| SDPA bf16 [1,24,2048,128]                       | **TBM**  | **TBM**  | **TBM**  | **TBM**           |
| SDPA bf16 [1,24,4096,128]                       | **TBM**  | **TBM**  | **TBM**  | —                 |
| SDPA bf16 [1,24,8192,128]                       | **TBM**  | **TBM**  | **TBM**  | —                 |
| Conv3D bf16 256→256, 1×256×16×64³               | **TBM**  | **TBM**  | **TBM**  | **TBM**           |

### Interpretation

**TBM.** Expected outcomes given H100 SXM5 80 GB HBM3:
- bf16 matmul @ 8192² ≈ **790 TFLOPs** (~80% of 989 dense peak),
- HBM3 memcpy ≈ **3.0 TB/s** (~90% of 3.35 TB/s spec),
- PCIe Gen5 ≈ **55 GB/s** sustained H↔D,
- SDPA picks flash kernel → ~350 TFLOPs at seq 8192 (not the math
  fallback at <100 TFLOPs).

Cross-card variance will tell us whether RunPod time-slices tenants on
the same GPU (expect >5% variance) or dedicates per tenant (expect ±3%).

---

## 9. Rubric-ready numbers (TL;DR)

| Metric                               | Value |
|--------------------------------------|-------|
| Model storage throughput, HF → Workbench | **TBM** |
| Model storage throughput, HF → RunPod runner, cold (`live`) | **TBM** |
| Model storage throughput, HF → RunPod runner, Model-Cache hit (`cache`) | **TBM** |
| Model storage throughput, Workbench → RunPod S3 volume (one-time seed) | **TBM** |
| Model storage throughput, RunPod network volume → runner (`volume`) | **TBM** |
| One-time seeding cost: LTX-2 ~90 GB → volume  | **TBM** |
| Per-step inference @ 720p × 49 fr, H100, bf16, 8 steps | **TBM** |
| Per-frame inference @ 720p × 49 fr   | **TBM** |
| Per-step inference @ 1080p × 121 fr  | **TBM** |
| Per-frame inference @ 1080p × 121 fr | **TBM** |
| Cold start, `bake`, scale-0 → ready  | **TBM** |
| Cold start, `live`, scale-0 → ready  | **TBM** |
| Cold start, `cache`, scale-0 → ready | **TBM** |
| Cold start, `volume`, scale-0 → ready | **TBM** |
| `setup()` wall, `bake`               | **TBM** |
| `setup()` wall, `live`               | **TBM** |
| `setup()` wall, `cache`              | **TBM** |
| `setup()` wall, `volume`             | **TBM** |
| `from_pretrained` wall, `bake`       | **TBM** |
| `from_pretrained` wall, `live` cold  | **TBM** |
| `from_pretrained` wall, `cache`      | **TBM** |
| `from_pretrained` wall, `volume`     | **TBM** |
| `git push` → Hub rebuild → ready, best | **TBM** |
| `git push` → Hub rebuild → ready, typical | **TBM** |
| `git push` → Hub rebuild → ready, worst | **TBM** |
| `runpod project start` iteration, warm | **TBM** |
| `runpod project start` iteration, cold | **TBM** |
| VRAM at idle (bf16, VAE tiling)      | **TBM** |
| VRAM peak @ 720p × 49 fr             | **TBM** |
| VRAM peak @ 1080p × 121 fr           | **TBM** |
| 1080p × 121 fr feasible @ 80 GB, bf16, no CPU offload? | **TBM** |
| Output asset latency (mp4 b64 inline vs null)  | **TBM** |
| Input asset latency (signed URL vs inline bytes) | n/a (no provider signed URL) |
| Debug experience                      | **TBM** |
| Accelerator identity                  | **TBM** |
| cuBLAS matmul 8192² bf16              | **TBM** |
| cuBLAS matmul 8192² fp16              | **TBM** |
| cuBLAS matmul 8192² tf32              | **TBM** |
| cuBLAS matmul 8192² fp32              | **TBM** |
| HBM3 device bandwidth (copy)          | **TBM** |
| PCIe Gen5 H↔D bandwidth               | **TBM** |
| SDPA bf16 (flash) @ seq 8192          | **TBM** |
| Conv3D bf16 (video-VAE-ish shape)     | **TBM** |
| Cross-card perf variance              | **TBM** |
