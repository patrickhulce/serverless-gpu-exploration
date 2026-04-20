#!/usr/bin/env bash
set -e
LOG=/runpod-volume/seed.log
echo "[seed] start $(date -u +%FT%TZ)" | tee -a "$LOG"
pip install --no-cache-dir "huggingface-hub>=0.34,<1.0" "hf-transfer>=0.1.8" 2>&1 | tail -2 | tee -a "$LOG"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/runpod-volume/hf-cache
mkdir -p /runpod-volume/ltx-2 /runpod-volume/hf-cache
T0=$(date +%s)
python -c "
from huggingface_hub import snapshot_download
import time
t0 = time.time()
snapshot_download(
    repo_id='Lightricks/LTX-2',
    local_dir='/runpod-volume/ltx-2',
    max_workers=16,
    allow_patterns=[
        'model_index.json','*.json','LICENSE','README.md',
        'scheduler/*','tokenizer/*','text_encoder/*',
        'transformer/*','vae/*','audio_vae/*',
        'connectors/*','vocoder/*',
    ],
)
print(f'DOWNLOAD_DONE_SEC={time.time()-t0:.1f}')
" 2>&1 | tee -a "$LOG"
T1=$(date +%s)
SIZE_BYTES=$(du -sb /runpod-volume/ltx-2 | awk '{print $1}')
echo "[seed] done $(date -u +%FT%TZ) elapsed=$((T1-T0))s size_bytes=$SIZE_BYTES" | tee -a "$LOG"
touch /runpod-volume/.seed_complete
exec sleep 36000
