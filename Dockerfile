# Default Dockerfile — identical to Dockerfile.live. RunPod's Hub
# validator expects a file literally named `Dockerfile` at the repo
# root. For the 4 LTX-2 variants this same base image is used; the
# storage strategy is picked at runtime via MODEL_SOURCE / MODEL_PATH
# env. The bake variant has its own Dockerfile.bake; diagnostics has
# Dockerfile.diag.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "diffusers==0.37.1" \
      "transformers>=4.52,<5.0" \
      "accelerate==1.1.1" \
      "sentencepiece==0.2.0" \
      "imageio-ffmpeg==0.5.1" \
      "huggingface-hub>=0.34,<1.0" \
      "hf-transfer>=0.1.8,<1.0" \
      "runpod>=1.7,<2.0"

WORKDIR /app
COPY app.py /app/app.py
COPY diagnostics.py /app/diagnostics.py
COPY handler.py /app/handler.py

# WORKER_ROLE=ltx2 (default) -> app.handler; WORKER_ROLE=diag -> diagnostics.handler.
# MODEL_SOURCE/MODEL_PATH only matter for ltx2 role.
ENV WORKER_ROLE=ltx2 \
    MODEL_SOURCE=live \
    MODEL_PATH=Lightricks/LTX-2

CMD ["python", "-u", "handler.py"]
