"""Top-level handler entry point.

RunPod's Hub validator expects a file literally named `handler.py` at
the repo root. Our actual handler logic lives in `app.py` (LTX-2) and
`diagnostics.py` (probe + GPU microbench). This file just chooses
between them at import time based on the `WORKER_ROLE` env var:

  WORKER_ROLE=ltx2  (default)  -> app.handler
  WORKER_ROLE=diag             -> diagnostics.handler

The diag endpoint's Dockerfile.diag sets WORKER_ROLE=diag and uses a
slimmer image that never imports diffusers.
"""

from __future__ import annotations

import os

_role = os.environ.get("WORKER_ROLE", "ltx2").lower()

if _role == "diag":
    from diagnostics import handler  # noqa: F401
else:
    from app import handler  # noqa: F401


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
