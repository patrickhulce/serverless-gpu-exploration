#!/usr/bin/env bash
# Drive guaranteed cold starts on the diagnostics endpoint, queue-
# based edition.
#
# Between every probe we:
#   1. PATCH /v1/endpoints/$EID workersMin=0, workersMax=0
#   2. poll GET /v1/endpoints/$EID until no live workers remain
#   3. PATCH back to workersMin=0, workersMax=1
#   4. POST /v2/$EID/runsync {"input":{"op":"probe"}}
#      (or {"op":"gpu"} when DIAG_MODE=gpu)
#
# The first cold is always pull+import+setup. Subsequent colds ought
# to be faster iff RunPod is reusing the same physical host. Compare
# probe_snapshot.hostname / boot_id / uuid across runs.
#
# Usage:
#   RUNPOD_API_KEY=... \
#   DIAG_ENDPOINT_ID=... \
#   DIAG_MODE=probe  (or "gpu")
#   TRIALS=3 \
#   ./drive_diagnostics.sh
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
: "${DIAG_ENDPOINT_ID:?set DIAG_ENDPOINT_ID}"
EID="$DIAG_ENDPOINT_ID"
MODE="${DIAG_MODE:-probe}"
TRIALS="${TRIALS:-3}"
SLEEP_BETWEEN_S="${SLEEP_BETWEEN_S:-15}"
DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S:-300}"
JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-1800}"

H_AUTH=(-H "Authorization: Bearer ${RUNPOD_API_KEY}")
H_JSON=(-H "Content-Type: application/json" -H "Accept: application/json")

mkdir -p results
TS=$(date +%Y%m%dT%H%M%S)

patch_workers() {
  local mn="$1" mx="$2"
  echo "[rp] PATCH workersMin=$mn workersMax=$mx" >&2
  curl -sS -X PATCH "https://rest.runpod.io/v1/endpoints/$EID/update" \
    "${H_AUTH[@]}" "${H_JSON[@]}" \
    -d "{\"workersMin\": $mn, \"workersMax\": $mx}" >/dev/null
}

worker_states() {
  curl -sS "https://rest.runpod.io/v1/endpoints/$EID" "${H_AUTH[@]}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); \
         w=d.get('workers') or d.get('workerList') or []; \
         print(','.join(str(x.get('status') or x.get('state') or '?') for x in w))"
}

wait_drained() {
  local deadline=$((SECONDS + DRAIN_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    local s
    s="$(worker_states || true)"
    # live = any non-terminated state
    if ! echo "$s" | grep -Eio 'RUNNING|PROVISIONING|STARTING|IDLE|THROTTLED' >/dev/null; then
      echo "[rp] drained (states=${s:-<none>})" >&2
      return 0
    fi
    echo "[rp] waiting... states=$s" >&2
    sleep 5
  done
  echo "[rp] WARN: drain timeout" >&2
}

post_probe() {
  local body
  if [ "$MODE" = "gpu" ]; then
    body='{"input":{"op":"gpu"}}'
  else
    body='{"input":{"op":"probe"}}'
  fi
  curl -sS --max-time "$JOB_TIMEOUT_S" \
    -X POST "https://api.runpod.ai/v2/$EID/runsync" \
    "${H_AUTH[@]}" "${H_JSON[@]}" \
    -d "$body"
}

for i in $(seq 1 "$TRIALS"); do
  echo "==============================="
  echo "[cold] trial $i of $TRIALS (mode=$MODE)"
  echo "==============================="
  patch_workers 0 0
  wait_drained
  patch_workers 0 1

  t0=$(date +%s.%N)
  OUT="$(post_probe)"
  t1=$(date +%s.%N)
  dt=$(python3 -c "print($t1 - $t0)")
  OUT_PATH="results/diag-${MODE}-${TS}-trial${i}.json"
  python3 -c "import sys,json; \
    d=json.loads(sys.argv[1]); \
    d['_wall_s']=float(sys.argv[2]); \
    open(sys.argv[3],'w').write(json.dumps(d, indent=2))" \
    "$OUT" "$dt" "$OUT_PATH"
  echo "[cold] wrote $OUT_PATH (wall=${dt}s)"

  if (( i < TRIALS )); then
    sleep "$SLEEP_BETWEEN_S"
  fi
done

echo "done. results -> results/diag-${MODE}-${TS}-trial*.json"
