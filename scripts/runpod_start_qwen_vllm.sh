#!/usr/bin/env bash
set -Eeuo pipefail

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_SERVER_LOG="${QWEN_SERVER_LOG:-/workspace/qwen_vllm.log}"
QWEN_SERVER_PID="${QWEN_SERVER_PID:-/workspace/qwen_vllm.pid}"
QWEN_SERVER_TIMEOUT_S="${QWEN_SERVER_TIMEOUT_S:-900}"
QWEN_VLLM_INSTALL="${QWEN_VLLM_INSTALL:-1}"
QWEN_VLLM_PACKAGE="${QWEN_VLLM_PACKAGE:-vllm==0.16.0}"
QWEN_VLLM_VENV="${QWEN_VLLM_VENV:-/workspace/open_vocab_nav_qwen_vllm_venv_vllm016}"
QWEN_VLLM_EXTRA_ARGS="${QWEN_VLLM_EXTRA_ARGS:---max-model-len 16384}"
QWEN_HF_HOME="${QWEN_HF_HOME:-/workspace/huggingface}"
QWEN_PIP_CACHE_DIR="${QWEN_PIP_CACHE_DIR:-/workspace/pip-cache}"
QWEN_TMPDIR="${QWEN_TMPDIR:-/workspace/tmp}"

models_url="http://${QWEN_HOST}:${QWEN_PORT}/v1/models"
mkdir -p "$(dirname "${QWEN_SERVER_LOG}")" "${QWEN_HF_HOME}" "${QWEN_PIP_CACHE_DIR}" "${QWEN_TMPDIR}"
export HF_HOME="${HF_HOME:-${QWEN_HF_HOME}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${QWEN_PIP_CACHE_DIR}}"
export TMPDIR="${TMPDIR:-${QWEN_TMPDIR}}"

endpoint_ready() {
  MODELS_URL="${models_url}" python3 - <<'PY'
import os
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(os.environ["MODELS_URL"], timeout=2.0) as response:
        sys.exit(0 if response.status < 500 else 1)
except (OSError, urllib.error.URLError, TimeoutError):
    sys.exit(1)
PY
}

if endpoint_ready; then
  echo "[qwen] endpoint already ready at ${models_url}"
  exit 0
fi

vllm_cmd=""
if [[ -x "${QWEN_VLLM_VENV}/bin/vllm" ]]; then
  vllm_cmd="${QWEN_VLLM_VENV}/bin/vllm"
else
  vllm_cmd="$(command -v vllm || true)"
fi
if [[ -z "${vllm_cmd}" ]]; then
  if [[ "${QWEN_VLLM_INSTALL}" != "1" ]]; then
    echo "[qwen] vllm command missing and QWEN_VLLM_INSTALL=${QWEN_VLLM_INSTALL}" >&2
    exit 2
  fi
  python3 -m venv "${QWEN_VLLM_VENV}"
  "${QWEN_VLLM_VENV}/bin/python" -m pip install --upgrade pip
  "${QWEN_VLLM_VENV}/bin/python" -m pip install --upgrade "${QWEN_VLLM_PACKAGE}"
  vllm_cmd="${QWEN_VLLM_VENV}/bin/vllm"
fi

extra_args=()
if [[ -n "${QWEN_VLLM_EXTRA_ARGS}" ]]; then
  while IFS= read -r -d '' arg; do
    extra_args+=("${arg}")
  done < <(QWEN_VLLM_EXTRA_ARGS="${QWEN_VLLM_EXTRA_ARGS}" python3 - <<'PY'
import os
import shlex
import sys

for arg in shlex.split(os.environ["QWEN_VLLM_EXTRA_ARGS"]):
    sys.stdout.write(arg + "\0")
PY
)
fi

echo "[qwen] starting ${QWEN_MODEL} at http://${QWEN_HOST}:${QWEN_PORT}/v1"
echo "[qwen] log: ${QWEN_SERVER_LOG}"
(
  set -Eeuo pipefail
  exec >>"${QWEN_SERVER_LOG}" 2>&1
  echo "[qwen] server start $(date -Iseconds)"
  if ((${#extra_args[@]} > 0)); then
    exec "${vllm_cmd}" serve "${QWEN_MODEL}" \
      --host "${QWEN_HOST}" \
      --port "${QWEN_PORT}" \
      "${extra_args[@]}"
  fi
  exec "${vllm_cmd}" serve "${QWEN_MODEL}" \
    --host "${QWEN_HOST}" \
    --port "${QWEN_PORT}"
) &
pid=$!
echo "${pid}" > "${QWEN_SERVER_PID}"

deadline=$((SECONDS + QWEN_SERVER_TIMEOUT_S))
while (( SECONDS < deadline )); do
  if endpoint_ready; then
    echo "[qwen] endpoint ready at ${models_url}"
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[qwen] server exited before endpoint became ready; tail follows" >&2
    tail -n 80 "${QWEN_SERVER_LOG}" >&2 || true
    exit 3
  fi
  sleep 5
done

echo "[qwen] timed out after ${QWEN_SERVER_TIMEOUT_S}s waiting for ${models_url}" >&2
tail -n 80 "${QWEN_SERVER_LOG}" >&2 || true
exit 4
