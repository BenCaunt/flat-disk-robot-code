#!/usr/bin/env bash
set -Eeuo pipefail

LOG="${LOG:-/workspace/open_vocab_nav_research_loop.log}"
EXIT_FILE="${EXIT_FILE:-/workspace/open_vocab_nav_research_loop.exit}"
PROJECT_DIR="${PROJECT_DIR:-/workspace/flat-disk-robot-code}"
CONFIG="${CONFIG:-experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep_runpod_linux.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/open_vocab_nav_research_loop}"
VARIANT="${VARIANT:-qwen_topomap_memory_clip}"
EPISODE="${EPISODE:-living_room_sofa}"
PARALLELISM="${PARALLELISM:-1}"
COMPLETE_EXIT_CODES="${COMPLETE_EXIT_CODES:-0,2}"
WARMHUB_REPO="${WARMHUB_REPO:-bencaunt-2/open-vocab-nav-research-loop}"
UV_EXTRAS="${UV_EXTRAS-thor}"
UV_WITH="${UV_WITH-torch,transformers}"
PREFLIGHT_ENDPOINTS="${PREFLIGHT_ENDPOINTS:-${PRELIGHT_ENDPOINTS:-1}}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
COMMIT_WARMHUB="${COMMIT_WARMHUB:-1}"
INIT_WARMHUB_REPO="${INIT_WARMHUB_REPO:-0}"
DRY_RUN="${DRY_RUN:-0}"
TASK_ID="${TASK_ID:-}"
AGENT_NAME="${AGENT_NAME:-${RUNPOD_POD_ID:-${HOSTNAME:-runpod-open-vocab-nav}}}"
CLAIM_TASK="${CLAIM_TASK:-1}"
FINISH_TASK="${FINISH_TASK:-1}"
START_QWEN_SERVER="${START_QWEN_SERVER:-0}"
START_THOR_XORG="${START_THOR_XORG:-1}"
THOR_XORG_DISPLAY="${THOR_XORG_DISPLAY:-0}"
PREPARE_OBJECT_DRIVE_ENV="${PREPARE_OBJECT_DRIVE_ENV:-1}"
OBJECT_DRIVE_VENV="${OBJECT_DRIVE_VENV:-/workspace/open_vocab_nav_object_drive_venv}"
OBJECT_DRIVE_PACKAGES="${OBJECT_DRIVE_PACKAGES:-torch,transformers<5,timm,einops}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_SERVER_LOG="${QWEN_SERVER_LOG:-/workspace/qwen_vllm.log}"
QWEN_SERVER_PID="${QWEN_SERVER_PID:-/workspace/qwen_vllm.pid}"
QWEN_SERVER_TIMEOUT_S="${QWEN_SERVER_TIMEOUT_S:-900}"
export QWEN_MODEL QWEN_HOST QWEN_PORT QWEN_SERVER_LOG QWEN_SERVER_PID QWEN_SERVER_TIMEOUT_S
export START_THOR_XORG THOR_XORG_DISPLAY
export PREPARE_OBJECT_DRIVE_ENV OBJECT_DRIVE_VENV OBJECT_DRIVE_PACKAGES

rm -f "${EXIT_FILE}"
mkdir -p "$(dirname "${LOG}")" "${OUTPUT_DIR}"

exec > >(tee -a "${LOG}") 2>&1

warmhub_cmd=(uv run --project sim)
research_prefix=(uv run --project sim)
if [[ -n "${UV_EXTRAS}" ]]; then
  extra_names=()
  IFS=',' read -r -a extra_names <<< "${UV_EXTRAS}"
  for extra in "${extra_names[@]}"; do
    if [[ -n "${extra}" ]]; then
      warmhub_cmd+=(--extra "${extra}")
      research_prefix+=(--extra "${extra}")
    fi
  done
fi
if [[ -n "${UV_WITH}" ]]; then
  with_names=()
  IFS=',' read -r -a with_names <<< "${UV_WITH}"
  for package in "${with_names[@]}"; do
    if [[ -n "${package}" ]]; then
      warmhub_cmd+=(--with "${package}")
      research_prefix+=(--with "${package}")
    fi
  done
fi
warmhub_cmd+=(flatdisk-sim-research-warmhub --repo "${WARMHUB_REPO}")

finish() {
  code=$?
  echo "${code}" > "${EXIT_FILE}"
  if [[ -n "${TASK_ID}" && "${FINISH_TASK}" == "1" && -d "${PROJECT_DIR}" ]]; then
    set +e
    cd "${PROJECT_DIR}"
    task_status="complete"
    if ! exit_code_is_complete "${code}"; then
      task_status="failed"
    fi
    finish_cmd=(
      "${warmhub_cmd[@]}" task-finish
      --task "${TASK_ID}" \
      --agent "${AGENT_NAME}" \
      --status "${task_status}" \
      --summary "Runpod open-vocab navigation job exited with code ${code}." \
      --evidence-artifact "${LOG}" \
      --evidence-artifact "${EXIT_FILE}" \
      --evidence-artifact "${OUTPUT_DIR}" \
      --next-action "Inspect research_loop_summary.json, Warmhub NavEvalRun records, and failure observations." \
      --confidence 0.7
    )
    if [[ "${START_QWEN_SERVER}" == "1" ]]; then
      finish_cmd+=(--evidence-artifact "${QWEN_SERVER_LOG}")
    fi
    "${finish_cmd[@]}"
  fi
  echo "[exit] ${code}"
  if exit_code_is_complete "${code}"; then
    exit 0
  fi
  exit "${code}"
}
trap finish EXIT

exit_code_is_complete() {
  local code="$1"
  local code_name
  local code_names=()
  IFS=',' read -r -a code_names <<< "${COMPLETE_EXIT_CODES}"
  for code_name in "${code_names[@]}"; do
    if [[ "${code}" == "${code_name}" ]]; then
      return 0
    fi
  done
  return 1
}

echo "[start] $(date -Iseconds)"
echo "[config] project_dir=${PROJECT_DIR}"
echo "[config] config=${CONFIG}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] variant=${VARIANT}"
echo "[config] episode=${EPISODE}"
echo "[config] warmhub_repo=${WARMHUB_REPO}"
echo "[config] uv_extras=${UV_EXTRAS}"
echo "[config] uv_with=${UV_WITH}"
echo "[config] complete_exit_codes=${COMPLETE_EXIT_CODES}"
echo "[config] start_thor_xorg=${START_THOR_XORG}"
echo "[config] start_qwen_server=${START_QWEN_SERVER}"
echo "[config] prepare_object_drive_env=${PREPARE_OBJECT_DRIVE_ENV}"
echo "[config] qwen_model=${QWEN_MODEL}"
echo "[config] qwen_endpoint=http://${QWEN_HOST}:${QWEN_PORT}/v1/chat/completions"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[abort] PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"

if [[ -n "${TASK_ID}" && "${CLAIM_TASK}" == "1" ]]; then
  "${warmhub_cmd[@]}" task-claim \
    --task "${TASK_ID}" \
    --owner "${AGENT_NAME}" \
    --note "Runpod job started on ${RUNPOD_POD_ID:-${HOSTNAME:-unknown-host}} with log ${LOG}."
fi

if [[ "${START_THOR_XORG}" == "1" ]]; then
  export DISPLAY="${DISPLAY:-:${THOR_XORG_DISPLAY}}"
  chmod +x scripts/runpod_start_thor_xorg.sh
  scripts/runpod_start_thor_xorg.sh
fi

if [[ "${START_QWEN_SERVER}" == "1" ]]; then
  chmod +x scripts/runpod_start_qwen_vllm.sh
  scripts/runpod_start_qwen_vllm.sh
fi

if [[ "${PREPARE_OBJECT_DRIVE_ENV}" == "1" ]]; then
  chmod +x scripts/runpod_prepare_object_drive_env.sh
  source scripts/runpod_prepare_object_drive_env.sh
fi

research_cmd=(
  "${research_prefix[@]}" flatdisk-sim-research-loop
  --config "${CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --variant "${VARIANT}"
  --episode "${EPISODE}"
  --parallelism "${PARALLELISM}"
  --warmhub-repo "${WARMHUB_REPO}"
)

if [[ "${PREFLIGHT_ENDPOINTS}" == "1" ]]; then
  research_cmd+=(--preflight-endpoints)
fi
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  research_cmd+=(--preflight-only)
fi
if [[ "${COMMIT_WARMHUB}" == "1" ]]; then
  research_cmd+=(--commit-warmhub)
fi
if [[ "${INIT_WARMHUB_REPO}" == "1" ]]; then
  research_cmd+=(--init-warmhub-repo)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  research_cmd+=(--dry-run)
fi

echo "[command] ${research_cmd[*]}"
"${research_cmd[@]}"
echo "[done] $(date -Iseconds)"
