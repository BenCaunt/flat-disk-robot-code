#!/usr/bin/env bash
# Source this script before running harness episodes that use transformer-backed
# object-drive detectors. It prepares one reusable Python env instead of letting
# each visual_servo_object call pay the uv --with cold-start cost.

set -Eeuo pipefail

OBJECT_DRIVE_VENV="${OBJECT_DRIVE_VENV:-/workspace/open_vocab_nav_object_drive_venv}"
OBJECT_DRIVE_PACKAGES="${OBJECT_DRIVE_PACKAGES:-torch,transformers,timm,einops}"
OBJECT_DRIVE_PYTHON="${FLATDISK_OBJECT_DRIVE_PYTHON:-${OBJECT_DRIVE_VENV}/bin/python}"

needs_install=0
if [[ ! -x "${OBJECT_DRIVE_PYTHON}" ]]; then
  echo "[object-drive-env] creating ${OBJECT_DRIVE_VENV}"
  uv venv "${OBJECT_DRIVE_VENV}"
  OBJECT_DRIVE_PYTHON="${OBJECT_DRIVE_VENV}/bin/python"
  needs_install=1
fi

export FLATDISK_OBJECT_DRIVE_PYTHON="${OBJECT_DRIVE_PYTHON}"

if [[ "${needs_install}" == 0 ]]; then
  if ! "${FLATDISK_OBJECT_DRIVE_PYTHON}" - <<'PY'
import importlib
for name in ("numpy", "PIL", "torch", "transformers", "timm", "einops", "zenoh"):
    importlib.import_module(name)
PY
  then
    needs_install=1
  fi
fi

if [[ "${needs_install}" == 1 ]]; then
  install_cmd=(uv pip install --python "${FLATDISK_OBJECT_DRIVE_PYTHON}" -e sim)
  package_names=()
  IFS=',' read -r -a package_names <<< "${OBJECT_DRIVE_PACKAGES}"
  for package in "${package_names[@]}"; do
    if [[ -n "${package}" ]]; then
      install_cmd+=("${package}")
    fi
  done
  echo "[object-drive-env] ${install_cmd[*]}"
  "${install_cmd[@]}"
else
  echo "[object-drive-env] using existing ${FLATDISK_OBJECT_DRIVE_PYTHON}"
fi

"${FLATDISK_OBJECT_DRIVE_PYTHON}" - <<'PY'
import importlib
for name in ("numpy", "PIL", "torch", "transformers", "timm", "einops", "zenoh"):
    importlib.import_module(name)
print("[object-drive-env] dependency import check passed")
PY
