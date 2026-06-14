#!/usr/bin/env bash
set -Eeuo pipefail

THOR_XORG_DISPLAY="${THOR_XORG_DISPLAY:-0}"
DISPLAY="${DISPLAY:-:${THOR_XORG_DISPLAY}}"
THOR_XORG_WIDTH="${THOR_XORG_WIDTH:-1024}"
THOR_XORG_HEIGHT="${THOR_XORG_HEIGHT:-768}"
THOR_XORG_INSTALL_DEPS="${THOR_XORG_INSTALL_DEPS:-1}"
export DISPLAY

display_ready() {
  if ! command -v xdpyinfo >/dev/null 2>&1; then
    return 1
  fi
  xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1
}

install_display_deps() {
  if [[ "${THOR_XORG_INSTALL_DEPS}" != "1" ]]; then
    return
  fi
  if [[ "$(id -u)" != "0" ]]; then
    echo "[thor-xorg] not root; skipping apt dependency install"
    return
  fi

  missing=()
  if ! command -v lspci >/dev/null 2>&1; then
    missing+=(pciutils)
  fi
  if ! command -v Xorg >/dev/null 2>&1; then
    missing+=(xserver-xorg-core x11-xserver-utils)
  fi
  if ! command -v xdpyinfo >/dev/null 2>&1; then
    missing+=(x11-utils)
  fi

  if ((${#missing[@]} > 0)); then
    echo "[thor-xorg] installing display packages: ${missing[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
  fi
}

link_nvidia_xorg_modules() {
  local nvidia_xorg_dir="/usr/lib/x86_64-linux-gnu/nvidia/xorg"
  if [[ ! -d "${nvidia_xorg_dir}" ]]; then
    return
  fi

  mkdir -p /usr/lib/xorg/modules/drivers /usr/lib/xorg/modules/extensions
  if [[ -f "${nvidia_xorg_dir}/nvidia_drv.so" ]]; then
    ln -sf "${nvidia_xorg_dir}/nvidia_drv.so" /usr/lib/xorg/modules/drivers/nvidia_drv.so
  fi
  if [[ -f "${nvidia_xorg_dir}/libglxserver_nvidia.so" ]]; then
    ln -sf "${nvidia_xorg_dir}/libglxserver_nvidia.so" /usr/lib/xorg/modules/extensions/libglxserver_nvidia.so
  fi
}

start_xorg() {
  local display_number="${DISPLAY#:}"
  if [[ "${display_number}" == *.* ]]; then
    display_number="${display_number%%.*}"
  fi

  if command -v ai2thor-xorg >/dev/null 2>&1; then
    ai2thor-xorg --width "${THOR_XORG_WIDTH}" --height "${THOR_XORG_HEIGHT}" start "${display_number}" || true
  else
    uv run --project sim --extra thor ai2thor-xorg \
      --width "${THOR_XORG_WIDTH}" \
      --height "${THOR_XORG_HEIGHT}" \
      start "${display_number}" || true
  fi
}

echo "[thor-xorg] ensuring AI2-THOR display ${DISPLAY}"
if display_ready; then
  echo "[thor-xorg] display already ready: ${DISPLAY}"
  exit 0
fi

install_display_deps
link_nvidia_xorg_modules
start_xorg
sleep 2

if display_ready; then
  echo "[thor-xorg] display ready: ${DISPLAY}"
  exit 0
fi

echo "[thor-xorg] display failed to start: ${DISPLAY}" >&2
if [[ -f "/var/log/ai2thor-xorg-error.${THOR_XORG_DISPLAY}.log" ]]; then
  tail -n 80 "/var/log/ai2thor-xorg-error.${THOR_XORG_DISPLAY}.log" >&2 || true
fi
if [[ -f "/var/log/ai2thor-xorg.${THOR_XORG_DISPLAY}.log" ]]; then
  tail -n 120 "/var/log/ai2thor-xorg.${THOR_XORG_DISPLAY}.log" >&2 || true
fi
exit 2
