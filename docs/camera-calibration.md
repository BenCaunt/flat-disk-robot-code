# ESP32 Checkerboard Camera Calibration

Use `scripts/checkerboard_calibration_logger.py` to collect checkerboard frames
from the ESP32 Zenoh stream and write calibration files that
`scripts/visual_localization.py` can read directly.

Run with the hloc environment:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HLOC_DEVICE=mps
```

## Capture And Calibrate

Print a checkerboard and count the **inner corners**, not the squares. A common
board is 9 by 6 inner corners.

```bash
../Hierarchical-Localization/.venv/bin/python scripts/checkerboard_calibration_logger.py capture-calibrate \
  --namespace flatdisk/xiao \
  --connect tcp/127.0.0.1:7447 \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 0.024 \
  --output-dir captures/esp32_calibration \
  --max-images 40 \
  --min-images 15 \
  --write-overlays
```

Move the board through the full image: center, edges, corners, tilted, near, and
far. Avoid blurry frames and extreme glare.

Output files:

- `captures/esp32_calibration/accepted/`: accepted checkerboard images
- `captures/esp32_calibration/overlays/`: detected-corner overlays
- `captures/esp32_calibration/camera_calibration.json`: main calibration file
- `captures/esp32_calibration/colmap_camera.txt`: one-line COLMAP camera
- `captures/esp32_calibration/visual_localization_args.txt`: copy/paste args

## Use With Visual Localization

For live robot localization:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py localize-zenoh \
  --map-dir maps/room1 \
  --namespace flatdisk/xiao \
  --connect tcp/127.0.0.1:7447 \
  --query-camera-calibration captures/esp32_calibration/camera_calibration.json \
  --matcher-conf NN-superpoint \
  --top-k 10
```

For maps built from the same ESP32 camera stream:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py build-map \
  --video captures/room_scan.mp4 \
  --map-dir maps/room1 \
  --camera-calibration captures/esp32_calibration/camera_calibration.json \
  --global-conf netvlad
```

If the calibration resolution differs from the query or map frame resolution,
`visual_localization.py` scales focal length and principal point automatically.
Distortion parameters are left unchanged.

The calibration JSON stores COLMAP-compatible `OPENCV` camera parameters under:

```text
colmap.camera_model
colmap.camera_params
colmap.camera_params_string
```
