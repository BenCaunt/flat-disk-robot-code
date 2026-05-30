# Visual Localization Prototype

This project now has a first-pass no-odometry visual localization pipeline in
`scripts/visual_localization.py`.

It uses the hloc checkout at:

```text
/Users/bencaunt/Documents/Hierarchical-Localization
```

Run it with that environment, because it contains PyTorch, pycolmap, hloc, and
Zenoh:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HLOC_DEVICE=mps

../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py --help
```

## 1. Build a Map From Video

Record a slow room scan video. Avoid motion blur, cover loops, and keep overlap
between neighboring views. Then build a sparse map:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py build-map \
  --video captures/room_scan.mp4 \
  --map-dir maps/room1 \
  --fps 2 \
  --max-frames 250 \
  --max-image-width 1280 \
  --global-conf netvlad
```

For fast debugging without global retrieval, use:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py build-map \
  --video captures/room_scan.mp4 \
  --map-dir maps/room1 \
  --fps 2 \
  --max-frames 80 \
  --global-conf none
```

Output layout:

- `maps/room1/images/`: extracted video frames
- `maps/room1/hloc/`: feature, pair, and match files
- `maps/room1/sfm/`: pycolmap sparse reconstruction
- `maps/room1/map.json`: manifest used by the live localizer

If the camera intrinsics are known, pass them during mapping:

```bash
--camera-model SIMPLE_RADIAL --camera-params "533.5,320,240,0"
```

or use a calibration JSON produced by `scripts/checkerboard_calibration_logger.py`:

```bash
--camera-calibration captures/esp32_calibration/camera_calibration.json
```

For a single video camera, the default is a shared camera model. Use
`--per-image-camera` only if the video frames do not share one camera.

## 2. Localize One Image

Use this before live Zenoh to validate the map and query intrinsics:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py localize-image \
  --map-dir maps/room1 \
  --image captures/latest.jpg \
  --top-k 10 \
  --matcher-conf NN-superpoint \
  --query-camera-model SIMPLE_RADIAL \
  --query-camera-params "260,160,120,0"
```

If query intrinsics are not provided, the script guesses a `SIMPLE_RADIAL`
camera with `f = 1.2 * max(width, height)`. That is only a fallback. Calibrating
the ESP32 camera will materially improve pose stability. Preferred:

```bash
--query-camera-calibration captures/esp32_calibration/camera_calibration.json
```

## 3. Rerun Map Demo

To visualize a built map, registered camera path, and optional query pose in
Rerun:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/rerun_hloc_map_demo.py \
  --map-dir maps/room1 \
  --pose-json captures/latest_pose.json \
  --output maps/room1/rerun_demo.rrd

../Hierarchical-Localization/.venv/bin/rerun maps/room1/rerun_demo.rrd
```

The local smoke-test demo was generated with:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/rerun_hloc_map_demo.py \
  --map-dir captures/freiburg_test_map \
  --pose-json captures/freiburg_test_pose_nn.json \
  --output captures/freiburg_test_map/rerun_demo.rrd
```

The default recording is 3D-only: sparse points, blue map-camera frustums, the
blue camera trajectory, and the red localized query pose. Add
`--include-query-image` or `--max-images 6` if you also want 2D image panels.

## 4. Live Zenoh Localization

Start the Zenoh router and ESP32 stream as before:

```bash
zenohd --listen tcp/0.0.0.0:7447
```

Then run:

```bash
../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py localize-zenoh \
  --map-dir maps/room1 \
  --namespace flatdisk/xiao \
  --connect tcp/127.0.0.1:7447 \
  --top-k 10 \
  --matcher-conf NN-superpoint \
  --max-rate 2 \
  --query-camera-calibration captures/esp32_calibration/camera_calibration.json
```

The script subscribes to:

```text
flatdisk/xiao/camera/jpeg
```

and publishes JSON visual poses to:

```text
flatdisk/xiao/pose/visual
```

The pose payload includes:

- `localized`
- `camera_center_world`
- `robot_center_world_approx`
- `qvec_wxyz_cam_from_world`
- `tvec_cam_from_world`
- `num_inliers`
- `num_matches`
- `inlier_ratio`
- `latency_ms`

For now `robot_center_world_approx` equals the camera center. To turn this into
the robot base pose, add calibrated camera-to-base extrinsics.

## Matcher Choice

- `--matcher-conf NN-superpoint`: faster, good for frequent updates once the
  scene is easy and the map is clean.
- `--matcher-conf superpoint+lightglue`: slower but stronger, useful for
  relocalization or harder views.

On the local smoke-test map, `NN-superpoint` localized against 6 candidates in
about 0.44 s. `superpoint+lightglue` localized the same query in about 2.3 s.

## Current Limitations

- No odometry, IMU, or temporal filter is used yet.
- Global retrieval only exists if the map was built with `--global-conf netvlad`
  or another hloc global descriptor. Without it, live localization picks evenly
  spaced map frames, which is weaker.
- Robot pose is approximate until camera-to-robot extrinsics are calibrated.
- Absolute scale comes from SfM unless the map is constrained with known metric
  information.
