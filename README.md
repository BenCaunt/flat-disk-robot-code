# Seeed XIAO ESP32S3 Sense Camera

PlatformIO project for reading the Seeed Studio XIAO ESP32S3 Sense camera with Arduino `esp_camera`.

## Build and Upload

```sh
pio run
pio run -t upload
pio device monitor
```

If `pio` is not on your `PATH`, this machine also has:

```sh
~/.platformio/penv/bin/pio run
~/.platformio/penv/bin/pio run -t upload
~/.platformio/penv/bin/pio device monitor
```

For serial-only BNO085 debugging, use the separate environment:

```sh
~/.platformio/penv/bin/pio run -e imu_serial_debug -t upload --upload-port /dev/cu.usbmodem2101
~/.platformio/penv/bin/pio device monitor --port /dev/cu.usbmodem2101 --baud 115200
```

The debug firmware does not start the `XIAO-CAM` access point or web server. It connects only as a Wi-Fi station using `include/secrets.h`, then accepts serial commands. The most useful IMU command is `direct`, which resets the BNO085, drains the SHTP advertisement packet, enables rotation-vector, accelerometer, gyro, and linear-acceleration reports, then streams decoded packets. Other commands include `help`, `pins`, `scan`, `raw`, `dump`, `prod`, `getfeat`, `init`, `softreset`, and `reset`.

For the Zenoh streaming prototype, copy `include/secrets.example.h` to `include/secrets.h`, copy `include/local_zenoh.example.h` to `include/local_zenoh.h`, set the laptop IP in `ZENOH_CONNECT`, then upload:

```sh
~/.platformio/penv/bin/pio run -e zenoh_stream -t upload --upload-port /dev/cu.usbmodem2101
zenohd --listen tcp/0.0.0.0:7447
.venv/bin/python scripts/zenoh_companion.py --duration 60
```

To visualize the stream in Rerun:

```sh
.venv/bin/python scripts/zenoh_companion.py --rerun
```

To record without opening the viewer:

```sh
.venv/bin/python scripts/zenoh_companion.py --duration 60 --rerun-save captures/session.rrd
```

The companion records camera and IMU from the robot stream, and records motor
training labels by subscribing directly to `flatdisk/xiao/cmd/motors/percent`.
The 1 Hz robot status motor values are logged separately under `telemetry/`.

To record every run automatically under a namespace folder with a unique file id:

```sh
.venv/bin/python scripts/zenoh_companion.py --duration 60 --connect tcp/192.168.1.238:7447 --logging-namespace flatdisk/xiao
```

This creates files like:

`captures/flatdisk/xiao/20260528_143201-9f1a2b3c.rrd`

The Zenoh build runs Wi-Fi station mode only and publishes:

- `flatdisk/xiao/camera/jpeg`: binary `FDV1` header plus JPEG payload, targeted at 10 Hz.
- `flatdisk/xiao/imu`: binary `FDI1` IMU sample, targeted at 60 Hz.
- `flatdisk/xiao/motors/encoders`: JSON AS5600 motor encoder counts, targeted at 50 Hz.
- `flatdisk/xiao/status`: JSON counters and device state.
- `flatdisk/xiao/time_sync`: binary `FDSR` replies to `flatdisk/xiao/cmd/time_sync`.

It also subscribes to motor command topics:

- `flatdisk/xiao/cmd/motors/percent`: JSON `{"m1":0,"m2":0}` or text `0,0`, values constrained to `-100..100`.
- `flatdisk/xiao/cmd/motors/us`: JSON `{"m1_us":1500,"m2_us":1500}` or text `1500,1500`, values constrained to `1000..2000`.
- `flatdisk/xiao/cmd/motors/stop`: any payload returns both outputs to neutral.

The motor failsafe still returns both outputs to neutral if commands stop for 1 second. The companion can keep commands alive while streaming, for example:

```sh
.venv/bin/python scripts/zenoh_companion.py --motor-us 1500 1500
.venv/bin/python scripts/zenoh_companion.py --motor-percent 0 0
```

For a small desktop motor-control GUI with live camera preview:

```sh
.venv/bin/python scripts/motor_gui.py
```

## Visible Object Drive

`scripts/object_drive_zenoh.py` is the inner phrase-grounded visual-servo tool
used by the hardware LLM harness. Test it directly before a full outer-loop run.

Dry run with Florence detections and Rerun logging, but no motor commands:

```sh
cd sim
uv run --extra harness python ../scripts/object_drive_zenoh.py \
  --prompt "chair" \
  --duration 6 \
  --forward-power 18 \
  --rerun \
  --rerun-save
```

Armed visible-object drive:

```sh
cd sim
uv run --extra harness python ../scripts/object_drive_zenoh.py \
  --prompt "chair" \
  --duration 6 \
  --forward-power 18 \
  --arm \
  --rerun \
  --rerun-save
```

## Hardware LLM Harness

`flatdisk-sim-run-hardware-harness` runs the Qwen/Codex outer loop directly
against the physical robot over Zenoh. It reuses the same harness session and
bounded tools as the THOR research eval, but it does not launch THOR and does
not use hidden evaluator state. The policy sees only the live robot camera
frame, IMU yaw, bounded tool results, motion strips, and harness memory.

The command is conservative by default: motor-capable tools are blocked unless
`--arm` is present.

Install the harness-side local dependencies:

```sh
cd sim
uv sync --extra harness
```

Preflight one camera/IMU observation without asking the model to act:

```sh
uv run --extra harness flatdisk-sim-run-hardware-harness \
  --goal "go to the chair" \
  --preflight-only
```

Run the Qwen outer loop while still blocking motor commands. Point
`--qwen-endpoint` at the OpenAI-compatible Qwen server you are using:

```sh
uv run --extra harness flatdisk-sim-run-hardware-harness \
  --goal "go to the chair" \
  --qwen-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct \
  --max-steps 3 \
  --rerun
```

After the prompt, detector overlays, and proposed actions look sane, arm the
bounded tool executor:

```sh
uv run --extra harness flatdisk-sim-run-hardware-harness \
  --goal "go to the chair" \
  --qwen-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --qwen-model Qwen/Qwen3-VL-8B-Instruct \
  --object-drive-detector florence-mlx \
  --turn-heading-kp 24 \
  --min-turn-percent 8 \
  --max-steps 4 \
  --arm \
  --rerun
```

Artifacts are written under `sim/scratch/hardware_llm_harness/<timestamp>/`.
Inspect `hardware_harness_summary.json`, `hardware_harness_report.md`,
`policy/memory.jsonl`, `policy/prompts/`, saved camera frames, motion strips,
and optional `policy/hardware_harness.rrd`.

## ACT Training

The `scripts/train_act.py` script trains a small ACT-style policy from Rerun `.rrd`
recordings. It expects camera frames at `/camera/image` and motor labels at
`/commands/motor1_percent` and `/commands/motor2_percent`.

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-train.txt
.venv/bin/python scripts/train_act.py \
  --data /Users/bencaunt/Documents/flat-disk-robot-code/captures/vla-act-ctrlr \
  --temporal-ensembling \
  --epochs 10
```

To train the same model in stick-mixed action space, use
`--action-representation forward_steer`. This trains on
`forward=(motor1+motor2)/2` and `steer=(motor1-motor2)/2`, while live inference
and replay convert predictions back to motor1/motor2 percentages.

For another useful ablation, add `--normalize-actions`. This centers and scales
the model action dimensions using train-split statistics, saves those stats into
the checkpoint, and live inference/replay denormalize before publishing or
comparing motor commands.

To make the policy less brittle when its own previous commands differ from the
demonstration, add train-only past-action corruption such as
`--past-action-noise-std-percent 2.0 --past-action-dropout-prob 0.15`.

If camera/action alignment is suspect, train or replay with
`--action-time-offset-s`. Positive values pair an image at time `t` with a later
recorded command at `t + offset`. To sweep offsets for one recording:

```sh
.venv/bin/python scripts/sweep_act_action_offset.py \
  captures/vla-act-no-obstacle-fast-collect/20260529_173242-a5aefa8a.rrd \
  --checkpoint runs/act_vla_no_obstacle_fast_collect_norm_aug/best.pt \
  --history-source demo \
  --no-temporal-ensembling
```

Outputs are written under `runs/act_vla/` by default:

- `config.json`: training and inference configuration, including temporal ensembling.
- `metrics.json`: per-epoch train/validation loss and percent-output error.
- `best.pt` and `last.pt`: PyTorch checkpoints.

To run live ACT inference from the Zenoh camera stream:

```sh
zenohd --listen tcp/0.0.0.0:7447
.venv/bin/python scripts/act_zenoh_inference.py --checkpoint runs/act_vla/best.pt
```

By default this runs inference and publishes monitor data to
`flatdisk/xiao/act/status`, but it does not drive the robot. Add `--arm` to
publish model outputs to `flatdisk/xiao/cmd/motors/percent`:

```sh
.venv/bin/python scripts/act_zenoh_inference.py --checkpoint runs/act_vla/best.pt --arm
```

The live camera frame is rotated 180 degrees before inference to match the motor
GUI preview and the recorded ACT training data. Motor outputs are clamped to
`+-10%` by default; use `--max-abs-output` to change that limit.

To save a Rerun recording of live ACT inference while driving:

```sh
.venv/bin/python scripts/act_zenoh_inference.py \
  --checkpoint runs/act_vla_no_obstacle_fast_collect_norm_aug/best.pt \
  --no-temporal-ensembling \
  --arm \
  --rerun-save
```

The ACT inference recording logs `/act/camera/model_view`, raw and clamped motor
predictions, chunk-0 and future chunk motor traces, forward/steer, motor deltas,
video counters, and selected robot status values. With no explicit path,
`--rerun-save` writes a unique recording under `captures/act-inference/`.

To replay a recorded `.rrd` through a checkpoint and compare predicted motor
outputs against the recorded commands:

```sh
.venv/bin/python scripts/replay_act_log.py \
  captures/vla-act-no-obstacle-fast-collect/20260529_173242-a5aefa8a.rrd \
  --checkpoint runs/act_vla_no_obstacle_fast_collect/best.pt \
  --write-rerun
```

This writes a CSV, a PNG plot, an NPZ of predicted chunks, and optionally a
Rerun comparison log under `runs/act_replay/`. Use
`--history-source predicted` to mirror live inference, or
`--history-source demo` to feed recorded past commands back into the model and
isolate frame-to-action prediction quality.

## Visual Localization

A first-pass no-odometry hloc pipeline lives in `scripts/visual_localization.py`.
It can build an SfM map from a video and then localize live Zenoh camera frames
inside that map. Use the hloc Python environment for this script:

```sh
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HLOC_DEVICE=mps

../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py build-map \
  --video captures/room_scan.mp4 \
  --map-dir maps/room1 \
  --fps 2 \
  --max-frames 250 \
  --global-conf netvlad

../Hierarchical-Localization/.venv/bin/python scripts/visual_localization.py localize-zenoh \
  --map-dir maps/room1 \
  --namespace flatdisk/xiao \
  --connect tcp/127.0.0.1:7447 \
  --matcher-conf NN-superpoint \
  --top-k 10 \
  --max-rate 2
```

See `docs/visual-localization.md` for camera intrinsics, pose output, and
matcher tradeoffs.

For ESP32 camera calibration, collect checkerboard frames from Zenoh and write a
calibration JSON that the visual localizer can read:

```sh
../Hierarchical-Localization/.venv/bin/python scripts/checkerboard_calibration_logger.py capture-calibrate \
  --namespace flatdisk/xiao \
  --connect tcp/127.0.0.1:7447 \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 0.024 \
  --output-dir captures/esp32_calibration
```

Then pass `--query-camera-calibration captures/esp32_calibration/camera_calibration.json`
to `localize-zenoh`. See `docs/camera-calibration.md`.

If the serial monitor shows `AUTH_EXPIRE`, `AUTH_FAIL`, or `HANDSHAKE_TIMEOUT`, the ESP32 is failing before Zenoh. Check the `include/secrets.h` Wi-Fi credentials, bring the board closer to the AP, or use a stronger 2.4 GHz network before evaluating stream rates.

## Use

By default the firmware starts a Wi-Fi access point:

- SSID: `XIAO-CAM`
- Password: `seeedstudio`
- URL: `http://192.168.4.1`

Endpoints:

- `/capture.jpg` returns one JPEG frame.
- `/stream` returns an MJPEG stream.
- `/wifi/status` returns station/AP connection state and the current router IP, if connected.
- `/motors?m1=0&m2=0` sets ESC outputs on D1 and D2 from `-100` to `100`.
- `/motors/stop` returns both ESC outputs to neutral.
- `/imu` returns the latest BNO085 quaternion, acceleration, gyro, and linear acceleration data as JSON.

To connect to your router too, copy `include/secrets.example.h` to `include/secrets.h` and set `WIFI_SSID` and `WIFI_PASSWORD`. The board keeps the `XIAO-CAM` access point up, starts the HTTP server immediately, and retries the router connection in the background. The board prints its router URL in the serial monitor when the station connection succeeds.

## Hardware Notes

This is configured for `seeed_xiao_esp32s3` with the XIAO ESP32S3 Sense camera module. The camera connector pin map matches Seeed's XIAO ESP32S3 Sense documentation: XCLK GPIO10, SCCB GPIO40/GPIO39, data GPIO15/17/18/16/14/12/11/48, VSYNC GPIO38, HREF GPIO47, PCLK GPIO13.

The motor controller outputs are RC-servo PWM for AM32 bidirectional drive:

- D1 / GPIO2: motor 1 signal
- D2 / GPIO3: motor 2 signal
- 50 Hz PWM
- 1000 us: full reverse
- 1500 us: neutral
- 2000 us: full forward

The firmware outputs neutral during boot and returns both channels to neutral if no motor command is received for 1 second. Connect ESP32 ground to the ESC signal ground.

The motor encoders are analog AS5600 angle outputs read at 12-bit resolution. The firmware unwraps each 0..4095 sample stream with directional crossover detection, so `*_count` is signed encoder counts since boot and `*_rotations` increments or decrements on wrap crossings:

- A3 / GPIO4: motor 1 AS5600 analog output
- A10 / GPIO9: motor 2 AS5600 analog output
- 4096 counts per revolution
- The Zenoh encoder topic publishes `m1_count`, `m2_count`, `m1_raw`, `m2_raw`, raw min/max diagnostics, `m1_rotations`, `m2_rotations`, and direction/diagnostic fields.

The IMU is configured for a BNO085/GY-BNO080-BNO085 module over I2C using a direct SHTP reader. This avoids depending on one breakout vendor's Arduino wrapper behavior:

- SDA: D4 / GPIO5
- SCL: D5 / GPIO6
- INT: D9 / GPIO8
- RST: D8 / GPIO7
- I2C clock: 400 kHz after initialization
- Address: `0x4B`
- Reports enabled by the normal web firmware: rotation vector, accelerometer, gyro, and linear acceleration

Connect IMU `VCC` to 3V3 unless your exact breakout explicitly requires 5V. Connect all grounds together. Leave `PS0` and `PS1` at the breakout's default/floating state for I2C on this module; tying both low stopped I2C ACKs during bring-up.
