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

The debug firmware does not start the `XIAO-CAM` access point or web server. It connects only as a Wi-Fi station using `include/secrets.h`, then accepts serial commands. The most useful IMU command is `direct`, which resets the BNO085, drains the SHTP advertisement packet, enables rotation-vector and accelerometer reports, then streams decoded packets. Other commands include `help`, `pins`, `scan`, `raw`, `dump`, `prod`, `getfeat`, `init`, `softreset`, and `reset`.

For the Zenoh streaming prototype, copy `include/local_zenoh.example.h` to `include/local_zenoh.h`, set the laptop IP in `ZENOH_CONNECT`, then upload:

```sh
~/.platformio/penv/bin/pio run -e zenoh_stream -t upload --upload-port /dev/cu.usbmodem2101
zenohd --listen tcp/0.0.0.0:7447
.venv/bin/python scripts/zenoh_companion.py --duration 60
```

The Zenoh build runs Wi-Fi station mode only and publishes:

- `flatdisk/xiao/camera/jpeg`: binary `FDV1` header plus JPEG payload, targeted at 10 Hz.
- `flatdisk/xiao/imu`: binary `FDI1` IMU sample, targeted at 60 Hz.
- `flatdisk/xiao/status`: JSON counters and device state.
- `flatdisk/xiao/time_sync`: binary `FDSR` replies to `flatdisk/xiao/cmd/time_sync`.

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

The IMU is configured for a BNO085/GY-BNO080-BNO085 module over I2C using a direct SHTP reader. This avoids depending on one breakout vendor's Arduino wrapper behavior:

- SDA: D4 / GPIO5
- SCL: D5 / GPIO6
- INT: D9 / GPIO8
- RST: D8 / GPIO7
- I2C clock: 400 kHz after initialization
- Address: `0x4B`
- Reports enabled by the normal web firmware: rotation vector and accelerometer

Connect IMU `VCC` to 3V3 unless your exact breakout explicitly requires 5V. Connect all grounds together. Leave `PS0` and `PS1` at the breakout's default/floating state for I2C on this module; tying both low stopped I2C ACKs during bring-up.
