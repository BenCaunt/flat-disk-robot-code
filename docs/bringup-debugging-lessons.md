# Bring-Up Debugging Lessons

Notes from the April 26, 2026 XIAO ESP32S3 Sense camera, BNO085 IMU, ESC, Wi-Fi, and Zenoh bring-up session.

## Summary

The camera failure was ultimately hardware: the original ribbon cable/module path was bad, and the replacement OV2640 camera produced a valid image. The IMU failure was more subtle: basic I2C ACKs did not mean the BNO085 SH-2 protocol path was working, and the SparkFun wrapper hid enough behavior that direct SHTP debugging was needed. Zenoh performance was blocked by Wi-Fi quality until the antenna was improved.

The biggest general lesson is to prove one layer at a time: electrical presence, protocol behavior, sensor data, local firmware behavior, transport, then visualization/control.

## What The Human Got Wrong

- Debugged too many variables at once early: camera, IMU, ESC, Wi-Fi, web server, power, and firmware were all in play. Unplugging the IMU and ESC to isolate the camera was the right move.
- Suspected the web server was interfering with hardware pins. It later caused responsiveness issues because the server loop was synchronous, but it was not stealing camera or IMU GPIOs.
- Treated I2C ACK at `0x4b` as stronger evidence than it was. ACK only proved electrical presence; it did not prove BNO085 SH-2 startup, report enable, or usable sensor data.
- Left `PS0` / `PS1` floating on the BNO085 breakout. Mode pins sampled at reset are not something to leave ambiguous, even though this particular breakout did not behave exactly like the Adafruit/SparkFun references.
- Trusted the original camera module/ribbon for too long. The evidence already pointed to bad image data: valid sensor PID, valid JPEGs, color bars, then black/green/striped real images.
- Expected `include/secrets.h` to affect behavior without reflashing. It is compile-time input, so the firmware had to be rebuilt and uploaded.

This was not a lack-of-effort problem. The main miss was not reducing the system to known-good, single-variable tests soon enough.

## Good Calls

- Asking for a minimal serial debugger was the right pivot. It moved the loop away from browser symptoms and toward objective bus/pin/protocol evidence.
- Questioning whether the breakout differed from Adafruit's board was correct. The chip protocol is shared, but breakout straps, reset behavior, pullups, labels, and mode defaults are board-specific.
- Swapping in a known-good OV2640 camera was decisive.
- Adding the bigger antenna was decisive for Zenoh. It changed the problem from Wi-Fi authentication/handshake instability to sustained 10 Hz video and about 60 Hz IMU.

## Platform-Specific Issues

- The Seeed XIAO ESP32S3 Sense camera uses a board-specific camera pin map, not the common AI Thinker ESP32-CAM map.
- The original camera detected as `OV3660`; the replacement OV2640 confirmed the firmware path was basically sound.
- Native USB CDC upload on the ESP32-S3 was flaky with the esptool stub. `115200` plus `--no-stub` was needed for reliable upload.
- macOS serial device names disappeared/reappeared, and stale PlatformIO monitor processes sometimes held the port.
- This ESP32 Arduino core rejected 16-bit LEDC PWM; the firmware had to use a valid lower resolution.
- The synchronous HTTP server could accept TCP connections while application code blocked the handler loop.
- The SparkFun BNO08x wrapper was unreliable for this bring-up. Direct SHTP was needed.
- The BNO085 emitted a 276-byte SHTP advertisement packet after reset. A too-small 256-byte packet buffer prevented draining that packet and blocked report setup.
- ESP32 Wi-Fi symptoms looked like bad credentials, but the bigger antenna showed link quality was the real blocker.

## Hardware Lessons

- A valid camera sensor ID only proves the control bus works. It does not prove the parallel image data path or ribbon cable is healthy.
- Color bars are a useful camera split test: clean bars plus bad real image points toward sensor/module/ribbon/exposure rather than HTTP.
- I2C scan results are necessary but not sufficient. Follow with protocol-level reads and real data reports.
- Reset pins and mode straps must be treated carefully. A reset can re-sample mode pins and make a previously visible device disappear if straps are wrong or floating.
- When a clone breakout behaves strangely, reduce wiring to power, ground, SDA, and SCL before adding `INT`, `RST`, or mode straps.

## Process To Use Next Time

1. Bring up one subsystem at a time.
2. Start with minimal wiring and a minimal firmware path.
3. Prove power and idle line voltages.
4. Prove bus visibility with a scanner.
5. Prove protocol-level communication.
6. Prove real data.
7. Only then add web UI, transport, visualization, and control.
8. Keep a known-good hardware substitute nearby for cables, sensors, and modules.

## Outcome

- Camera: fixed by replacing the bad original ribbon/module path with the OV2640 camera.
- IMU: fixed by bypassing the SparkFun wrapper and using direct SHTP with a large enough packet buffer.
- Web server: kept as a useful fallback, but made less blocking.
- Zenoh: achieved target rates after improving Wi-Fi signal with a bigger antenna.
- Motor PWM: exposed through firmware with neutral/failsafe behavior preserved.
