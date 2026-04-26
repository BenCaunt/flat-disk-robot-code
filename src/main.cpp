#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>
#include <Wire.h>
#include <SparkFun_BNO08x_Arduino_Library.h>
#include "esp_camera.h"
#include "camera_pins.h"

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID ""
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD ""
#endif

namespace {
constexpr char kApSsid[] = "XIAO-CAM";
constexpr char kApPassword[] = "seeedstudio";
constexpr uint32_t kWifiConnectTimeoutMs = 15000;
constexpr uint32_t kWifiRetryIntervalMs = 30000;
constexpr uint8_t kMotor1Pin = D1;
constexpr uint8_t kMotor2Pin = D2;
constexpr uint8_t kMotor1PwmChannel = 4;
constexpr uint8_t kMotor2PwmChannel = 5;
constexpr uint32_t kEscPwmFrequencyHz = 50;
constexpr uint8_t kEscPwmResolutionBits = 14;
constexpr uint32_t kEscPwmPeriodUs = 1000000UL / kEscPwmFrequencyHz;
constexpr uint16_t kEscMinPulseUs = 1000;
constexpr uint16_t kEscNeutralPulseUs = 1500;
constexpr uint16_t kEscMaxPulseUs = 2000;
constexpr uint32_t kEscCommandTimeoutMs = 1000;
constexpr uint8_t kImuSdaPin = SDA;
constexpr uint8_t kImuSclPin = SCL;
constexpr uint8_t kImuPrimaryAddress = 0x4B;
constexpr uint8_t kImuSecondaryAddress = 0x4A;
constexpr uint32_t kImuInitI2cClockHz = 100000;
constexpr uint32_t kImuI2cClockHz = 400000;
constexpr uint16_t kImuReportIntervalMs = 20;
constexpr uint32_t kImuRetryIntervalMs = 5000;
constexpr uint16_t kImuWireTimeoutMs = 250;
constexpr uint16_t kDebugI2cScanTimeoutMs = 8;
constexpr uint16_t kImuBootDelayMs = 250;
constexpr uint16_t kImuResetRecoveryMs = 2500;
constexpr uint8_t kImuInitAttempts = 3;
constexpr int8_t kImuIntPin = D9;
constexpr int8_t kImuResetPin = D8;
constexpr uint8_t kShtpChannelCommand = 0;
constexpr uint8_t kShtpChannelExecutable = 1;
constexpr uint8_t kShtpDefaultControlChannel = 2;
constexpr uint8_t kShtpDefaultInputNormalChannel = 3;
constexpr uint8_t kShtpDefaultInputWakeChannel = 4;
constexpr uint8_t kShtpWireChunkSize = 32;
constexpr uint16_t kShtpMaxPacketBytes = 1024;
constexpr uint32_t kShtpDirectReportIntervalUs = 50000;
constexpr uint32_t kShtpDumpDurationMs = 3000;

WebServer server(80);
BNO08x imu;
HardwareSerial imuRvcSerial(1);
uint8_t shtpTxSeq[6] = {};
uint8_t shtpControlChannel = kShtpDefaultControlChannel;
uint8_t shtpInputNormalChannel = kShtpDefaultInputNormalChannel;
uint8_t shtpInputWakeChannel = kShtpDefaultInputWakeChannel;
int motor1Percent = 0;
int motor2Percent = 0;
uint32_t lastMotorCommandMs = 0;
uint32_t lastWifiConnectAttemptMs = 0;
bool imuReady = false;
uint8_t imuAddress = 0;
bool cameraColorbarEnabled = false;
bool imuReportsEnabled = false;
bool imuAutoInitAttempted = false;
bool wifiWasConnected = false;
bool wifiConnectTimedOut = false;
uint32_t lastImuInitAttemptMs = 0;
String imuInitMode = "";
bool cameraReady = false;
uint32_t cameraFrameOkCount = 0;
uint32_t cameraFrameFailCount = 0;
uint32_t lastCameraOkMs = 0;
uint32_t lastCameraFailMs = 0;
uint32_t lastCameraFrameBytes = 0;
uint16_t lastCameraFrameWidth = 0;
uint16_t lastCameraFrameHeight = 0;
String lastCameraError = "not initialized";
String imuLastError = "not initialized";

struct ImuState {
  float qi = 0.0f;
  float qj = 0.0f;
  float qk = 0.0f;
  float qr = 1.0f;
  float quatRadAccuracy = 0.0f;
  uint8_t quatAccuracy = 0;
  float accelX = 0.0f;
  float accelY = 0.0f;
  float accelZ = 0.0f;
  uint8_t accelAccuracy = 0;
  float gyroX = 0.0f;
  float gyroY = 0.0f;
  float gyroZ = 0.0f;
  uint8_t gyroAccuracy = 0;
  float linearAccelX = 0.0f;
  float linearAccelY = 0.0f;
  float linearAccelZ = 0.0f;
  uint8_t linearAccelAccuracy = 0;
  uint32_t updatedMs = 0;
};

ImuState imuState;

void handleCameraCaptureTest();
bool initImuDirect();
void updateImuDirect();

camera_config_t makeCameraConfig() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  return config;
}

void sendPlain(int code, const char *message) {
  server.send(code, "text/plain", message);
}

void prepareImuControlPins() {
  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, OUTPUT);
  digitalWrite(kImuResetPin, HIGH);
}

void pulseBnoReset() {
  prepareImuControlPins();
  digitalWrite(kImuResetPin, HIGH);
  delay(10);
  digitalWrite(kImuResetPin, LOW);
  delay(25);
  digitalWrite(kImuResetPin, HIGH);
  delay(kImuResetRecoveryMs);
}

String imuPinsJson() {
  prepareImuControlPins();
  String response = "{\"int_gpio\":";
  response += String(kImuIntPin);
  response += ",\"reset_gpio\":";
  response += String(kImuResetPin);
  response += ",\"int_level\":";
  response += String(digitalRead(kImuIntPin));
  response += ",\"reset_level\":";
  response += String(digitalRead(kImuResetPin));
  response += "}";
  return response;
}

String scanI2cBusJson() {
  String response;
  response.reserve(420);
  response += "{\"sda_gpio\":";
  response += String(kImuSdaPin);
  response += ",\"scl_gpio\":";
  response += String(kImuSclPin);
  response += ",\"int_gpio\":";
  response += String(kImuIntPin);
  response += ",\"reset_gpio\":";
  response += String(kImuResetPin);
  response += ",\"int_level\":";
  response += String(digitalRead(kImuIntPin));
  response += ",\"reset_level\":";
  response += String(digitalRead(kImuResetPin));
  response += ",\"addresses\":[";

  bool first = true;
  for (uint8_t address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission();
    if (error == 0) {
      if (!first) {
        response += ",";
      }
      response += "\"0x";
      if (address < 16) {
        response += "0";
      }
      response += String(address, HEX);
      response += "\"";
      first = false;
    }
    delay(1);
  }

  response += "]}";
  return response;
}

void logI2cScan() {
  const String scan = scanI2cBusJson();
  Serial.print("I2C scan: ");
  Serial.println(scan);
}

bool i2cAddressAcks(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

bool sendBnoSoftReset(uint8_t address) {
  static const uint8_t resetPacket[] = {5, 0, 1, 0, 1};
  Wire.beginTransmission(address);
  Wire.write(resetPacket, sizeof(resetPacket));
  return Wire.endTransmission() == 0;
}

bool enableImuReports() {
  const bool rotationOk = imu.enableRotationVector(kImuReportIntervalMs);
  Serial.printf("enableRotationVector(%u ms): %s\n",
                kImuReportIntervalMs,
                rotationOk ? "ok" : "failed");
  delay(10);
  const bool accelOk = imu.enableAccelerometer(kImuReportIntervalMs);
  Serial.printf("enableAccelerometer(%u ms): %s\n",
                kImuReportIntervalMs,
                accelOk ? "ok" : "failed");
  return rotationOk || accelOk;
}

bool tryBeginImuAt(uint8_t address, int8_t intPin, int8_t resetPin, const char *mode) {
  if (!i2cAddressAcks(address)) {
    return false;
  }

  Serial.printf("Trying BNO08x at 0x%02x using %s\n", address, mode);
  if (imu.begin(address, Wire, intPin, resetPin)) {
    imuAddress = address;
    imuInitMode = mode;
    return true;
  }
  return false;
}

bool initImu() {
  return initImuDirect();
}

void updateImu() {
  if (!imuReady) {
    if (!imuAutoInitAttempted && millis() - lastImuInitAttemptMs > kImuRetryIntervalMs) {
      imuAutoInitAttempted = true;
      imuReady = initImu();
    }
    return;
  }

  updateImuDirect();
}

uint32_t pulseUsToDuty(uint16_t pulseUs) {
  const uint32_t maxDuty = (1UL << kEscPwmResolutionBits) - 1;
  return (static_cast<uint32_t>(pulseUs) * maxDuty + (kEscPwmPeriodUs / 2)) / kEscPwmPeriodUs;
}

uint16_t motorPercentToPulseUs(int percent) {
  percent = constrain(percent, -100, 100);
  if (percent >= 0) {
    return kEscNeutralPulseUs + ((kEscMaxPulseUs - kEscNeutralPulseUs) * percent) / 100;
  }
  return kEscNeutralPulseUs + ((kEscNeutralPulseUs - kEscMinPulseUs) * percent) / 100;
}

void writeEscPulse(uint8_t channel, uint16_t pulseUs) {
  ledcWrite(channel, pulseUsToDuty(pulseUs));
}

void setMotors(int motor1, int motor2) {
  motor1Percent = constrain(motor1, -100, 100);
  motor2Percent = constrain(motor2, -100, 100);
  writeEscPulse(kMotor1PwmChannel, motorPercentToPulseUs(motor1Percent));
  writeEscPulse(kMotor2PwmChannel, motorPercentToPulseUs(motor2Percent));
  lastMotorCommandMs = millis();
}

void stopMotors() {
  setMotors(0, 0);
}

void initMotorOutputs() {
  ledcSetup(kMotor1PwmChannel, kEscPwmFrequencyHz, kEscPwmResolutionBits);
  ledcSetup(kMotor2PwmChannel, kEscPwmFrequencyHz, kEscPwmResolutionBits);
  ledcAttachPin(kMotor1Pin, kMotor1PwmChannel);
  ledcAttachPin(kMotor2Pin, kMotor2PwmChannel);
  stopMotors();
  Serial.printf("ESC PWM ready: motor1 D1/GPIO%u, motor2 D2/GPIO%u, neutral %uus\n",
                kMotor1Pin,
                kMotor2Pin,
                kEscNeutralPulseUs);
}

void updateMotorFailsafe() {
  if ((motor1Percent != 0 || motor2Percent != 0) &&
      millis() - lastMotorCommandMs > kEscCommandTimeoutMs) {
    stopMotors();
    Serial.println("Motor command timed out; outputs returned to neutral.");
  }
}

bool initCamera() {
  cameraReady = false;
  camera_config_t config = makeCameraConfig();
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    lastCameraError = String("init failed 0x") + String(err, HEX);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    Serial.println("Camera sensor handle is null");
    lastCameraError = "sensor handle is null";
    return false;
  }

  Serial.printf("Camera ready. Sensor PID: 0x%04x\n", sensor->id.PID);
  if (sensor->id.PID == OV3660_PID) {
    sensor->set_vflip(sensor, 1);
    sensor->set_brightness(sensor, 1);
    sensor->set_saturation(sensor, -2);
  } else {
    sensor->set_brightness(sensor, 0);
    sensor->set_saturation(sensor, 0);
  }
  sensor->set_framesize(sensor, FRAMESIZE_QVGA);
  sensor->set_contrast(sensor, 0);
  sensor->set_exposure_ctrl(sensor, 1);
  sensor->set_gain_ctrl(sensor, 1);
  sensor->set_whitebal(sensor, 1);
  cameraReady = true;
  lastCameraError = "";
  return true;
}

void applyCameraDefaults(sensor_t *sensor) {
  if (sensor == nullptr) {
    return;
  }

  if (sensor->id.PID == OV3660_PID) {
    sensor->set_vflip(sensor, 1);
    sensor->set_brightness(sensor, 1);
    sensor->set_saturation(sensor, -2);
  } else {
    sensor->set_brightness(sensor, 0);
    sensor->set_saturation(sensor, 0);
  }

  sensor->set_framesize(sensor, FRAMESIZE_QVGA);
  sensor->set_contrast(sensor, 0);
  sensor->set_exposure_ctrl(sensor, 1);
  sensor->set_gain_ctrl(sensor, 1);
  sensor->set_whitebal(sensor, 1);
}

void updateCameraFrameOk(camera_fb_t *fb) {
  if (fb == nullptr) {
    return;
  }

  ++cameraFrameOkCount;
  lastCameraOkMs = millis();
  lastCameraFrameBytes = fb->len;
  lastCameraFrameWidth = fb->width;
  lastCameraFrameHeight = fb->height;
  lastCameraError = "";
}

void updateCameraFrameFail(const char *error) {
  ++cameraFrameFailCount;
  lastCameraFailMs = millis();
  lastCameraError = error;
  Serial.println(error);
}

String cameraStatusJson() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    String response = "{\"ready\":false";
    response += ",\"error\":\"";
    response += lastCameraError;
    response += "\"";
    response += ",\"psram\":";
    response += psramFound() ? "true" : "false";
    response += ",\"ok_frames\":";
    response += String(cameraFrameOkCount);
    response += ",\"failed_frames\":";
    response += String(cameraFrameFailCount);
    response += "}";
    return response;
  }

  const camera_status_t &status = sensor->status;
  const resolution_info_t &res = resolution[status.framesize];
  String response;
  response.reserve(640);
  response += "{\"ready\":";
  response += cameraReady ? "true" : "false";
  response += ",\"pid\":\"0x";
  response += String(sensor->id.PID, HEX);
  response += "\"";
  response += ",\"slave_address\":\"0x";
  response += String(sensor->slv_addr, HEX);
  response += "\"";
  response += ",\"framesize\":";
  response += String(status.framesize);
  response += ",\"width\":";
  response += String(res.width);
  response += ",\"height\":";
  response += String(res.height);
  response += ",\"quality\":";
  response += String(status.quality);
  response += ",\"brightness\":";
  response += String(status.brightness);
  response += ",\"contrast\":";
  response += String(status.contrast);
  response += ",\"saturation\":";
  response += String(status.saturation);
  response += ",\"aec\":";
  response += String(status.aec);
  response += ",\"aec2\":";
  response += String(status.aec2);
  response += ",\"ae_level\":";
  response += String(status.ae_level);
  response += ",\"aec_value\":";
  response += String(status.aec_value);
  response += ",\"agc\":";
  response += String(status.agc);
  response += ",\"agc_gain\":";
  response += String(status.agc_gain);
  response += ",\"gainceiling\":";
  response += String(status.gainceiling);
  response += ",\"awb\":";
  response += String(status.awb);
  response += ",\"awb_gain\":";
  response += String(status.awb_gain);
  response += ",\"wb_mode\":";
  response += String(status.wb_mode);
  response += ",\"raw_gma\":";
  response += String(status.raw_gma);
  response += ",\"lenc\":";
  response += String(status.lenc);
  response += ",\"bpc\":";
  response += String(status.bpc);
  response += ",\"wpc\":";
  response += String(status.wpc);
  response += ",\"colorbar\":";
  response += cameraColorbarEnabled ? "true" : "false";
  response += ",\"psram\":";
  response += psramFound() ? "true" : "false";
  response += ",\"ok_frames\":";
  response += String(cameraFrameOkCount);
  response += ",\"failed_frames\":";
  response += String(cameraFrameFailCount);
  response += ",\"last_frame_bytes\":";
  response += String(lastCameraFrameBytes);
  response += ",\"last_frame_width\":";
  response += String(lastCameraFrameWidth);
  response += ",\"last_frame_height\":";
  response += String(lastCameraFrameHeight);
  response += ",\"last_ok_age_ms\":";
  response += lastCameraOkMs == 0 ? String("null") : String(millis() - lastCameraOkMs);
  response += ",\"last_fail_age_ms\":";
  response += lastCameraFailMs == 0 ? String("null") : String(millis() - lastCameraFailMs);
  response += ",\"error\":\"";
  response += lastCameraError;
  response += "\"";
  response += "}";
  return response;
}

void startWifi() {
  WiFi.mode(WIFI_AP_STA);
  WiFi.setSleep(false);
  WiFi.softAP(kApSsid, kApPassword);
  Serial.printf("Access point started: %s / %s\n", kApSsid, kApPassword);
  Serial.print("AP URL: http://");
  Serial.println(WiFi.softAPIP());

  if (strlen(WIFI_SSID) > 0) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    lastWifiConnectAttemptMs = millis();
    wifiConnectTimedOut = false;
    Serial.printf("Connecting to Wi-Fi SSID in background: %s\n", WIFI_SSID);
  }
}

void updateWifi() {
  if (strlen(WIFI_SSID) == 0) {
    return;
  }

  const wl_status_t status = WiFi.status();
  if (status == WL_CONNECTED) {
    if (!wifiWasConnected) {
      wifiWasConnected = true;
      wifiConnectTimedOut = false;
      Serial.print("Router URL: http://");
      Serial.println(WiFi.localIP());
    }
    return;
  }

  if (wifiWasConnected) {
    wifiWasConnected = false;
    Serial.printf("Wi-Fi disconnected, status=%d\n", status);
  }

  const uint32_t ageMs = millis() - lastWifiConnectAttemptMs;
  if (!wifiConnectTimedOut && ageMs > kWifiConnectTimeoutMs) {
    wifiConnectTimedOut = true;
    Serial.println("Wi-Fi connection timed out; AP remains available and STA will retry.");
  }

  if (ageMs > kWifiRetryIntervalMs) {
    Serial.printf("Retrying Wi-Fi SSID: %s\n", WIFI_SSID);
    WiFi.disconnect(false);
    delay(10);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    lastWifiConnectAttemptMs = millis();
    wifiConnectTimedOut = false;
  }
}

void handleRoot() {
  const String html =
      "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
      "<title>XIAO ESP32S3 Camera</title>"
      "<style>body{font-family:sans-serif;margin:24px;max-width:760px}img{max-width:100%;height:auto}"
      "a{display:inline-block;margin:8px 12px 16px 0}label{display:block;margin:10px 0}"
      "input{width:100%}button{margin:8px 8px 8px 0;padding:8px 12px}</style></head><body>"
      "<h1>XIAO ESP32S3 Camera</h1>"
      "<a href='/capture.jpg'>Capture JPEG</a><a href='/stream'>MJPEG Stream</a>"
      "<a href='/camera/status'>Camera Status</a><a href='/camera/colorbar?on=1'>Color Bars On</a>"
      "<a href='/camera/colorbar?on=0'>Color Bars Off</a><a href='/camera/reset'>Camera Reset</a>"
      "<a href='/camera/reinit'>Camera Reinit</a><a href='/camera/capture-test'>Capture Test</a>"
      "<a href='/camera/bright-test'>Bright Test</a><a href='/camera/manual-exposure-test'>Manual Exposure Test</a>"
      "<a href='/wifi/status'>WiFi Status</a><a href='/i2c/scan'>I2C Scan</a><a href='/imu/pins'>IMU Pins</a>"
      "<a href='/imu/reset'>IMU Reset</a><a href='/imu/reinit'>IMU Reinit</a>"
      "<h2>Motors</h2>"
      "<label>D1 <input id='m1' type='range' min='-100' max='100' value='0'></label>"
      "<label>D2 <input id='m2' type='range' min='-100' max='100' value='0'></label>"
      "<button onclick='sendMotors()'>Send</button><button onclick='stopMotors()'>Stop</button>"
      "<h2>IMU</h2><pre id='imu'>Waiting for IMU data...</pre>"
      "<p><img src='/capture.jpg' alt='camera capture'></p>"
      "<script>"
      "let motorsActive=false;"
      "function sendMotors(){motorsActive=true;fetch(`/motors?m1=${m1.value}&m2=${m2.value}`)}"
      "function stopMotors(){motorsActive=false;m1.value=0;m2.value=0;fetch('/motors/stop')}"
      "setInterval(()=>{if(motorsActive)sendMotors()},250)"
      "setInterval(()=>fetch('/imu').then(r=>r.json()).then(d=>document.getElementById('imu').textContent=JSON.stringify(d,null,2)),500)"
      "</script>"
      "</body></html>";
  server.send(200, "text/html", html);
}

void handleMotors() {
  const int motor1 = server.hasArg("m1") ? server.arg("m1").toInt() : motor1Percent;
  const int motor2 = server.hasArg("m2") ? server.arg("m2").toInt() : motor2Percent;
  setMotors(motor1, motor2);

  const String response = String("{\"m1\":") + motor1Percent + ",\"m2\":" + motor2Percent +
                          ",\"m1_us\":" + motorPercentToPulseUs(motor1Percent) +
                          ",\"m2_us\":" + motorPercentToPulseUs(motor2Percent) + "}";
  server.send(200, "application/json", response);
}

void handleMotorStop() {
  stopMotors();
  server.send(200, "application/json", "{\"m1\":0,\"m2\":0}");
}

void handleWifiStatus() {
  String response;
  response.reserve(320);
  response += "{\"sta_status\":";
  response += String(static_cast<int>(WiFi.status()));
  response += ",\"sta_connected\":";
  response += WiFi.status() == WL_CONNECTED ? "true" : "false";
  response += ",\"sta_ip\":\"";
  response += WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : "";
  response += "\",\"ap_ip\":\"";
  response += WiFi.softAPIP().toString();
  response += "\",\"last_connect_attempt_age_ms\":";
  response += String(millis() - lastWifiConnectAttemptMs);
  response += ",\"connect_timed_out\":";
  response += wifiConnectTimedOut ? "true" : "false";
  if (WiFi.status() == WL_CONNECTED) {
    response += ",\"rssi\":";
    response += String(WiFi.RSSI());
  }
  response += "}";
  server.send(200, "application/json", response);
}

void handleImu() {
  if (!imuReady) {
    String response = "{\"ready\":false";
    response += ",\"error\":\"";
    response += imuLastError;
    response += "\"";
    response += ",\"last_init_attempt_age_ms\":";
    response += String(millis() - lastImuInitAttemptMs);
    response += ",\"pins\":";
    response += imuPinsJson();
    response += "}";
    server.send(200, "application/json", response);
    return;
  }

  const int intLevel = digitalRead(kImuIntPin);
  String response;
  response.reserve(640);
  response += "{\"ready\":true";
  response += ",\"address\":\"0x";
  if (imuAddress < 16) {
    response += "0";
  }
  response += String(imuAddress, HEX);
  response += "\"";
  response += ",\"init_mode\":\"";
  response += imuInitMode;
  response += "\"";
  response += ",\"error\":\"";
  response += imuLastError;
  response += "\"";
  response += ",\"reports_enabled\":";
  response += imuReportsEnabled ? "true" : "false";
  response += ",\"int_level\":";
  response += String(intLevel);
  response += ",\"age_ms\":";
  response += String(millis() - imuState.updatedMs);
  response += ",\"quat\":{\"i\":";
  response += String(imuState.qi, 6);
  response += ",\"j\":";
  response += String(imuState.qj, 6);
  response += ",\"k\":";
  response += String(imuState.qk, 6);
  response += ",\"real\":";
  response += String(imuState.qr, 6);
  response += ",\"rad_accuracy\":";
  response += String(imuState.quatRadAccuracy, 6);
  response += ",\"accuracy\":";
  response += String(imuState.quatAccuracy);
  response += "},\"accel_mps2\":{\"x\":";
  response += String(imuState.accelX, 6);
  response += ",\"y\":";
  response += String(imuState.accelY, 6);
  response += ",\"z\":";
  response += String(imuState.accelZ, 6);
  response += ",\"accuracy\":";
  response += String(imuState.accelAccuracy);
  response += "},\"gyro_radps\":{\"x\":";
  response += String(imuState.gyroX, 6);
  response += ",\"y\":";
  response += String(imuState.gyroY, 6);
  response += ",\"z\":";
  response += String(imuState.gyroZ, 6);
  response += ",\"accuracy\":";
  response += String(imuState.gyroAccuracy);
  response += "},\"linear_accel_mps2\":{\"x\":";
  response += String(imuState.linearAccelX, 6);
  response += ",\"y\":";
  response += String(imuState.linearAccelY, 6);
  response += ",\"z\":";
  response += String(imuState.linearAccelZ, 6);
  response += ",\"accuracy\":";
  response += String(imuState.linearAccelAccuracy);
  response += "}}";

  server.send(200, "application/json", response);
}

void handleI2cScan() {
  server.send(200, "application/json", scanI2cBusJson());
}

void handleImuPins() {
  String response = imuPinsJson();
  server.send(200, "application/json", response);
}

void handleImuReset() {
  pulseBnoReset();
  imuReady = false;
  imuAddress = 0;
  imuReportsEnabled = false;
  imuInitMode = "";

  String response = "{\"reset_pulsed\":true";
  response += ",\"pins\":";
  response += imuPinsJson();
  response += ",\"i2c\":";
  response += scanI2cBusJson();
  response += "}";
  server.send(200, "application/json", response);
}

void handleImuReinit() {
  imuAutoInitAttempted = true;
  imuReady = initImu();
  handleImu();
}

void handleCameraStatus() {
  server.send(200, "application/json", cameraStatusJson());
}

void handleCameraColorbar() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    server.send(503, "application/json", "{\"ready\":false}");
    return;
  }

  cameraColorbarEnabled = server.hasArg("on") ? server.arg("on").toInt() != 0 : !cameraColorbarEnabled;
  sensor->set_colorbar(sensor, cameraColorbarEnabled ? 1 : 0);

  String response = String("{\"ready\":true,\"colorbar\":") +
                    (cameraColorbarEnabled ? "true" : "false") + "}";
  server.send(200, "application/json", response);
}

void handleCameraReset() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    server.send(503, "application/json", cameraStatusJson());
    return;
  }

  const int result = sensor->reset(sensor);
  delay(200);
  applyCameraDefaults(sensor);
  cameraColorbarEnabled = false;

  String response = String("{\"ready\":true,\"reset_result\":") + result + "}";
  server.send(200, "application/json", response);
}

void handleCameraFramesize() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    server.send(503, "application/json", cameraStatusJson());
    return;
  }

  if (!server.hasArg("size")) {
    server.send(400, "application/json", "{\"error\":\"missing size\"}");
    return;
  }

  const int requested = server.arg("size").toInt();
  if (requested < 0 || requested >= FRAMESIZE_INVALID) {
    server.send(400, "application/json", "{\"error\":\"invalid size\"}");
    return;
  }

  const int result = sensor->set_framesize(sensor, static_cast<framesize_t>(requested));
  String response = String("{\"ready\":true,\"framesize\":") + requested +
                    ",\"result\":" + result + "}";
  server.send(200, "application/json", response);
}

void handleCameraTune() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    server.send(503, "application/json", cameraStatusJson());
    return;
  }

  if (server.hasArg("quality")) {
    sensor->set_quality(sensor, constrain(server.arg("quality").toInt(), 4, 63));
  }
  if (server.hasArg("brightness")) {
    sensor->set_brightness(sensor, constrain(server.arg("brightness").toInt(), -2, 2));
  }
  if (server.hasArg("contrast")) {
    sensor->set_contrast(sensor, constrain(server.arg("contrast").toInt(), -2, 2));
  }
  if (server.hasArg("saturation")) {
    sensor->set_saturation(sensor, constrain(server.arg("saturation").toInt(), -2, 2));
  }
  if (server.hasArg("aec")) {
    sensor->set_exposure_ctrl(sensor, constrain(server.arg("aec").toInt(), 0, 1));
  }
  if (server.hasArg("aec2")) {
    sensor->set_aec2(sensor, constrain(server.arg("aec2").toInt(), 0, 1));
  }
  if (server.hasArg("ae_level")) {
    sensor->set_ae_level(sensor, constrain(server.arg("ae_level").toInt(), -2, 2));
  }
  if (server.hasArg("aec_value")) {
    sensor->set_aec_value(sensor, constrain(server.arg("aec_value").toInt(), 0, 1200));
  }
  if (server.hasArg("agc")) {
    sensor->set_gain_ctrl(sensor, constrain(server.arg("agc").toInt(), 0, 1));
  }
  if (server.hasArg("agc_gain")) {
    sensor->set_agc_gain(sensor, constrain(server.arg("agc_gain").toInt(), 0, 30));
  }
  if (server.hasArg("gainceiling")) {
    sensor->set_gainceiling(sensor, static_cast<gainceiling_t>(constrain(server.arg("gainceiling").toInt(), 0, 6)));
  }
  if (server.hasArg("awb")) {
    sensor->set_whitebal(sensor, constrain(server.arg("awb").toInt(), 0, 1));
  }
  if (server.hasArg("awb_gain")) {
    sensor->set_awb_gain(sensor, constrain(server.arg("awb_gain").toInt(), 0, 1));
  }
  if (server.hasArg("wb_mode")) {
    sensor->set_wb_mode(sensor, constrain(server.arg("wb_mode").toInt(), 0, 4));
  }
  if (server.hasArg("raw_gma")) {
    sensor->set_raw_gma(sensor, constrain(server.arg("raw_gma").toInt(), 0, 1));
  }
  if (server.hasArg("lenc")) {
    sensor->set_lenc(sensor, constrain(server.arg("lenc").toInt(), 0, 1));
  }
  if (server.hasArg("bpc")) {
    sensor->set_bpc(sensor, constrain(server.arg("bpc").toInt(), 0, 1));
  }
  if (server.hasArg("wpc")) {
    sensor->set_wpc(sensor, constrain(server.arg("wpc").toInt(), 0, 1));
  }

  server.send(200, "application/json", cameraStatusJson());
}

void handleCameraBrightTest() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    server.send(503, "application/json", cameraStatusJson());
    return;
  }

  cameraColorbarEnabled = false;
  sensor->set_colorbar(sensor, 0);
  sensor->set_quality(sensor, 8);
  sensor->set_brightness(sensor, 2);
  sensor->set_contrast(sensor, 2);
  sensor->set_saturation(sensor, 0);
  sensor->set_exposure_ctrl(sensor, 1);
  sensor->set_aec2(sensor, 1);
  sensor->set_ae_level(sensor, 2);
  sensor->set_gain_ctrl(sensor, 1);
  sensor->set_gainceiling(sensor, GAINCEILING_128X);
  sensor->set_whitebal(sensor, 1);
  sensor->set_awb_gain(sensor, 1);
  sensor->set_raw_gma(sensor, 1);
  sensor->set_lenc(sensor, 1);
  sensor->set_bpc(sensor, 1);
  sensor->set_wpc(sensor, 1);
  delay(1200);
  handleCameraCaptureTest();
}

void handleCameraManualExposureTest() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (sensor == nullptr) {
    server.send(503, "application/json", cameraStatusJson());
    return;
  }

  cameraColorbarEnabled = false;
  sensor->set_colorbar(sensor, 0);
  sensor->set_quality(sensor, 8);
  sensor->set_brightness(sensor, 2);
  sensor->set_contrast(sensor, 2);
  sensor->set_saturation(sensor, 0);
  sensor->set_exposure_ctrl(sensor, 0);
  sensor->set_gain_ctrl(sensor, 0);
  sensor->set_aec_value(sensor, 1200);
  sensor->set_agc_gain(sensor, 30);
  sensor->set_gainceiling(sensor, GAINCEILING_128X);
  sensor->set_whitebal(sensor, 1);
  sensor->set_awb_gain(sensor, 1);
  sensor->set_raw_gma(sensor, 1);
  sensor->set_lenc(sensor, 1);
  sensor->set_bpc(sensor, 1);
  sensor->set_wpc(sensor, 1);
  delay(300);
  handleCameraCaptureTest();
}

void handleCameraReinit() {
  esp_camera_deinit();
  cameraReady = false;
  cameraColorbarEnabled = false;
  delay(150);
  cameraReady = initCamera();
  server.send(cameraReady ? 200 : 503, "application/json", cameraStatusJson());
}

void handleCameraCaptureTest() {
  if (!cameraReady) {
    server.send(503, "application/json", cameraStatusJson());
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (fb == nullptr) {
    updateCameraFrameFail("Camera capture test failed");
    server.send(503, "application/json", cameraStatusJson());
    return;
  }

  updateCameraFrameOk(fb);
  String response = "{\"ok\":true";
  response += ",\"width\":";
  response += String(fb->width);
  response += ",\"height\":";
  response += String(fb->height);
  response += ",\"bytes\":";
  response += String(fb->len);
  response += ",\"format\":";
  response += String(fb->format);
  response += ",\"status\":";
  response += cameraStatusJson();
  response += "}";
  esp_camera_fb_return(fb);
  server.send(200, "application/json", response);
}

void handleCapture() {
  if (!cameraReady) {
    server.send(503, "text/plain", "Camera is not ready");
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (fb == nullptr) {
    updateCameraFrameFail("Camera capture failed");
    sendPlain(503, "Camera capture failed");
    return;
  }
  updateCameraFrameOk(fb);

  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: image/jpeg");
  client.println("Cache-Control: no-store");
  client.printf("Content-Length: %u\r\n", static_cast<unsigned int>(fb->len));
  client.println("Connection: close");
  client.println();
  client.write(fb->buf, fb->len);

  Serial.printf("Captured %ux%u JPEG, %u bytes\n",
                static_cast<unsigned int>(fb->width),
                static_cast<unsigned int>(fb->height),
                static_cast<unsigned int>(fb->len));
  esp_camera_fb_return(fb);
}

void handleStream() {
  if (!cameraReady) {
    server.send(503, "text/plain", "Camera is not ready");
    return;
  }

  WiFiClient client = server.client();
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  client.println("Cache-Control: no-store");
  client.println("Connection: close");
  client.println();

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb == nullptr) {
      updateCameraFrameFail("Stream capture failed");
      break;
    }
    updateCameraFrameOk(fb);

    client.println("--frame");
    client.println("Content-Type: image/jpeg");
    client.printf("Content-Length: %u\r\n\r\n", static_cast<unsigned int>(fb->len));
    client.write(fb->buf, fb->len);
    client.println();

    esp_camera_fb_return(fb);
    updateImu();
    updateMotorFailsafe();
    delay(30);
  }
}

void startServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/motors", HTTP_GET, handleMotors);
  server.on("/motors/stop", HTTP_GET, handleMotorStop);
  server.on("/wifi/status", HTTP_GET, handleWifiStatus);
  server.on("/imu", HTTP_GET, handleImu);
  server.on("/imu/pins", HTTP_GET, handleImuPins);
  server.on("/imu/reset", HTTP_GET, handleImuReset);
  server.on("/imu/reinit", HTTP_GET, handleImuReinit);
  server.on("/i2c/scan", HTTP_GET, handleI2cScan);
  server.on("/camera/status", HTTP_GET, handleCameraStatus);
  server.on("/camera/colorbar", HTTP_GET, handleCameraColorbar);
  server.on("/camera/reset", HTTP_GET, handleCameraReset);
  server.on("/camera/reinit", HTTP_GET, handleCameraReinit);
  server.on("/camera/capture-test", HTTP_GET, handleCameraCaptureTest);
  server.on("/camera/tune", HTTP_GET, handleCameraTune);
  server.on("/camera/bright-test", HTTP_GET, handleCameraBrightTest);
  server.on("/camera/manual-exposure-test", HTTP_GET, handleCameraManualExposureTest);
  server.on("/camera/framesize", HTTP_GET, handleCameraFramesize);
  server.on("/capture.jpg", HTTP_GET, handleCapture);
  server.on("/stream", HTTP_GET, handleStream);
  server.onNotFound([]() { sendPlain(404, "Not found"); });
  server.begin();
  Serial.println("HTTP camera server started");
}

uint8_t debugPrintI2cScan() {
  Wire.end();
  delay(10);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kDebugI2cScanTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);

  Serial.print("I2C scan on SDA D4/GPIO");
  Serial.print(kImuSdaPin);
  Serial.print(", SCL D5/GPIO");
  Serial.print(kImuSclPin);
  Serial.print(":");

  uint8_t count = 0;
  for (uint8_t address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission();
    if (error == 0) {
      Serial.print(" 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.print(address, HEX);
      ++count;
    }
    delay(1);
  }

  if (count == 0) {
    Serial.print(" none");
  }
  Serial.println();
  return count;
}

uint8_t debugPrintI2cScanAtClock(uint32_t clockHz) {
  Wire.end();
  delay(10);
  pinMode(kImuSdaPin, INPUT_PULLUP);
  pinMode(kImuSclPin, INPUT_PULLUP);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kDebugI2cScanTimeoutMs);
  Wire.setClock(clockHz);

  Serial.printf("I2C scan at %lu Hz on SDA D4/GPIO%u, SCL D5/GPIO%u:",
                static_cast<unsigned long>(clockHz),
                kImuSdaPin,
                kImuSclPin);

  uint8_t count = 0;
  for (uint8_t address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission();
    if (error == 0) {
      Serial.print(" 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.print(address, HEX);
      ++count;
    }
    delay(1);
  }

  if (count == 0) {
    Serial.print(" none");
  }
  Serial.println();
  return count;
}

void debugScanI2cClockRates() {
  static const uint32_t rates[] = {10000, 50000, 100000, 400000};
  Serial.println("Scanning nominal I2C pins at multiple clock rates:");
  for (const uint32_t rate : rates) {
    debugPrintI2cScanAtClock(rate);
  }
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);
}

uint8_t debugScanI2cPins(uint8_t sda, uint8_t scl, bool printAddresses) {
  Wire.end();
  delay(10);
  Wire.begin(sda, scl);
  Wire.setTimeOut(kDebugI2cScanTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);

  uint8_t count = 0;
  for (uint8_t address = 1; address < 127; ++address) {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission();
    if (error == 0) {
      if (printAddresses) {
        Serial.print(" 0x");
        if (address < 16) {
          Serial.print("0");
        }
        Serial.print(address, HEX);
      }
      ++count;
    }
    delay(1);
  }
  return count;
}

void debugPrintAllPinLevels() {
  struct NamedPin {
    const char *name;
    uint8_t pin;
  };

  static const NamedPin pins[] = {
      {"D0", D0}, {"D1", D1}, {"D2", D2}, {"D3", D3},
      {"D4", D4}, {"D5", D5}, {"D6", D6}, {"D7", D7},
      {"D8", D8}, {"D9", D9}, {"D10", D10}};

  Serial.println("Digital pin levels with INPUT_PULLUP:");
  for (const NamedPin &entry : pins) {
    pinMode(entry.pin, INPUT_PULLUP);
    delay(1);
    Serial.printf("  %s/GPIO%u=%d\n", entry.name, entry.pin, digitalRead(entry.pin));
  }
}

void debugScanCandidatePinPairs() {
  struct Pair {
    const char *label;
    uint8_t sda;
    uint8_t scl;
  };

  static const Pair pairs[] = {
      {"D4 SDA / D5 SCL", D4, D5},
      {"D5 SDA / D4 SCL", D5, D4},
      {"D3 SDA / D4 SCL", D3, D4},
      {"D4 SDA / D3 SCL", D4, D3},
      {"D0 SDA / D1 SCL", D0, D1},
      {"D1 SDA / D0 SCL", D1, D0},
      {"D2 SDA / D3 SCL", D2, D3},
      {"D3 SDA / D2 SCL", D3, D2},
      {"D6 SDA / D7 SCL", D6, D7},
      {"D7 SDA / D6 SCL", D7, D6},
  };

  Serial.println("Scanning candidate I2C pin pairs:");
  for (const Pair &pair : pairs) {
    Serial.printf("  %-18s GPIO%u/GPIO%u:", pair.label, pair.sda, pair.scl);
    const uint8_t count = debugScanI2cPins(pair.sda, pair.scl, true);
    if (count == 0) {
      Serial.print(" none");
    }
    Serial.println();
  }

  Wire.end();
  delay(10);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);
}

void debugScanAllPinPairs() {
  struct NamedPin {
    const char *name;
    uint8_t pin;
  };

  static const NamedPin pins[] = {
      {"D0", D0}, {"D1", D1}, {"D2", D2}, {"D3", D3},
      {"D4", D4}, {"D5", D5}, {"D6", D6}, {"D7", D7}, {"D10", D10}};

  uint8_t respondingPairs = 0;
  Serial.println("Exhaustive I2C scan over exposed non-control D pins; printing only pairs with ACKs:");
  for (const NamedPin &sda : pins) {
    for (const NamedPin &scl : pins) {
      if (sda.pin == scl.pin) {
        continue;
      }

      const uint8_t count = debugScanI2cPins(sda.pin, scl.pin, false);
      if (count == 0) {
        continue;
      }

      ++respondingPairs;
      Serial.printf("  %s SDA / %s SCL GPIO%u/GPIO%u:",
                    sda.name,
                    scl.name,
                    sda.pin,
                    scl.pin);
      debugScanI2cPins(sda.pin, scl.pin, true);
      Serial.println();
    }
  }

  if (respondingPairs == 0) {
    Serial.println("  no responding devices found on any exposed D-pin pair");
  }

  Wire.end();
  delay(10);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);
}

void debugPrintPins() {
  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, INPUT_PULLUP);
  Serial.printf("Pins: INT D9/GPIO%u=%d, RST D8/GPIO%u=%d\n",
                kImuIntPin,
                digitalRead(kImuIntPin),
                kImuResetPin,
                digitalRead(kImuResetPin));
}

void debugControlPinDiag() {
  Serial.println("INT/RST GPIO electrical check:");
  pinMode(kImuIntPin, INPUT);
  pinMode(kImuResetPin, INPUT);
  delay(5);
  Serial.printf("  floating input: INT=%d RST=%d\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));

  pinMode(kImuIntPin, INPUT_PULLDOWN);
  pinMode(kImuResetPin, INPUT_PULLDOWN);
  delay(5);
  Serial.printf("  input pulldown: INT=%d RST=%d (1 suggests an external pullup is connected)\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));

  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, INPUT_PULLUP);
  delay(5);
  Serial.printf("  input pullup:   INT=%d RST=%d\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));

  pinMode(kImuResetPin, OUTPUT);
  digitalWrite(kImuResetPin, LOW);
  delay(25);
  Serial.printf("  drive RST low:  INT=%d RST=%d\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));

  digitalWrite(kImuResetPin, HIGH);
  delay(250);
  Serial.printf("  drive RST high after 250ms: INT=%d RST=%d\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));

  pinMode(kImuResetPin, INPUT_PULLUP);
  delay(5);
  Serial.printf("  released:       INT=%d RST=%d\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));
}

void debugBusDiag() {
  Wire.end();
  delay(10);

  Serial.println("SDA/SCL GPIO electrical check:");
  pinMode(kImuSdaPin, INPUT);
  pinMode(kImuSclPin, INPUT);
  delay(5);
  Serial.printf("  floating input: SDA=%d SCL=%d\n",
                digitalRead(kImuSdaPin),
                digitalRead(kImuSclPin));

  pinMode(kImuSdaPin, INPUT_PULLDOWN);
  pinMode(kImuSclPin, INPUT_PULLDOWN);
  delay(5);
  Serial.printf("  input pulldown: SDA=%d SCL=%d (1 suggests an external pullup is connected)\n",
                digitalRead(kImuSdaPin),
                digitalRead(kImuSclPin));

  pinMode(kImuSdaPin, INPUT_PULLUP);
  pinMode(kImuSclPin, INPUT_PULLUP);
  delay(5);
  Serial.printf("  input pullup:   SDA=%d SCL=%d\n",
                digitalRead(kImuSdaPin),
                digitalRead(kImuSclPin));

  pinMode(kImuSdaPin, OUTPUT_OPEN_DRAIN);
  digitalWrite(kImuSdaPin, LOW);
  pinMode(kImuSclPin, INPUT_PULLUP);
  delay(5);
  Serial.printf("  drive SDA low:  SDA=%d SCL=%d\n",
                digitalRead(kImuSdaPin),
                digitalRead(kImuSclPin));

  pinMode(kImuSdaPin, INPUT_PULLUP);
  pinMode(kImuSclPin, OUTPUT_OPEN_DRAIN);
  digitalWrite(kImuSclPin, LOW);
  delay(5);
  Serial.printf("  drive SCL low:  SDA=%d SCL=%d\n",
                digitalRead(kImuSdaPin),
                digitalRead(kImuSclPin));

  pinMode(kImuSdaPin, INPUT_PULLUP);
  pinMode(kImuSclPin, INPUT_PULLUP);
  delay(5);
  Serial.printf("  released:       SDA=%d SCL=%d\n",
                digitalRead(kImuSdaPin),
                digitalRead(kImuSclPin));

  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);
}

int16_t readLeI16(const uint8_t *data) {
  return static_cast<int16_t>(static_cast<uint16_t>(data[0]) |
                              (static_cast<uint16_t>(data[1]) << 8));
}

uint32_t readLeU32(const uint8_t *data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

float qToFloat(int16_t value, uint8_t qPoint) {
  return static_cast<float>(value) / static_cast<float>(1UL << qPoint);
}

void printHexByte(uint8_t value) {
  Serial.print("0x");
  if (value < 16) {
    Serial.print("0");
  }
  Serial.print(value, HEX);
}

void printHexBytes(const uint8_t *data, uint16_t length, uint16_t limit = 48) {
  const uint16_t count = min(length, limit);
  for (uint16_t i = 0; i < count; ++i) {
    Serial.print(" ");
    printHexByte(data[i]);
  }
  if (length > limit) {
    Serial.print(" ...");
  }
}

bool debugPrintRvcFrame(const uint8_t *frame) {
  if (frame[0] != 0xAA || frame[1] != 0xAA) {
    return false;
  }

  uint8_t checksum = 0;
  for (uint8_t i = 2; i < 18; ++i) {
    checksum += frame[i];
  }

  const int16_t yawRaw = readLeI16(&frame[3]);
  const int16_t pitchRaw = readLeI16(&frame[5]);
  const int16_t rollRaw = readLeI16(&frame[7]);
  const int16_t accelXRaw = readLeI16(&frame[9]);
  const int16_t accelYRaw = readLeI16(&frame[11]);
  const int16_t accelZRaw = readLeI16(&frame[13]);

  Serial.printf("RVC frame index=%u yaw=%.2f pitch=%.2f roll=%.2f accel_mg=(%d,%d,%d) checksum=%s\n",
                frame[2],
                yawRaw / 100.0f,
                pitchRaw / 100.0f,
                rollRaw / 100.0f,
                accelXRaw,
                accelYRaw,
                accelZRaw,
                checksum == frame[18] ? "ok" : "bad");
  return true;
}

void debugSniffRvc() {
  constexpr uint32_t kSniffMs = 3000;
  constexpr size_t kRawLimit = 128;

  Wire.end();
  delay(10);
  imuRvcSerial.end();
  delay(10);
  imuRvcSerial.begin(115200, SERIAL_8N1, kImuSdaPin, kImuSclPin);
  delay(30);

  uint8_t raw[kRawLimit] = {};
  size_t rawCount = 0;
  uint8_t frame[19] = {};
  uint8_t frameCount = 0;
  uint16_t bytesSeen = 0;
  uint8_t decodedFrames = 0;
  const uint32_t start = millis();

  Serial.printf("Sniffing UART-RVC for %lu ms on RX=D4/GPIO%u, TX=D5/GPIO%u at 115200 8N1\n",
                static_cast<unsigned long>(kSniffMs),
                kImuSdaPin,
                kImuSclPin);

  while (millis() - start < kSniffMs) {
    while (imuRvcSerial.available()) {
      const uint8_t byte = static_cast<uint8_t>(imuRvcSerial.read());
      ++bytesSeen;
      if (rawCount < kRawLimit) {
        raw[rawCount++] = byte;
      }

      if (frameCount == 0 && byte != 0xAA) {
        continue;
      }
      if (frameCount == 1 && byte != 0xAA) {
        frameCount = byte == 0xAA ? 1 : 0;
        frame[0] = 0xAA;
        continue;
      }

      frame[frameCount++] = byte;
      if (frameCount == sizeof(frame)) {
        if (debugPrintRvcFrame(frame)) {
          ++decodedFrames;
        }
        frameCount = 0;
      }
    }
    delay(1);
  }

  Serial.printf("UART-RVC bytes_seen=%u decoded_frames=%u raw_first_%u:",
                bytesSeen,
                decodedFrames,
                static_cast<unsigned int>(rawCount));
  for (size_t i = 0; i < rawCount; ++i) {
    Serial.print(" 0x");
    if (raw[i] < 16) {
      Serial.print("0");
    }
    Serial.print(raw[i], HEX);
  }
  if (rawCount == 0) {
    Serial.print(" none");
  }
  Serial.println();

  imuRvcSerial.end();
  delay(10);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);
}

bool debugReadShtpHeader(uint8_t address) {
  uint8_t header[4] = {};
  const uint8_t requested = Wire.requestFrom(address, static_cast<uint8_t>(4));
  Serial.printf("Raw read 0x%02x requested 4 got %u:", address, requested);
  for (uint8_t i = 0; i < requested && Wire.available(); ++i) {
    header[i] = Wire.read();
    Serial.print(" 0x");
    if (header[i] < 16) {
      Serial.print("0");
    }
    Serial.print(header[i], HEX);
  }
  while (Wire.available()) {
    Wire.read();
  }
  Serial.println();

  if (requested == 4) {
    uint16_t packetSize = static_cast<uint16_t>(header[0]) | (static_cast<uint16_t>(header[1]) << 8);
    packetSize &= ~0x8000;
    Serial.printf("SHTP header packet_size=%u channel=%u seq=%u\n",
                  packetSize,
                  header[2],
                  header[3]);
    return packetSize >= 4;
  }
  return false;
}

struct ShtpPacket {
  uint16_t length = 0;
  uint8_t channel = 0;
  uint8_t sequence = 0;
  uint16_t payloadLength = 0;
  uint8_t bytes[kShtpMaxPacketBytes] = {};
};

bool debugI2cReadExact(uint8_t address, uint8_t *buffer, uint8_t length) {
  const uint8_t received = Wire.requestFrom(address, length);
  if (received != length) {
    while (Wire.available()) {
      Wire.read();
    }
    return false;
  }

  for (uint8_t i = 0; i < length; ++i) {
    if (!Wire.available()) {
      return false;
    }
    buffer[i] = static_cast<uint8_t>(Wire.read());
  }
  return true;
}

bool debugReadShtpPacket(uint8_t address, ShtpPacket &packet, bool printErrors) {
  packet = {};

  uint8_t header[4] = {};
  if (!debugI2cReadExact(address, header, sizeof(header))) {
    return false;
  }

  uint16_t packetLength = static_cast<uint16_t>(header[0]) | (static_cast<uint16_t>(header[1]) << 8);
  packetLength &= ~0x8000;
  if (packetLength < 4 || packetLength > kShtpMaxPacketBytes) {
    if (printErrors) {
      Serial.printf("SHTP invalid packet length=%u header:", packetLength);
      printHexBytes(header, sizeof(header), sizeof(header));
      Serial.println();
    }
    return false;
  }

  uint16_t remaining = packetLength;
  uint16_t copied = 0;
  bool firstRead = true;
  while (remaining > 0) {
    const uint16_t wanted = firstRead ? remaining : remaining + 4;
    const uint16_t requested = wanted < kShtpWireChunkSize ? wanted : kShtpWireChunkSize;
    uint8_t chunk[kShtpWireChunkSize] = {};
    if (!debugI2cReadExact(address, chunk, static_cast<uint8_t>(requested))) {
      if (printErrors) {
        Serial.printf("SHTP packet read failed after %u/%u bytes\n", copied, packetLength);
      }
      return false;
    }

    uint8_t *source = chunk;
    uint16_t cargoRead = requested;
    if (!firstRead) {
      source += 4;
      cargoRead -= 4;
    }
    memcpy(packet.bytes + copied, source, cargoRead);
    copied += cargoRead;
    remaining -= cargoRead;
    firstRead = false;
  }

  packet.length = packetLength;
  packet.channel = packet.bytes[2];
  packet.sequence = packet.bytes[3];
  packet.payloadLength = packet.length - 4;
  return true;
}

void debugPrintShtpPacket(const ShtpPacket &packet) {
  Serial.printf("SHTP packet len=%u channel=%u seq=%u payload=%u:",
                packet.length,
                packet.channel,
                packet.sequence,
                packet.payloadLength);
  printHexBytes(packet.bytes + 4, packet.payloadLength);
  Serial.println();
}

void copyAdvertText(char *dest, size_t destSize, const uint8_t *src, uint8_t length) {
  const size_t copyLength = min(destSize - 1, static_cast<size_t>(length));
  memcpy(dest, src, copyLength);
  dest[copyLength] = '\0';
}

void debugDecodeAdvertisement(const uint8_t *payload, uint16_t length) {
  if (length == 0 || payload[0] != 0) {
    return;
  }

  uint16_t cursor = 1;
  uint32_t guid = 0;
  uint8_t channelNo = 0;
  bool wake = false;
  char appName[32] = "";
  char channelName[32] = "";

  while (cursor + 2 <= length) {
    const uint8_t tag = payload[cursor++];
    const uint8_t tagLength = payload[cursor++];
    if (cursor + tagLength > length) {
      Serial.printf("Advertisement tag overflow tag=0x%02x len=%u\n", tag, tagLength);
      return;
    }
    const uint8_t *value = payload + cursor;
    cursor += tagLength;

    switch (tag) {
      case 1:
        guid = tagLength >= 4 ? readLeU32(value) : 0;
        appName[0] = '\0';
        channelName[0] = '\0';
        break;
      case 6:
        channelNo = tagLength >= 1 ? value[0] : 0;
        wake = false;
        break;
      case 7:
        channelNo = tagLength >= 1 ? value[0] : 0;
        wake = true;
        break;
      case 8:
        copyAdvertText(appName, sizeof(appName), value, tagLength);
        break;
      case 9:
        copyAdvertText(channelName, sizeof(channelName), value, tagLength);
        Serial.printf("advert guid=%lu app=%s channel=%s no=%u wake=%s\n",
                      static_cast<unsigned long>(guid),
                      appName,
                      channelName,
                      channelNo,
                      wake ? "true" : "false");
        if (guid == 2 && strcmp(appName, "sensorhub") == 0) {
          if (strcmp(channelName, "control") == 0) {
            shtpControlChannel = channelNo;
          } else if (strcmp(channelName, "inputNormal") == 0) {
            shtpInputNormalChannel = channelNo;
          } else if (strcmp(channelName, "inputWake") == 0) {
            shtpInputWakeChannel = channelNo;
          }
        }
        break;
      case 0x81:
        Serial.print("advert report lengths:");
        for (uint8_t i = 0; i + 1 < tagLength; i += 2) {
          const uint8_t reportId = value[i];
          if (reportId == 0x01 || reportId == 0x05 || reportId == 0xFB ||
              reportId == 0xFA || reportId == 0xEF || reportId == 0xF1 ||
              reportId == 0xF8 || reportId == 0xFC) {
            Serial.printf(" 0x%02x=%u", reportId, value[i + 1]);
          }
        }
        Serial.println();
        break;
      default:
        break;
    }
  }
}

uint8_t directReportLength(uint8_t reportId) {
  switch (reportId) {
    case 0x01:
    case 0x02:
    case 0x04:
      return 10;
    case 0x05:
      return 14;
    case 0xFB:
    case 0xFA:
      return 5;
    case 0xEF:
      return 2;
    case 0xF1:
    case 0xF8:
      return 16;
    case 0xFC:
      return 17;
    default:
      return 0;
  }
}

void debugDecodeSensorPayload(const uint8_t *payload, uint16_t length) {
  uint16_t cursor = 0;
  while (cursor < length) {
    const uint8_t reportId = payload[cursor];
    const uint8_t reportLength = directReportLength(reportId);
    if (reportLength == 0 || cursor + reportLength > length) {
      Serial.printf("direct unknown report id=0x%02x remaining=%u:",
                    reportId,
                    length - cursor);
      printHexBytes(payload + cursor, length - cursor);
      Serial.println();
      return;
    }

    const uint8_t *report = payload + cursor;
    if (reportId == 0x01) {
      const float x = qToFloat(readLeI16(report + 4), 8);
      const float y = qToFloat(readLeI16(report + 6), 8);
      const float z = qToFloat(readLeI16(report + 8), 8);
      const uint8_t status = report[2] & 0x03;
      imuState.accelX = x;
      imuState.accelY = y;
      imuState.accelZ = z;
      imuState.accelAccuracy = status;
      imuState.updatedMs = millis();
      Serial.printf("direct accel x=%.3f y=%.3f z=%.3f status=%u\n", x, y, z, status);
    } else if (reportId == 0x02) {
      const float x = qToFloat(readLeI16(report + 4), 9);
      const float y = qToFloat(readLeI16(report + 6), 9);
      const float z = qToFloat(readLeI16(report + 8), 9);
      const uint8_t status = report[2] & 0x03;
      imuState.gyroX = x;
      imuState.gyroY = y;
      imuState.gyroZ = z;
      imuState.gyroAccuracy = status;
      imuState.updatedMs = millis();
      Serial.printf("direct gyro x=%.3f y=%.3f z=%.3f status=%u\n", x, y, z, status);
    } else if (reportId == 0x04) {
      const float x = qToFloat(readLeI16(report + 4), 8);
      const float y = qToFloat(readLeI16(report + 6), 8);
      const float z = qToFloat(readLeI16(report + 8), 8);
      const uint8_t status = report[2] & 0x03;
      imuState.linearAccelX = x;
      imuState.linearAccelY = y;
      imuState.linearAccelZ = z;
      imuState.linearAccelAccuracy = status;
      imuState.updatedMs = millis();
      Serial.printf("direct linear_accel x=%.3f y=%.3f z=%.3f status=%u\n", x, y, z, status);
    } else if (reportId == 0x05) {
      const float qi = qToFloat(readLeI16(report + 4), 14);
      const float qj = qToFloat(readLeI16(report + 6), 14);
      const float qk = qToFloat(readLeI16(report + 8), 14);
      const float qr = qToFloat(readLeI16(report + 10), 14);
      const float accuracy = qToFloat(readLeI16(report + 12), 12);
      const uint8_t status = report[2] & 0x03;
      imuState.qi = qi;
      imuState.qj = qj;
      imuState.qk = qk;
      imuState.qr = qr;
      imuState.quatRadAccuracy = accuracy;
      imuState.quatAccuracy = status;
      imuState.updatedMs = millis();
      Serial.printf("direct quat i=%.4f j=%.4f k=%.4f r=%.4f rad_acc=%.4f status=%u\n",
                    qi,
                    qj,
                    qk,
                    qr,
                    accuracy,
                    status);
    } else if (reportId == 0xFB) {
      Serial.printf("direct timestamp base=%lu\n", static_cast<unsigned long>(readLeU32(report + 1)));
    } else if (reportId == 0xFA) {
      Serial.printf("direct timestamp rebase=%ld\n", static_cast<long>(static_cast<int32_t>(readLeU32(report + 1))));
    } else if (reportId == 0xF8) {
      Serial.printf("direct product reset=%u sw=%u.%u.%u part=%lu build=%lu\n",
                    report[1],
                    report[2],
                    report[3],
                    static_cast<unsigned int>(static_cast<uint16_t>(report[12]) |
                                              (static_cast<uint16_t>(report[13]) << 8)),
                    static_cast<unsigned long>(readLeU32(report + 4)),
                    static_cast<unsigned long>(readLeU32(report + 8)));
    } else if (reportId == 0xF1) {
      Serial.printf("direct command response cmd=0x%02x seq=%u resp=%u values:",
                    report[2],
                    report[1],
                    report[4]);
      printHexBytes(report + 5, 11, 11);
      Serial.println();
    } else if (reportId == 0xFC) {
      Serial.printf("direct feature response sensor=0x%02x interval_us=%lu\n",
                    report[1],
                    static_cast<unsigned long>(readLeU32(report + 5)));
    } else {
      Serial.printf("direct report id=0x%02x len=%u\n", reportId, reportLength);
    }

    cursor += reportLength;
  }
}

void debugDecodeShtpPacket(const ShtpPacket &packet) {
  const uint8_t *payload = packet.bytes + 4;
  if (packet.channel == kShtpChannelCommand) {
    debugDecodeAdvertisement(payload, packet.payloadLength);
  } else if (packet.channel == shtpControlChannel ||
             packet.channel == shtpInputNormalChannel ||
             packet.channel == shtpInputWakeChannel) {
    debugDecodeSensorPayload(payload, packet.payloadLength);
  }
}

void debugDumpShtpPackets(uint32_t durationMs, bool decode) {
  const uint32_t previousTimeout = kImuWireTimeoutMs;
  Wire.setTimeOut(25);
  Wire.setClock(kImuInitI2cClockHz);

  uint16_t packets = 0;
  uint16_t misses = 0;
  uint32_t nextPollMs = 0;
  const uint32_t start = millis();
  while (millis() - start < durationMs) {
    const bool intLow = digitalRead(kImuIntPin) == LOW;
    if (!intLow && millis() < nextPollMs) {
      delay(1);
      continue;
    }
    nextPollMs = millis() + 25;

    ShtpPacket packet;
    if (debugReadShtpPacket(kImuPrimaryAddress, packet, false)) {
      ++packets;
      debugPrintShtpPacket(packet);
      if (decode) {
        debugDecodeShtpPacket(packet);
      }
    } else {
      ++misses;
      delay(2);
    }
  }

  Wire.setTimeOut(previousTimeout);
  Serial.printf("SHTP dump finished packets=%u misses=%u channels control=%u input=%u wake=%u\n",
                packets,
                misses,
                shtpControlChannel,
                shtpInputNormalChannel,
                shtpInputWakeChannel);
}

bool debugWaitForIntLow(uint32_t timeoutMs) {
  const uint32_t start = millis();
  while (millis() - start < timeoutMs) {
    if (digitalRead(kImuIntPin) == LOW) {
      return true;
    }
    delay(1);
  }
  return digitalRead(kImuIntPin) == LOW;
}

bool sendShtpPayload(uint8_t channel, const uint8_t *payload, uint16_t payloadLength, bool verbose) {
  if (channel >= sizeof(shtpTxSeq)) {
    if (verbose) {
      Serial.printf("Direct SHTP bad channel=%u\n", channel);
    }
    return false;
  }
  if (payloadLength + 4 > kShtpWireChunkSize) {
    if (verbose) {
      Serial.printf("Direct SHTP payload too large=%u\n", payloadLength);
    }
    return false;
  }

  uint8_t packet[kShtpWireChunkSize] = {};
  const uint16_t packetLength = payloadLength + 4;
  packet[0] = packetLength & 0xFF;
  packet[1] = (packetLength >> 8) & 0x7F;
  packet[2] = channel;
  packet[3] = shtpTxSeq[channel]++;
  memcpy(packet + 4, payload, payloadLength);

  const bool intReady = debugWaitForIntLow(250);
  Wire.beginTransmission(kImuPrimaryAddress);
  Wire.write(packet, packetLength);
  const uint8_t error = Wire.endTransmission();
  if (verbose) {
    Serial.printf("Direct SHTP write channel=%u seq=%u payload=%u int_ready=%s result=%u\n",
                  channel,
                  packet[3],
                  payloadLength,
                  intReady ? "true" : "false",
                  error);
  }
  return error == 0;
}

bool debugSendShtpPayload(uint8_t channel, const uint8_t *payload, uint16_t payloadLength) {
  return sendShtpPayload(channel, payload, payloadLength, true);
}

bool debugSendAdvertRequest() {
  const uint8_t payload[] = {0x00, 0x01};
  Serial.println("Direct SHTP advertisement request");
  return debugSendShtpPayload(kShtpChannelCommand, payload, sizeof(payload));
}

bool debugSendProductIdRequest() {
  const uint8_t payload[] = {0xF9, 0x00};
  Serial.println("Direct SH-2 product ID request");
  return debugSendShtpPayload(shtpControlChannel, payload, sizeof(payload));
}

bool debugSendGetFeature(uint8_t reportId) {
  const uint8_t payload[] = {0xFE, reportId};
  Serial.printf("Direct SH-2 get feature report=0x%02x\n", reportId);
  return debugSendShtpPayload(shtpControlChannel, payload, sizeof(payload));
}

bool debugSendSetFeature(uint8_t reportId, uint32_t intervalUs) {
  uint8_t payload[17] = {};
  payload[0] = 0xFD;
  payload[1] = reportId;
  payload[2] = 0x00;
  payload[3] = 0x00;
  payload[4] = 0x00;
  payload[5] = intervalUs & 0xFF;
  payload[6] = (intervalUs >> 8) & 0xFF;
  payload[7] = (intervalUs >> 16) & 0xFF;
  payload[8] = (intervalUs >> 24) & 0xFF;
  Serial.printf("Direct SH-2 set feature report=0x%02x interval_us=%lu\n",
                reportId,
                static_cast<unsigned long>(intervalUs));
  return debugSendShtpPayload(shtpControlChannel, payload, sizeof(payload));
}

bool sendProductIdRequest(bool verbose) {
  const uint8_t payload[] = {0xF9, 0x00};
  return sendShtpPayload(shtpControlChannel, payload, sizeof(payload), verbose);
}

bool sendSetFeature(uint8_t reportId, uint32_t intervalUs, bool verbose) {
  uint8_t payload[17] = {};
  payload[0] = 0xFD;
  payload[1] = reportId;
  payload[2] = 0x00;
  payload[3] = 0x00;
  payload[4] = 0x00;
  payload[5] = intervalUs & 0xFF;
  payload[6] = (intervalUs >> 8) & 0xFF;
  payload[7] = (intervalUs >> 16) & 0xFF;
  payload[8] = (intervalUs >> 24) & 0xFF;
  return sendShtpPayload(shtpControlChannel, payload, sizeof(payload), verbose);
}

uint16_t updateImuStateFromDirectPayload(const uint8_t *payload, uint16_t length) {
  uint16_t cursor = 0;
  uint16_t updates = 0;
  while (cursor < length) {
    const uint8_t reportId = payload[cursor];
    const uint8_t reportLength = directReportLength(reportId);
    if (reportLength == 0 || cursor + reportLength > length) {
      return updates;
    }

    const uint8_t *report = payload + cursor;
    if (reportId == 0x01) {
      imuState.accelX = qToFloat(readLeI16(report + 4), 8);
      imuState.accelY = qToFloat(readLeI16(report + 6), 8);
      imuState.accelZ = qToFloat(readLeI16(report + 8), 8);
      imuState.accelAccuracy = report[2] & 0x03;
      imuState.updatedMs = millis();
      ++updates;
    } else if (reportId == 0x02) {
      imuState.gyroX = qToFloat(readLeI16(report + 4), 9);
      imuState.gyroY = qToFloat(readLeI16(report + 6), 9);
      imuState.gyroZ = qToFloat(readLeI16(report + 8), 9);
      imuState.gyroAccuracy = report[2] & 0x03;
      imuState.updatedMs = millis();
      ++updates;
    } else if (reportId == 0x04) {
      imuState.linearAccelX = qToFloat(readLeI16(report + 4), 8);
      imuState.linearAccelY = qToFloat(readLeI16(report + 6), 8);
      imuState.linearAccelZ = qToFloat(readLeI16(report + 8), 8);
      imuState.linearAccelAccuracy = report[2] & 0x03;
      imuState.updatedMs = millis();
      ++updates;
    } else if (reportId == 0x05) {
      imuState.qi = qToFloat(readLeI16(report + 4), 14);
      imuState.qj = qToFloat(readLeI16(report + 6), 14);
      imuState.qk = qToFloat(readLeI16(report + 8), 14);
      imuState.qr = qToFloat(readLeI16(report + 10), 14);
      imuState.quatRadAccuracy = qToFloat(readLeI16(report + 12), 12);
      imuState.quatAccuracy = report[2] & 0x03;
      imuState.updatedMs = millis();
      ++updates;
    }

    cursor += reportLength;
  }
  return updates;
}

uint16_t drainDirectShtpPackets(uint32_t durationMs, bool updateState) {
  uint16_t updates = 0;
  uint32_t nextPollMs = 0;
  const uint32_t start = millis();
  while (millis() - start < durationMs) {
    const bool intLow = digitalRead(kImuIntPin) == LOW;
    if (!intLow && millis() < nextPollMs) {
      delay(1);
      continue;
    }
    nextPollMs = millis() + 25;

    ShtpPacket packet;
    if (debugReadShtpPacket(kImuPrimaryAddress, packet, false)) {
      if (packet.channel == kShtpChannelCommand) {
        debugDecodeAdvertisement(packet.bytes + 4, packet.payloadLength);
      } else if (updateState && (packet.channel == shtpInputNormalChannel ||
                                 packet.channel == shtpInputWakeChannel)) {
        updates += updateImuStateFromDirectPayload(packet.bytes + 4, packet.payloadLength);
      }
    } else {
      delay(2);
    }
  }
  return updates;
}

bool initImuDirect() {
  imuReady = false;
  imuAddress = 0;
  imuReportsEnabled = false;
  imuInitMode = "direct_shtp";
  imuLastError = "starting";
  lastImuInitAttemptMs = millis();
  memset(shtpTxSeq, 0, sizeof(shtpTxSeq));
  shtpControlChannel = kShtpDefaultControlChannel;
  shtpInputNormalChannel = kShtpDefaultInputNormalChannel;
  shtpInputWakeChannel = kShtpDefaultInputWakeChannel;

  prepareImuControlPins();
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(25);
  Wire.setClock(kImuInitI2cClockHz);
  logI2cScan();

  if (!i2cAddressAcks(kImuPrimaryAddress)) {
    imuLastError = "no ACK on 0x4b";
    Serial.println("BNO08x IMU not found at I2C address 0x4b");
    return false;
  }

  pulseBnoReset();
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(25);
  Wire.setClock(kImuInitI2cClockHz);
  drainDirectShtpPackets(700, false);
  sendProductIdRequest(false);
  drainDirectShtpPackets(250, false);

  const bool rotationWriteOk = sendSetFeature(0x05, kShtpDirectReportIntervalUs, false);
  delay(20);
  const bool accelWriteOk = sendSetFeature(0x01, kShtpDirectReportIntervalUs, false);
  const uint16_t updates = drainDirectShtpPackets(1000, true);

  imuAddress = kImuPrimaryAddress;
  imuReportsEnabled = updates > 0;
  imuReady = imuReportsEnabled;
  Wire.setClock(kImuI2cClockHz);

  if (!imuReportsEnabled) {
    imuLastError = rotationWriteOk || accelWriteOk
                     ? "direct SHTP configured but no reports received"
                     : "direct SHTP report enable write failed";
    Serial.println(imuLastError);
    return false;
  }

  imuLastError = "";
  Serial.printf("BNO08x direct SHTP ready on SDA D4/GPIO%u, SCL D5/GPIO%u, address 0x%02x\n",
                kImuSdaPin,
                kImuSclPin,
                imuAddress);
  return true;
}

void updateImuDirect() {
  if (digitalRead(kImuIntPin) != LOW) {
    return;
  }

  for (uint8_t i = 0; i < 2; ++i) {
    ShtpPacket packet;
    if (!debugReadShtpPacket(kImuPrimaryAddress, packet, false)) {
      break;
    }
    if (packet.channel == shtpInputNormalChannel || packet.channel == shtpInputWakeChannel) {
      updateImuStateFromDirectPayload(packet.bytes + 4, packet.payloadLength);
    }
    if (digitalRead(kImuIntPin) != LOW) {
      break;
    }
  }
}

void debugDirectShtpSession() {
  Serial.println("Starting direct SHTP session on BNO08x I2C without SparkFun SH2 init.");
  memset(shtpTxSeq, 0, sizeof(shtpTxSeq));
  shtpControlChannel = kShtpDefaultControlChannel;
  shtpInputNormalChannel = kShtpDefaultInputNormalChannel;
  shtpInputWakeChannel = kShtpDefaultInputWakeChannel;
  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, INPUT_PULLUP);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(25);
  Wire.setClock(kImuInitI2cClockHz);

  debugPrintPins();
  if (!i2cAddressAcks(kImuPrimaryAddress)) {
    Serial.println("Direct SHTP aborted: no ACK at 0x4B.");
    return;
  }

  Serial.println("Direct SHTP initial packet drain");
  debugDumpShtpPackets(500, true);
  debugSendAdvertRequest();
  debugDumpShtpPackets(700, true);
  debugSendProductIdRequest();
  debugDumpShtpPackets(700, true);
  debugSendSetFeature(0x05, kShtpDirectReportIntervalUs);
  delay(20);
  debugSendSetFeature(0x01, kShtpDirectReportIntervalUs);
  delay(20);
  debugSendSetFeature(0x02, kShtpDirectReportIntervalUs);
  delay(20);
  debugSendSetFeature(0x04, kShtpDirectReportIntervalUs);
  debugSendGetFeature(0x05);
  debugSendGetFeature(0x01);
  debugSendGetFeature(0x02);
  debugSendGetFeature(0x04);
  Serial.println("Direct SHTP report stream");
  debugDumpShtpPackets(kShtpDumpDurationMs, true);
}

void debugPrintImuEvent();

void debugPulseReset() {
  Serial.println("Pulsing BNO RST low, then releasing high via input pullup.");
  pinMode(kImuResetPin, OUTPUT);
  digitalWrite(kImuResetPin, HIGH);
  delay(20);
  digitalWrite(kImuResetPin, LOW);
  delay(25);
  digitalWrite(kImuResetPin, HIGH);
  delay(20);
  pinMode(kImuResetPin, INPUT_PULLUP);

  for (uint8_t i = 0; i < 8; ++i) {
    delay(250);
    Serial.printf("Reset wait %ums: INT=%d RST=%d ",
                  static_cast<unsigned int>((i + 1) * 250),
                  digitalRead(kImuIntPin),
                  digitalRead(kImuResetPin));
    debugPrintI2cScan();
  }
}

void debugTryBnoBegin() {
  Serial.println("Trying SparkFun BNO08x begin without automatic reset.");
  imuReady = false;
  imuAddress = 0;
  imuInitMode = "";
  imuReportsEnabled = false;

  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, INPUT_PULLUP);
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);

  if (tryBeginImuAt(kImuPrimaryAddress, kImuIntPin, -1, "int_only") ||
      tryBeginImuAt(kImuSecondaryAddress, kImuIntPin, -1, "int_only")) {
    imuReady = true;
    imuReportsEnabled = enableImuReports();
    Wire.setClock(kImuI2cClockHz);
    Serial.printf("BNO begin OK address=0x%02x mode=%s reports=%s\n",
                  imuAddress,
                  imuInitMode.c_str(),
                  imuReportsEnabled ? "enabled" : "failed");
    return;
  }

  Serial.println("BNO begin failed.");
}

void debugTryBnoBeginFullReset() {
  Serial.println("Trying SparkFun BNO08x begin with INT and RST pins.");
  imuReady = false;
  imuAddress = 0;
  imuInitMode = "";
  imuReportsEnabled = false;

  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, INPUT_PULLUP);
  Wire.end();
  delay(10);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);

  if (tryBeginImuAt(kImuPrimaryAddress, kImuIntPin, kImuResetPin, "int_reset") ||
      tryBeginImuAt(kImuSecondaryAddress, kImuIntPin, kImuResetPin, "int_reset")) {
    imuReady = true;
    imuReportsEnabled = enableImuReports();
    Wire.setClock(kImuI2cClockHz);
    Serial.printf("BNO begin OK address=0x%02x mode=%s reports=%s\n",
                  imuAddress,
                  imuInitMode.c_str(),
                  imuReportsEnabled ? "enabled" : "failed");
    return;
  }

  Serial.println("BNO begin with reset failed.");
}

void debugStreamImuEvents(uint32_t durationMs) {
  const uint32_t start = millis();
  uint16_t eventLoops = 0;
  Serial.printf("Streaming IMU events for %lu ms, ready=%s reports=%s\n",
                static_cast<unsigned long>(durationMs),
                imuReady ? "true" : "false",
                imuReportsEnabled ? "true" : "false");

  if (!imuReady) {
    debugTryBnoBegin();
  }
  if (!imuReady) {
    debugTryBnoBeginFullReset();
  }
  if (!imuReady) {
    Serial.println("Cannot stream IMU events because driver init is not ready.");
    return;
  }
  if (!imuReportsEnabled) {
    imuReportsEnabled = enableImuReports();
    Serial.printf("Report enable retry: %s\n", imuReportsEnabled ? "ok" : "failed");
  }

  while (millis() - start < durationMs) {
    debugPrintImuEvent();
    ++eventLoops;
    delay(5);
  }
  Serial.printf("Finished IMU event stream loops=%u\n", eventLoops);
}

void debugHoldResetHigh() {
  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, OUTPUT);
  digitalWrite(kImuResetPin, HIGH);
  delay(20);
  Serial.printf("RST actively driven high: INT=%d RST=%d\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));
}

void debugReleaseResetPullup() {
  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, INPUT_PULLUP);
  delay(20);
  Serial.printf("RST released to input pullup: INT=%d RST=%d\n",
                digitalRead(kImuIntPin),
                digitalRead(kImuResetPin));
}

void debugRunBatch() {
  Serial.println("batch: pin levels");
  debugPrintAllPinLevels();
  Serial.println("batch: INT/RST electrical behavior");
  debugControlPinDiag();
  Serial.println("batch: SDA/SCL electrical behavior");
  debugBusDiag();
  Serial.println("batch: reset released, nominal I2C scan");
  debugReleaseResetPullup();
  debugPrintI2cScan();
  Serial.println("batch: clock-rate I2C scans");
  debugScanI2cClockRates();
  Serial.println("batch: UART-RVC sniff");
  debugSniffRvc();
  Serial.println("batch: reset actively high, nominal I2C scan");
  debugHoldResetHigh();
  debugPrintI2cScan();
  Serial.println("batch: candidate I2C pin pairs");
  debugScanCandidatePinPairs();
  Serial.println("batch: exhaustive I2C pin-pair scan");
  debugScanAllPinPairs();
  Serial.println("batch: raw SHTP reads");
  debugReadShtpHeader(kImuPrimaryAddress);
  debugReadShtpHeader(kImuSecondaryAddress);
  Serial.println("batch: direct SHTP report enable/read");
  debugDirectShtpSession();
  Serial.println("batch: final pins");
  debugPrintPins();
}

void debugPrintImuEvent() {
  if (!imuReady) {
    return;
  }

  if (imu.wasReset()) {
    Serial.println("BNO reported reset; enabling reports again.");
    imuReportsEnabled = enableImuReports();
  }

  for (uint8_t i = 0; i < 8 && imu.getSensorEvent(); ++i) {
    const uint8_t eventId = imu.getSensorEventID();
    if (eventId == SENSOR_REPORTID_ROTATION_VECTOR) {
      Serial.printf("quat i=%.4f j=%.4f k=%.4f r=%.4f acc=%u\n",
                    imu.getQuatI(),
                    imu.getQuatJ(),
                    imu.getQuatK(),
                    imu.getQuatReal(),
                    imu.getQuatAccuracy());
    } else if (eventId == SENSOR_REPORTID_ACCELEROMETER) {
      Serial.printf("accel x=%.3f y=%.3f z=%.3f acc=%u\n",
                    imu.getAccelX(),
                    imu.getAccelY(),
                    imu.getAccelZ(),
                    imu.getAccelAccuracy());
    } else {
      Serial.printf("event id=0x%02x\n", eventId);
    }
  }
}

void debugPrintHelp() {
  Serial.println("Commands: help, batch, pins, ctrldiag, levels, busdiag, scan, scanfreqs, scanpins, scanall, rvc, raw, dump, direct, prod, getfeat, init, initfull, events, softreset, reset, rsthigh, rstpullup");
  Serial.println("Use Enter after each command. This debug build does not start the AP/web server.");
}

void debugHandleCommand(const String &command) {
  if (command.length() == 0) {
    return;
  }

  Serial.print("<<<BEGIN ");
  Serial.print(command);
  Serial.println(">>>");

  if (command == "help" || command == "?") {
    debugPrintHelp();
  } else if (command == "batch") {
    debugRunBatch();
  } else if (command == "pins") {
    debugPrintPins();
  } else if (command == "ctrldiag") {
    debugControlPinDiag();
  } else if (command == "levels") {
    debugPrintAllPinLevels();
  } else if (command == "busdiag") {
    debugBusDiag();
  } else if (command == "scan") {
    debugPrintI2cScan();
  } else if (command == "scanfreqs") {
    debugScanI2cClockRates();
  } else if (command == "scanpins") {
    debugScanCandidatePinPairs();
  } else if (command == "scanall") {
    debugScanAllPinPairs();
  } else if (command == "rvc") {
    debugSniffRvc();
  } else if (command == "raw") {
    debugReadShtpHeader(kImuPrimaryAddress);
    debugReadShtpHeader(kImuSecondaryAddress);
  } else if (command == "dump") {
    debugDumpShtpPackets(kShtpDumpDurationMs, true);
  } else if (command == "direct") {
    debugDirectShtpSession();
  } else if (command == "prod") {
    debugSendProductIdRequest();
    debugDumpShtpPackets(700, true);
  } else if (command == "getfeat") {
    debugSendGetFeature(0x05);
    debugSendGetFeature(0x01);
    debugDumpShtpPackets(700, true);
  } else if (command == "init") {
    debugTryBnoBegin();
  } else if (command == "initfull") {
    debugTryBnoBeginFullReset();
  } else if (command == "events") {
    debugStreamImuEvents(3000);
  } else if (command == "softreset") {
    Serial.printf("Soft reset packet to 0x%02x: %s\n",
                  kImuPrimaryAddress,
                  sendBnoSoftReset(kImuPrimaryAddress) ? "ACK" : "NACK");
    delay(500);
    debugPrintI2cScan();
  } else if (command == "reset") {
    debugPulseReset();
  } else if (command == "rsthigh") {
    debugHoldResetHigh();
  } else if (command == "rstpullup") {
    debugReleaseResetPullup();
  } else if (command.length() > 0) {
    Serial.print("Unknown command: ");
    Serial.println(command);
    debugPrintHelp();
  }

  Serial.print("<<<END ");
  Serial.print(command);
  Serial.println(">>>");
}

void debugStartWifiStaOnly() {
  constexpr uint32_t kDebugWifiConnectTimeoutMs = 4000;
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  if (strlen(WIFI_SSID) == 0) {
    Serial.println("No WIFI_SSID configured; serial debugging only.");
    return;
  }

  Serial.print("Connecting STA Wi-Fi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  const uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < kDebugWifiConnectTimeoutMs) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("STA IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("STA Wi-Fi timed out; continuing serial debug.");
  }
}

void debugSerialSetup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.println("Booting XIAO ESP32S3 BNO085 serial debugger");
  Serial.printf("SDA D4/GPIO%u, SCL D5/GPIO%u, INT D9/GPIO%u, RST D8/GPIO%u\n",
                kImuSdaPin,
                kImuSclPin,
                kImuIntPin,
                kImuResetPin);
  debugStartWifiStaOnly();

  pinMode(kImuIntPin, INPUT_PULLUP);
  pinMode(kImuResetPin, INPUT_PULLUP);
  Wire.begin(kImuSdaPin, kImuSclPin);
  Wire.setTimeOut(kImuWireTimeoutMs);
  Wire.setClock(kImuInitI2cClockHz);

  debugPrintHelp();
  debugPrintPins();
  debugPrintI2cScan();
  debugReadShtpHeader(kImuPrimaryAddress);
}

void debugSerialLoop() {
  static String command;

  while (Serial.available()) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      command.trim();
      debugHandleCommand(command);
      command = "";
    } else if (isPrintable(c)) {
      command += c;
    }
  }

  debugPrintImuEvent();
}
}  // namespace

void setup() {
#if defined(IMU_SERIAL_DEBUG_MODE)
  debugSerialSetup();
#else
  Serial.begin(115200);
  delay(1000);
  Serial.println();
  Serial.println("Booting XIAO ESP32S3 camera app");
  initMotorOutputs();
  cameraReady = initCamera();

  if (!cameraReady) {
    Serial.println("Camera setup failed; check the camera ribbon/module connection.");
  }

  startWifi();
  startServer();
  imuReady = false;
  imuAutoInitAttempted = false;
  lastImuInitAttemptMs = millis();
#endif
}

void loop() {
#if defined(IMU_SERIAL_DEBUG_MODE)
  debugSerialLoop();
#else
  server.handleClient();
  updateWifi();
  updateMotorFailsafe();
  updateImu();
#endif
}
