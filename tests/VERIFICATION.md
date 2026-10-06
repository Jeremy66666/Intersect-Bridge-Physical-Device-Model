# Verification — 2026-09-16

## Hardware-free automated tests

`handshake_tests.exe` exited with code 0. It compiled the actual Qt point handler against fake executables placed in the test build directory, and used the same native motor ACK-wait helper. No physical SDK was linked into either emulator; no production hardware executable was launched.

Passing groups:

1. Protocol validation: complete frame, incomplete frame, invalid hex, wrong capture ID, wrong session/point, failed file save, explicit FAIL, stale/frozen heartbeat timeout, cancellation overriding ACK.
2. Three-point end-to-end capture: JSON data saved, matching positions, motor advances, repeated request does not duplicate capture.
3. Pause during capture: save current point, hold motor, resume without log truncation or duplicate acquisition.
4. Backend failure: abort without advancement or success record. Updated test emulates 2.5 s motor cleanup and checks normal exit, not force-kill.
5. Partial 2047-pixel spectrum: reject and abort.
6. Wrong capture ID: reject and abort.
7. Late reply after timeout: reject, do not retry blindly or start another scan on faulted backend.
8. Stop during capture: no success ACK or next point; late raw reply retained separately.
9. Capture Dark during scan: rejected without overwriting the scan command.

Tests use a 1.5 s capture timeout; production uses 15 s. Emulated success is delayed 100 ms; emulated late response 2.2 s. This does not measure actual exposure or motor timing.

The v2.1 regression also checks the real initial-delay calculation: 1 s exposure / 1.1 s cycle gives 1250 ms, longer settings extend the wait. It checks first-frame selection for one/multiple/full/wrapped legacy buffers and rejects empty or invalid frame counts. The real SDK's error after an immediate query was observed on hardware by the user; the restored delay and legacy five-frame capture window must still be confirmed on hardware. They are not emulated as guaranteed success.

## Static preservation checks

- All 120 handwritten `scanX`/`scanY` calls match the backup in order and arguments.
- All original motor numeric constants match the backup.
- The 2048-point Raman shift array matches the backup.
- Original Downloads backend/motor source and executables still match backup SHA-256 hashes.
- Rollback script was syntax-checked, not executed.

## Remaining physical validation

Focus Monitor regression scenarios exercise explicit Laser ON gating, repeated preview acquisition, graph updates, no preview result files, suppression of preview pixels in the log, scan/dark exclusion, Monitor OFF stopping further captures, successful scanning after monitor stops, and preview failure stopping the loop. The fake backend implements the same pipe message format; actual optical focusing still requires supervised hardware testing.

v2.2 adds shared-policy checks for pending status, stale output rejection, success after pending, timeout at the deadline, rejection of late success, hard SDK errors and invalid frame metadata. New mock integration scenarios use the same status policy for repeated rc=1 followed by ready, and repeated rc=1 through timeout. The success scenario verifies only one capture command per point; the timeout scenario verifies no motor advancement. Mock durations are shortened and do not reproduce physical USB behaviour.

Supervised testing with the actual spectrometer and motors is required: SDK frame count/index reporting, exact cycle time, optical quality, travel clearance, timing distribution, and real acquisition success rate. There is no claim of a measured improvement from 90% to another percentage or strict 1-second point cadence.

## 1D Scan verification

Both production Qt and motor builds completed successfully. The full `handshake_tests.exe` suite passed, including X and Y scans with 13 saved points, matching coordinates and scan-mode metadata. Existing preview, pause/resume, cancellation, timeout and malformed-frame tests also passed.

`motor_path_test.exe` additionally compiled the actual motor source against test-only SDK stubs. It verified selected-axis-only homing and all 13 absolute moves (0 through 24 mm), unchanged other-axis/Z positions, and rejection of a 20 mm configured travel range before any motion. It exited with code 0. No physical hardware was operated. The production motor target does not define `MOTOR_PATH_TESTING` and continues to use the real SDK.

The 120 handwritten 2D scan calls and the Raman calibration array remain identical to the pre-1D backup. Physical homing clearance, controller calibration and actual 24 mm travel still require supervised verification.

Run the additional hardware-free check with `tests\build\motor_path_test.exe` using the same Qt PATH as below.

## Re-run commands

Dual-save update: production Qt rebuilt successfully and the full hardware-free suite passed. X/Y tests each produced 13 JSON and 13 TXT point files in mode/time-named directories. Each TXT was parsed with the existing Image Data parser and all 2048 intensities were compared with its JSON counterpart. No physical acquisition was performed and old user scan directories were not modified.

Build `tests/` with VS2019, CMake NMake, Debug, `CMAKE_PREFIX_PATH=C:/Qt/6.6.3/msvc2019_64`. Put `C:\Qt\6.6.3\msvc2019_64\bin` on PATH and run `tests\build\handshake_tests.exe`. Do not replace the production executables with the test emulators.
