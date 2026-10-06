# Raman handshake v2 — 2026-09-16

## Scope

The Qt controller, motor handshake and Hamamatsu single-spectrum capture path were updated together.
Downloads originals are untouched. The editable backend and motor copies now live in `hardware/`.
No laser/motor hardware has been operated for verification. Software build and mock tests do not establish a real-world success percentage.

Preserved: all 120 handwritten XY calls, their relative-motion directions and distances, motor serials, homing/start offsets, Z scan list and return position, 2048-point Raman-shift calibration, 1,000,000 us exposure, 1,100,000 us cycle, low sensor gain and laser power controls.
The existing XY functions `scanX` and `scanY` remain; the handwritten path has NOT been replaced with a new grid generator.
The current sequence measures 4 Z positions and 120 XY points, NOT 7 Z positions or a fully acquired 121-point grid. Its XY initial position is moved to but not measured. This has deliberately not been changed.

## Run

1. Close older Qt controllers and backend/motor processes. Do not run old/new controllers together.
2. Open `build/Desktop_Qt_6_6_3_MSVC2019_64bit-Debug/laser_test.exe`.
3. Laser ON only enables the laser after readback; Start Z scan then 2D scan runs the unchanged motion sequence.
4. In the log, verify `[Handshake v2] Data:` and then `[Point saved]` records.
5. First test a short supervised run, observing the existing homing and Z motion. Do not leave the laser unattended.

The program automatically loads `handshake_v2/backend/SampleApl.exe` and `handshake_v2/motor/RamanAutofocus.exe` beside the Qt executable. Do not manually launch the older Downloads motor/backend.

The existing Z motion includes a move to 20 mm after homing, then the configured offset. This was NOT introduced by this change. Check the physical travel/clearance before running, as before.

## 1D Scan (X/Y selector)

Choose Y axis or X axis, then click **1D Scan**. The selected axis homes to absolute zero, then measures **0, 2, 4, ... 24 mm: 13 points including both endpoints**. It finishes at 24 mm and does not automatically return. The other lateral axis and Z are neither homed nor moved by this branch; focus must already be set. Existing 2D/Z mode is unchanged.

Before homing, the program requests the selected controller's configured travel limits and rejects unavailable/insufficient limits. It does not change those limits. After each absolute move, it checks the reported position before acquiring (100 device-step tolerance, approximately 0.003 mm using the existing calibration). Controller settings and physical clearance must be correct; software limit checks cannot establish sample/objective clearance during homing. Confirm the path is clear before starting.

Each point uses the working numbered handshake and saves its spectrum/positions with `scan_mode` equal to `1d-x` or `1d-y`. Saved 1D and 2D captures **automatically draw live**; there is no additional Live button. **Monitor remains focus preview only** and must be stopped before a scan. Stop does not turn off the laser.

The pre-1D Qt/motor sources and binaries are backed up under the Codex thread directory in `backup_before_1dscan_20260916`.

## Behaviour

### Focus Monitor (added after the working scan version)

- Click Laser ON first, then Monitor. Monitor now acquires fresh spectra repeatedly; it no longer merely polls an old log. It does not enable the laser implicitly and never starts a motor or homes an axis.
- Preview uses the same verified exposure/status-wait/read/cleanup path, one frame at a time. Each successful frame refreshes the graph with a timestamp. This is a manual focusing aid, NOT automatic focusing; the Y axis still autoscales as before.
- Preview pixels travel only through stdout into Qt memory. They are excluded from the text log and are not written as capture-result files or scan JSON files. Small command/ack files and ordinary diagnostic messages still exist. Existing saved data are not deleted.
- Monitor OFF or Stop prevents further preview captures. An in-flight exposure finishes safely before another acquisition can start. Stop does not switch the laser off. For shutdown: Monitor OFF, wait for the current frame, then Laser OFF and confirm.
- Motor scanning and Capture Dark are blocked while Focus Monitor is active. Stop Monitor before using them. Formal scan points now update the graph directly when saved, so Monitor is not needed to watch a normal scan.
- Capture errors stop the preview loop and require the usual fault recovery; no old frame is presented as a new capture.

### Saved scan

- One session ID and sequential point IDs connect motor -> Qt -> backend -> saved spectrum -> motor ACK.
- The backend retains the legacy five-frame buffer/capture window, reads only the first available complete frame (2048 pixels) for the stationary point, then stops capture and releases the buffer before reporting success. It retains the old initial 1.25 s wait before querying status (scaled up for longer exposure/cycle settings), then checks readiness with an 8 s overall polling deadline.
- v2.2: ONLY for `USB2_getCaptureStatus`, rc=1 means continue polling the SAME capture every 50 ms until ready or timeout. Inspection of the installed DLL's IL confirmed it returns 1 when captured-minus-delivered frame count is <=0. Other SDK calls still treat nonzero results as errors; status errors other than 1 still abort. Every query clears the output frame/index fields. A pending status is never a successful point ACK. This replaces the v2.1 behaviour that aborted immediately on rc=1.
- Expected v2.2 logs: `[Capture handshake v2.2]`, optional `[Capture waiting]`, then `[Capture ready]` and Qt `[Point saved]`. Persistent absence of frames produces a timeout without advancing the motor. Why a specific physical capture never becomes ready, if that occurs, still requires device diagnosis.
- Qt waits up to 15 s for that exact capture ID, validates all pixels and atomically saves the point before sending OK.
- Invalid/truncated data, backend errors, timeouts and save failures abort the scan without advancing to the next point. No blind retry after an uncertain timeout. Fix the underlying problem, confirm laser OFF and restart the app before retrying a faulted backend.
- Pause during exposure lets the current capture finish and save, but withholds motor permission until Resume. Resume retains the log and does not reacquire a saved point.
- A stopped/failed scan cannot be resumed as if complete. Start begins a NEW run with the original homing sequence; inspect clearance first.
- Stop requests motor cancellation; motion-message waits are bounded at 120 s, missing Qt heartbeat at 30 s. Manual Pause sends fresh heartbeats while the app is responsive.
- Qt allows up to 8 s for the three motor connections to close after cancellation before considering forced process termination (previously 2 s).
- Stop/scan failure does NOT turn off the laser. Use Laser OFF and confirm readback. Software controls are not a substitute for the instrument's hardware safety controls.
- Capture Dark is mutually exclusive with scanning and cannot reuse an old spectrum as a new dark capture.

## Data

New Capture Dark measurements are saved in `dark_measurements/` beside the executable, as `dark_yyyyMMdd_HHmmss_zzz.txt`. The folder is created automatically, filenames avoid overwriting, and the existing index/Raman-shift/intensity TXT columns are unchanged. Existing dark files in the executable directory are not moved. Write failures report failure instead of success.

New scans use the executable's `scan_sessions/1d_x_scan_yyyyMMdd_HHmmss_zzz/`, `1d_y_scan_yyyyMMdd_HHmmss_zzz/`, or `2d_scan_yyyyMMdd_HHmmss_zzz/` (local start time). The protocol session ID remains a UUID inside the files. Existing UUID directories are unchanged.

- `point_0001.json`, etc.: validated pixels, controller-reported XYZ steps, UTC timestamps and capture time.
- `point_0001.txt`, etc.: the same raw intensities in the legacy hexadecimal `Image Data` format, plus comment headers with mode, session, point, UTC time and XYZ steps. Both JSON and TXT must save successfully before success ACK. If TXT fails after JSON saves, JSON is retained for recovery and the scan aborts without advancing.
- `request.txt` / `response.txt`: latest protocol messages, not complete trajectory history.
- `failure.txt` and `cancel.txt`: present when relevant.

Raw backend per-ID replies remain under `scan_sessions/_backend_<id>/capture_results/` for diagnosis, including late replies discarded by Qt. `laser_log.txt` remains compatible with the existing hexadecimal Image Data parser, with stdout buffered by complete lines.
Position readings are controller-reported steps, not a new autofocus measurement. Capture duration excludes motor movement; measure successive point timestamps for overall throughput.
At 1 s exposure plus motor/USB overhead, a strict 1 point/s is not promised. The purpose is preventing false-success acknowledgements and mismatched points, not claiming 100% hardware reliability.

## Build / verification

`build_handshake.cmd` rebuilds the two local backends and the existing Qt build. Use the installed VS2019/Qt6.6.3 toolchain; close running programs before rebuilding.
`tests/` builds hardware-free backend/motor emulators and exercises the actual Qt point handler and motor handshake helper. It never imports the physical device SDK. The test-only capture timeout is shorter than production.

## Rollback

The pre-change source and executable backup is:

`C:\Users\Jeremy666\.codex\visualizations\2026\08\26\01a03fa9-2a21-7b23-875f-a65ea5fd7fc2\backup_before_handshake_20260916`

Close the controller, confirm the laser is safely off, then run `restore_previous.ps1` in that backup directory. It checks hashes, saves the current Qt source/executable in a new workspace rollback-safety folder, and restores the previous Qt source/executable. The restored Qt executable uses the untouched Downloads backends again. New scan data and new local backend copies are not deleted.
