# INTERSECT Bridge + Physical Device + Model

This project connects a portable Raman instrument, a three-axis motor, and a PCA-SVM model with INTERSECT.

A user can send a scan request, check the system, and read the analysis results through the INTERSECT Python SDK.

## What each part does

| Part | Job |
| --- | --- |
| Portable Raman instrument | Uses a laser and records a Raman spectrum. |
| Three-axis motor | Moves the sample along X, Y, and Z. |
| Qt control window | Controls the devices and saves each scan point. |
| PCA-SVM model | Reads the saved spectra and gives a class label and scores. |
| INTERSECT bridge | Receives requests and sends back status and results. |
| Docker | Runs RabbitMQ, the message broker that passes messages between the client and the service. |
| Launcher | Opens the local tools from one window after setup. |

Docker runs the message broker. The Qt program and Python service run on the Windows computer connected to the devices.

## Current workflow

```text
Client request
      |
INTERSECT SDK + RabbitMQ
      |
Python service
      |
Qt control window <--> Raman instrument + motor
      |
Saved spectra --> PCA-SVM --> Results returned to the client
```

The Python service and Qt use local JSON files to communicate. They must run on the same device computer.

**PCA-SVM is connected to the automatic INTERSECT workflow.** CNN and Transformer scripts are also in this repository, but they are not connected to this automatic workflow yet. The CNN button in Qt is a separate local action.

## Important: setup is still needed

This repository keeps the original program code unchanged. It is a source copy, not a ready-to-run package for every computer.

Some files still contain paths from the original computer. New users need to set up their own paths, software, and devices.

The public repository does not include:

- The real `config.json` and its password.
- The original scan data and dark spectra.
- Vendor DLLs, library files, and the Raman vendor sample source.
- The built EXE files or the original Python environment.

To run real hardware, you need the correct vendor drivers and device programs. The Raman backend must support this project's scan-handshake messages. A standard vendor program may not have this support. Ask the project owner which device files are needed, and follow the vendor's rules for sharing them.

## Main folders

| Folder or file | Contents |
| --- | --- |
| `intersect_bridge/` | Python service, client, launcher, Qt bridge generator, and Docker settings. |
| `main.cpp` | Original Qt control program. |
| `hardware/motor/` | Local motor source. |
| `physical_device/RamanAutofocus/` | Copy of the external motor project used on the original computer. |
| `build/Desktop_Qt_6_6_3_MSVC2019_64bit-Debug/` | Saved model files and model scripts, not a complete build folder. |
| `model_replacements_20260921/` | Other CNN, SVM, and Transformer scripts. |
| `tests/` | Test tools, including simulated devices. |
| `third_party/intersect-python-sdk/` | Official INTERSECT Python SDK source, version 0.9.3. |
| `third_party/intersect-architecture/` | Official INTERSECT architecture documents. |
| `DEPENDENCY_SNAPSHOT.json` | Software versions from the original computer. |
| `COPY_MANIFEST.json` | Original file locations and SHA-256 checks for the copied files. |

## 1. Get the code and install the software

The original setup used Windows, 64-bit Python 3.12, Qt 6.6.3 with Widgets and Charts, and Visual Studio 2019 with C++ build tools.

Install these tools and Docker Desktop before building the real-device program. You also need the Raman and Thorlabs motor drivers.

Open PowerShell in the folder where you want to keep the project:

```powershell
git clone https://github.com/Jeremy66666/Intersect-Bridge-Physical-Device-Model.git
Set-Location .\Intersect-Bridge-Physical-Device-Model\intersect_bridge
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All commands below run from the `intersect_bridge` folder.

## 2. Set up your files and paths

Create your own settings file:

```powershell
Copy-Item config.example.json config.json
notepad config.json
```

Edit these settings:

| Setting | What to put here |
| --- | --- |
| `model_dir` | The folder with `meta.json`, `pure_scaler_minmax.joblib`, `pure_pca_minmax.joblib`, and `pure_svm_minmax.joblib`. |
| `dark_file` | The full path to one dark-spectrum TXT file, including its file name. |
| `ipc_dir` | The local folder used by Python and Qt to pass messages. The default matches the bridge Qt program. |
| `scan_root` | The folder where Qt saves scan sessions. |
| `results_dir` | The folder where Python saves classification results. |
| `intersect.brokers[0].password` | Your message broker password. Replace `CHANGE_ME`. |

Relative paths in `config.json` start from the folder containing that file. A full Windows path can use forward slashes, for example `C:/Raman/dark/dark_example.txt`.

The example dark-spectrum path points to an old file from the original computer. That file is not in this repository. Capture your own dark spectrum with the right measurement settings, then set `dark_file` to that saved file. Restart the Python service after changing the dark file or model.

Before a real scan, also check:

- The Raman backend, motor program, CNN script, and output paths in `main.cpp`.
- The motor serial numbers and scan settings in the motor project you use.
- The Qt and Visual Studio paths in `intersect_bridge/build_qt.cmd` and `intersect_bridge/start_qt.cmd`.
- The backend and motor build paths in the root build scripts, if you use those scripts.

Put the matching Raman backend at `build/Desktop_Qt_6_6_3_MSVC2019_64bit-Debug/handshake_v2/backend/SampleApl.exe` under the repository root, with its required vendor files. Build or install the motor program and set its path in `main.cpp`.

These are manual setup steps. This README does not change those paths for you.

## 3. Build the bridge Qt program

After the device files and tool paths are ready:

```powershell
.\build_qt.cmd
```

This creates a separate Qt source copy in `generated/` and builds `build_qt/bin/raman_intersect.exe`. The generator reads the original Qt source without changing it.

If generation reports an anchor error, check the source and generator before continuing.

## 4. Start the system

Use three PowerShell windows for the manual workflow.

**Window 1: start Docker's message broker, then open Qt.** Keep Docker Desktop running. Replace `YOUR_BROKER_PASSWORD` with the same password you put in `config.json`.

```powershell
$env:RAMAN_BROKER_PASSWORD = 'YOUR_BROKER_PASSWORD'
docker compose up -d
docker compose ps
.\start_qt.cmd
```

The broker should show `Up`. It runs in the background, so it does not need its own PowerShell window.

In Qt, check the device connection, prepare the dark spectrum, and use the local laser controls. Before sending a remote scan request, turn on the laser locally when the experiment is ready and select **Enable remote scans (existing travel paths)**. Check that the existing motor path is suitable for your setup. The checkbox is off again after Qt restarts.

**Window 2: start the Python service.** Open another PowerShell window and enter your `intersect_bridge` folder first.

```powershell
.\.venv\Scripts\python.exe service.py
```

Keep this window running. You do not need a separate SVM window; the service runs PCA-SVM analysis in the background.

**Window 3: send a client request.** Open another PowerShell window and enter the same folder.

```powershell
.\.venv\Scripts\python.exe client.py status
.\.venv\Scripts\python.exe client.py start-scan --mode 1d-y --request-id experiment-001
```

The reply gives a `session_id`. This means the scan request was accepted; it does not mean the whole scan has finished. Use a new request ID for a new experiment. If a request times out, check status first and use the same request ID when retrying that request.

## 5. Read the results

Replace `SESSION_ID` with the ID returned by the start request:

```powershell
.\.venv\Scripts\python.exe client.py status
.\.venv\Scripts\python.exe client.py results SESSION_ID
```

Wait for the scan to finish and for all saved points to be analyzed. Scan completion and analysis completion can happen at different times.

New scans are normally saved in `intersect_bridge/build_qt/bin/scan_sessions/`. Analysis results are normally saved in `intersect_bridge/runtime/results/<session_id>/`.

Qt's **Capture Dark** button normally saves a TXT file in `build_qt/bin/dark_measurements/`. Set `dark_file` to the file you want the service to use.

## API inputs and outputs

| Client command | Input | Output |
| --- | --- | --- |
| `status` | No scan input. | Qt status, scan state, and analysis errors. |
| `start-scan` | Scan mode and request ID. Modes: `1d-x`, `1d-y`, or `2d`. | Session ID, session folder, and scan state. |
| `results` | Session ID; optional `--after-point` for the next page. | Point results with labels and class scores. |
| `cancel-scan` | Session ID and request ID. | Reply from the Qt scan controller. |

Example cancel request:

```powershell
.\.venv\Scripts\python.exe client.py cancel-scan SESSION_ID --request-id cancel-001
```

The remote API uses the existing scan paths. It does not provide free X/Y/Z movement or remote laser power control. Canceling a scan or stopping the Python service does not automatically turn off the laser. Use the local laser controls to turn it off.

Low-confidence results can be `UNKNOWN`. Invalid or low-quality spectra can be `REJECTED`. A class score is not a measure of classification accuracy.

## Use the launcher after setup

Install the launcher tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-launcher.txt
.\.venv\Scripts\python.exe launcher.py
```

You can also build the launcher EXE:

```powershell
.\build_launcher.cmd
```

Keep `RamanLauncher.exe` in the `intersect_bridge` folder. It still needs `config.json`, the local Python environment, the built Qt program, Docker, and the device files. It helps with startup after setup; it does not install all dependencies.

## Use a second computer

Start with the full workflow on the device computer first.

The second computer needs `client.py`, `common.py`, its own `config.json`, and `intersect-sdk==0.9.3`. The client only uses the `intersect` settings. The two computers must use the same service hierarchy and broker login.

The current Docker settings only allow connections from the device computer at `127.0.0.1:1883`. To use a second computer, set the broker's port binding to the device computer's LAN address, allow the required connection on your lab network, and set the client's broker host to that address. Network access needs this extra setup.

## Common messages

- **Cannot support MINIO:** this setup does not use MinIO. This warning alone does not mean the scan failed.
- **No event callback:** the command-line client waits for a reply, rather than listening for other services' events.
- **Missing model or dark file:** check `config.json`. Both are needed before the service starts.
- **No response:** check Docker, the Python service, Qt, and the broker settings before sending another scan request.

## INTERSECT source and licenses

This project uses [INTERSECT Python SDK 0.9.3](https://github.com/INTERSECT-SDK/python-sdk/tree/v0.9.3). The [INTERSECT architecture repository](https://github.com/ORNL/intersect-architecture) explains the design; it is not the device-control program.

The copied SDK keeps its BSD 3-Clause license. The architecture documents keep their CC BY 4.0 license. See their original `LICENSE` files under `third_party/` and the versions recorded in `THIRD_PARTY_ORIGINS.json`.

The software-version record comes from the original computer. A fresh installation on another computer still needs testing.
