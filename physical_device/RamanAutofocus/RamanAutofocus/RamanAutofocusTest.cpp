#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <windows.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>   // <== NEW: for fabs / round
#ifdef MOTOR_PATH_TESTING
#include "../../tests/motor_sdk_stub.h"
#else
#include "../dlls/Thorlabs.MotionControl.KCube.DCServo.h"
#endif
#include "scan_handshake.h"
#include "line_scan_plan.h"

static std::string sessionDir, sessionId;
static unsigned pointId = 0;


// ======================= Z axis parameters =======================
// Z axis basic settings: offsets, range, step, serial, steps/mm.
const double Z_OFFSET_AFTER_HOME = 4.10;   // go to this Z (mm) after homing
const double Z_RANGE             = 0.4;   // Z scan range (mm)
const double Z_STEP              = 0.1;   // Z step (mm)
const int    MOTOR_SERIAL_NO_Z   = 27266446; // Z device serial number
const double STEPS_PER_MM_Z      = 34700;    // Z steps per mm

// ======================= X/Y axis parameters =======================
// X/Y axis basic settings: start offsets, range, step, serials, steps/mm.
const double X_OFFSET_AFTER_HOME = 2.0;      // X start (mm from home)
const double X_RANGE             = 1.0;      // X scan range (mm)
const double X_STEP              = 0.1;      // X step (mm)
const int    MOTOR_SERIAL_NO_X   = 27266528; // X device serial number
const double STEPS_PER_MM_X      = 34700;    // X steps per mm

const double Y_OFFSET_AFTER_HOME = 2.0;      // Y start (mm from home)
const double Y_RANGE             = 1.0;      // Y scan range (mm)
const double Y_STEP              = 0.1;      // Y step (mm)
const int    MOTOR_SERIAL_NO_Y   = 27266445; // Y device serial number
const double STEPS_PER_MM_Y      = 34700;    // Y steps per mm

// ======================= Common timing parameters =======================
// Timing for polling, initial settle, and per-move extra wait.
const int POLLING_INTERVAL_MS = 200; // device polling interval
const int INIT_DELAY_MS       = 100; // wait after starting polling
const int MOVE_DELAY_MS       = 10;  // wait after each move + Qt trigger

// ======================= Helpers: unit conversion =======================
// Convert mm to controller steps. No offset is applied here.
int relativeStepsZ(double mm) { return static_cast<int>(mm * STEPS_PER_MM_Z + 0.5); }
int relativeStepsX(double mm) { return static_cast<int>(mm * STEPS_PER_MM_X + 0.5); }
int relativeStepsY(double mm) { return static_cast<int>(mm * STEPS_PER_MM_Y + 0.5); }

// ======================= Helpers: wait (specific message) =======================
// Block until a specific (type,id) message is received from the device.
static bool waitForMessage(const char* serialNo, WORD expType, WORD expId) {
    WORD mt=0, mi=0; DWORD md=0;
    const auto started = std::chrono::steady_clock::now();
    while (!ScanHandshake::cancelled(sessionDir)) {
        if (CC_MessageQueueSize(serialNo) > 0 && CC_GetNextMessage(serialNo, &mt, &mi, &md)
            && mt == expType && mi == expId) return true;
        if (std::chrono::steady_clock::now() - started > std::chrono::seconds(120)) break;
        Sleep(20);
    }
    CC_StopImmediate(serialNo);
    std::cerr << "[Error] Movement cancelled or timed out; scan aborted." << std::endl;
    return false;
}

// ======================= Robust homing for any axis =======================
// Do homing, then wait for position to stabilize. If final position is far from 0,
// retry once. Return true on success and output the final "home" steps.
bool homeAxisAndConfirm(const char* serial, const char* name, int& homePosOut,
                        int stableWindowSteps = 5, int maxDriftAccept = 1000,
                        int settleMs = 500, int checkIters = 10, int checkIntervalMs = 100)
{
    // Inner helper: run homing once and check stability.
    auto doHome = [&](bool announce) -> bool {
        if (ScanHandshake::cancelled(sessionDir)) return false;
        if (announce) std::cout << "Homing " << name << "..." << std::endl;
        CC_ClearMessageQueue(serial);       // drop old messages
        if (CC_Home(serial) != 0) {         // start homing
            std::cerr << "[Error] CC_Home(" << name << ") invoke failed" << std::endl;
            return false;
        }
        if (!waitForMessage(serial, 2, 0)) return false; // (2,0) = Homed
        if (announce) std::cout << name << " homing message received, stabilizing..." << std::endl;
        Sleep(settleMs);                    // small settle time after homed

        // Read position multiple times until change is small.
        int pos = CC_GetPosition(serial);
        int last = pos;
        for (int i=0; i<checkIters; ++i) {
            Sleep(checkIntervalMs);
            pos = CC_GetPosition(serial);
            if (abs(pos - last) <= stableWindowSteps) break;
            last = pos;
        }
        homePosOut = pos;
        std::cout << name << " final position after homing = " << pos << " steps" << std::endl;
        return true;
    };

    // First homing try.
    if (!doHome(true)) return false;

    // If far from 0, try one more time. If still bad, fail.
    if (abs(homePosOut) > maxDriftAccept) {
        std::cerr << "[Warn] " << name << " homed but far from zero (" << homePosOut
                  << " steps). Retrying homing once..." << std::endl;
        if (!doHome(false)) return false;
        if (abs(homePosOut) > maxDriftAccept) {
            std::cerr << "[Error] " << name << " still far from zero after retry ("
                      << homePosOut << " steps). Please check limit switch / parameters." << std::endl;
            return false;
        }
    }
    return true;
}

// ======================= Helpers: move and wait =======================
// Move to an absolute step position and wait for MoveCompleted.
bool moveMotorToPosition(const char* serialNo, int position) {
    if (ScanHandshake::cancelled(sessionDir)) return false;
    CC_ClearMessageQueue(serialNo);               // drop old messages
    if (CC_MoveToPosition(serialNo, position) != 0) {
        std::cerr << "Failed to start movement to position " << position << std::endl;
        return false;
    }
    return waitForMessage(serialNo, 2, 1);       // (2,1) = MoveCompleted
}

// ======================= Qt trigger handshake =======================
// A stale reply, failed acquisition, or missing Qt heartbeat never advances a point.
bool triggerAndWaitQt() {
    ++pointId;
    std::ostringstream request;
    request << "STEP " << sessionId << ' ' << pointId << ' '
            << CC_GetPosition("27266528") << ' ' << CC_GetPosition("27266445") << ' '
            << CC_GetPosition("27266446") << '\n';
    const bool ok = ScanHandshake::publish(sessionDir + "/request.txt", request.str()) &&
                    ScanHandshake::waitForAck(sessionDir, sessionId, pointId);
    std::cout << "[Point " << pointId << "] " << (ok ? "saved, advance" : "FAILED/cancelled, do not advance") << std::endl;
    return ok;
}

// ======================= Helpers: formatted move + trigger (X/Y only) =======================
// Move X to a mm target (from home), trigger Qt, and wait a short time.
bool moveX_to_mm_and_trigger(const char* serialX, int homeX, double mmAbs) {
    int pos = homeX + relativeStepsX(mmAbs);
    std::cout << "X -> " << std::fixed << std::setprecision(4) << mmAbs
              << " mm (motor pos: " << pos << ") ... ";
    if (!moveMotorToPosition(serialX, pos)) { std::cerr << "Failed.\n"; return false; }
    std::cout << "OK" << std::endl;
    if (!triggerAndWaitQt()) return false;
    Sleep(MOVE_DELAY_MS);
    return true;
}

// Move Y to a mm target (from home), trigger Qt, and wait a short time.
bool moveY_to_mm_and_trigger(const char* serialY, int homeY, double mmAbs) {
    int pos = homeY + relativeStepsY(mmAbs);
    std::cout << "Y -> " << std::fixed << std::setprecision(4) << mmAbs
              << " mm (motor pos: " << pos << ") ... ";
    if (!moveMotorToPosition(serialY, pos)) { std::cerr << "Failed.\n"; return false; }
    std::cout << "OK" << std::endl;
    if (!triggerAndWaitQt()) return false;
    Sleep(MOVE_DELAY_MS);
    return true;
}

// ======================= Helpers: scan utilities (inclusive endpoints) =======================
// NOTE: 这里改成严格“每步 0.1 mm 的相对移动”。
// start / end 只用来决定：方向 (+/-) 和一共走几步。
// 真正移动用 CC_MoveRelative，每次 ±0.1 mm，对升 / 降方向一视同仁。

// Scan X from start to end with step X_STEP. Each point triggers Qt.
bool scanX(const char* serialX, int /*homeX*/, double start, double end) {
    if (ScanHandshake::cancelled(sessionDir)) return false;
    double distance = std::fabs(end - start);
    int steps = static_cast<int>(std::round(distance / X_STEP));
    if (steps <= 0) return true;

    int deltaSteps = relativeStepsX(X_STEP);
    double dir = (end >= start) ? +1.0 : -1.0;
    if (dir < 0) deltaSteps = -deltaSteps;

    double curMm = start;
    for (int i = 0; i < steps; ++i) {
        if (ScanHandshake::cancelled(sessionDir)) return false;
        curMm += dir * X_STEP;

        int currentPos = CC_GetPosition(serialX);
        int targetPos  = currentPos + deltaSteps;

        std::cout << "X -> " << std::fixed << std::setprecision(4) << curMm
                  << " mm (motor pos: " << targetPos << ") ... ";

        CC_ClearMessageQueue(serialX);
        if (CC_MoveRelative(serialX, deltaSteps) != 0) {
            std::cerr << "Failed.\n";
            return false;
        }
        if (!waitForMessage(serialX, 2, 1)) return false; // MoveCompleted
        std::cout << "OK" << std::endl;

        if (!triggerAndWaitQt()) return false;
        Sleep(MOVE_DELAY_MS);
    }
    return true;
}

// Scan Y from start to end with step Y_STEP. Each point triggers Qt.
bool scanY(const char* serialY, int /*homeY*/, double start, double end) {
    if (ScanHandshake::cancelled(sessionDir)) return false;
    double distance = std::fabs(end - start);
    int steps = static_cast<int>(std::round(distance / Y_STEP));
    if (steps <= 0) return true;

    int deltaSteps = relativeStepsY(Y_STEP);
    double dir = (end >= start) ? +1.0 : -1.0;
    if (dir < 0) deltaSteps = -deltaSteps;

    double curMm = start;
    for (int i = 0; i < steps; ++i) {
        if (ScanHandshake::cancelled(sessionDir)) return false;
        curMm += dir * Y_STEP;

        int currentPos = CC_GetPosition(serialY);
        int targetPos  = currentPos + deltaSteps;

        std::cout << "Y -> " << std::fixed << std::setprecision(4) << curMm
                  << " mm (motor pos: " << targetPos << ") ... ";

        CC_ClearMessageQueue(serialY);
        if (CC_MoveRelative(serialY, deltaSteps) != 0) {
            std::cerr << "Failed.\n";
            return false;
        }
        if (!waitForMessage(serialY, 2, 1)) return false; // MoveCompleted
        std::cout << "OK" << std::endl;

        if (!triggerAndWaitQt()) return false;
        Sleep(MOVE_DELAY_MS);
    }
    return true;
}

// ======================= Main sequence =======================
// Open devices, start polling, home X/Y/Z (robust), then do Z scan and X/Y scans.
int __cdecl wmain(int argc, wchar_t* argv[]) {
    if (argc != 3 && argc != 4) { std::cerr << "Start this version from the updated Qt controller." << std::endl; return 2; }
    const bool lineX = argc == 4 && std::wstring(argv[3]) == L"1d-x";
    const bool lineY = argc == 4 && std::wstring(argv[3]) == L"1d-y";
    if (argc == 4 && !lineX && !lineY) { std::cerr << "Invalid scan mode." << std::endl; return 2; }
    // This installation uses an ASCII workspace path; reject lossy conversion.
    for (int a = 1; a <= 2; ++a) {
        for (const wchar_t* p = argv[a]; *p; ++p) {
            if (*p > 127) { std::cerr << "Handshake path must be ASCII." << std::endl; return 2; }
            (a == 1 ? sessionDir : sessionId).push_back(static_cast<char>(*p));
        }
    }
    if (GetFileAttributesA(sessionDir.c_str()) == INVALID_FILE_ATTRIBUTES) return 2;
    int scanResult = 1;
    char currentDir[MAX_PATH]; GetCurrentDirectoryA(MAX_PATH, currentDir);
    std::cout << "Current working directory: " << currentDir << std::endl;

    // Build device list (USB/driver discovery). Exit on failure.
    std::cout << "Building device list..." << std::endl;
    if (TLI_BuildDeviceList() != 0) {
        std::cerr << "Failed to build device list. Please check connections." << std::endl;
        return 1;
    }

    // Open X, Y, Z devices in order. If any fails, close what was opened and exit.
    char serialX[16]; sprintf_s(serialX, "%d", MOTOR_SERIAL_NO_X);
    char serialY[16]; sprintf_s(serialY, "%d", MOTOR_SERIAL_NO_Y);
    char serialZ[16]; sprintf_s(serialZ, "%d", MOTOR_SERIAL_NO_Z);

    std::cout << "Opening X device " << serialX << " ..." << std::endl;
    if (CC_Open(serialX) != 0) {
        std::cerr << "Failed to connect to X device " << serialX << std::endl;
        return 1;
    }
    std::cout << "Opening Y device " << serialY << " ..." << std::endl;
    if (CC_Open(serialY) != 0) {
        std::cerr << "Failed to connect to Y device " << serialY << std::endl;
        CC_Close(serialX);
        return 1;
    }
    std::cout << "Opening Z device " << serialZ << " ..." << std::endl;
    if (CC_Open(serialZ) != 0) {
        std::cerr << "Failed to connect to Z device " << serialZ << std::endl;
        CC_Close(serialX);
        CC_Close(serialY);
        return 1;
    }

    // Start polling on all three, then wait a short time for state to settle.
    CC_StartPolling(serialX, POLLING_INTERVAL_MS);
    CC_StartPolling(serialY, POLLING_INTERVAL_MS);
    CC_StartPolling(serialZ, POLLING_INTERVAL_MS);
    Sleep(INIT_DELAY_MS);

    // Declare before cleanup jumps; 1D must bypass ALL legacy XYZ homing/Z motion.
    int homeX = 0, homeY = 0, homeZ = 0;
    if (lineX || lineY) {
        const char* selected = lineX ? serialX : serialY;
        const char* name = lineX ? "X" : "Y";
        const auto points = lineScanPositions();
        if (CC_RequestSettings(selected) != 0) {
            std::cerr << "[1D] Cannot request travel limits; no movement." << std::endl;
            goto ALL_CLEANUP;
        }
        Sleep(500);
        const int minimum = CC_GetStageAxisMinPos(selected);
        const int maximum = CC_GetStageAxisMaxPos(selected);
        if (!lineScanWithinLimits(points, minimum, maximum)) {
            std::cerr << "[1D] 0..24 mm exceeds/has unavailable controller limits: "
                      << minimum << ".." << maximum << " steps. No movement." << std::endl;
            goto ALL_CLEANUP;
        }
        std::cout << "[1D] " << name << ": home selected axis, then 0..24 mm / 2 mm / 13 spectra. Other axes remain fixed." << std::endl;
        int selectedHome = 0;
        if (!homeAxisAndConfirm(selected, name, selectedHome)) goto ALL_CLEANUP;
        for (int target : points) {
            if (!moveMotorToPosition(selected, target)) goto ALL_CLEANUP;
            // Move-complete can represent a clipped move: check the actual endpoint.
            Sleep(POLLING_INTERVAL_MS);
            if (std::abs(CC_GetPosition(selected) - target) > 100) {
                std::cerr << "[1D] Target not reached; abort without acquiring at wrong position." << std::endl;
                goto ALL_CLEANUP;
            }
            std::cout << "[1D] " << name << " at " << target / 34700.0 << " mm" << std::endl;
            if (!triggerAndWaitQt()) goto ALL_CLEANUP;
        }
        scanResult = 0;
        goto ALL_CLEANUP;
    }

    // ===== 在这里关闭 backlash =====
    CC_SetBacklash(serialX, 0);  // X 轴 backlash 设为 0 step
    CC_SetBacklash(serialY, 0);  // Y 轴 backlash 设为 0 step
    CC_SetBacklash(serialZ, 0);  // Z 轴 backlash 设为 0 step

    // Home X, then Y, then Z using the robust homing helper. Abort on failure.
    if (!homeAxisAndConfirm(serialX, "X", homeX)) {
        std::cerr << "[Fatal] X homing failed." << std::endl;
        goto ALL_CLEANUP;
    }
    if (!homeAxisAndConfirm(serialY, "Y", homeY)) {
        std::cerr << "[Fatal] Y homing failed." << std::endl;
        goto ALL_CLEANUP;
    }
    if (!homeAxisAndConfirm(serialZ, "Z", homeZ)) {
        std::cerr << "[Fatal] Z homing failed." << std::endl;
        goto ALL_CLEANUP;
    }
    std::cout << "Homed positions: X=" << homeX << ", Y=" << homeY << ", Z=" << homeZ << std::endl;

    // ===== After Z homing: move to 20 mm, then back to 7.1 mm =====
    {
        double highZ = 20.0;
        int posHighZ = homeZ + relativeStepsZ(highZ);
        std::cout << "Move Z to " << highZ << " mm (motor pos: " << posHighZ << ")..." << std::endl;
        if (!moveMotorToPosition(serialZ, posHighZ)) {
            std::cerr << "Z move to 20 mm failed." << std::endl;
            goto ALL_CLEANUP;
        }

        double backZ = Z_OFFSET_AFTER_HOME; // configured Z offset
        int posBackZ = homeZ + relativeStepsZ(backZ);
        std::cout << "Move Z back to " << backZ << " mm (motor pos: " << posBackZ << ")..." << std::endl;
        if (!moveMotorToPosition(serialZ, posBackZ)) {
            std::cerr << "Z move back to 7.1 mm failed." << std::endl;
            goto ALL_CLEANUP;
        }
    }

    // -------- Z scan block --------
    // Move Z to start offset, then scan Z from offset to offset+range by Z_STEP.
    {
        int startPosZ = homeZ + relativeStepsZ(Z_OFFSET_AFTER_HOME);
        std::cout << "Move Z up " << Z_OFFSET_AFTER_HOME << "mm to startPos: " << startPosZ << std::endl;
        if (!moveMotorToPosition(serialZ, startPosZ)) {
            std::cerr << "Z move to start failed." << std::endl;
            goto ALL_CLEANUP;
        }

        std::cout << "At (home + " << Z_OFFSET_AFTER_HOME << "mm), start Z scan..." << std::endl;

        std::vector<double> zList;
        // 每次只移动 0.1 mm，当前位置已在 7.1 mm，从 7.2 mm 开始扫
        for (double z = Z_STEP; z <= Z_RANGE + 1e-8; z += Z_STEP) zList.push_back(z);

        std::cout << "\n---- Moving Z motor ----\n";
        std::cout << "Step size: " << Z_STEP << " mm, total steps: " << zList.size() << std::endl;

        for (size_t i = 0; i < zList.size(); ++i) {
            double zPosRel = zList[i]; // relative to start offset
            int motorPos = startPosZ + relativeStepsZ(zPosRel);
            std::cout << "Moving to Z_offset = " << std::setw(7) << std::fixed << std::setprecision(4)
                      << (Z_OFFSET_AFTER_HOME + zPosRel) << " mm (motor pos: " << motorPos << ") ... ";
            if (!moveMotorToPosition(serialZ, motorPos)) {
                std::cerr << "Failed to move, aborting Z scan." << std::endl;
                goto ALL_CLEANUP;
            }
            std::cout << "OK" << std::endl;
            if (!triggerAndWaitQt()) goto ALL_CLEANUP; // acquire at this unchanged Z
            Sleep(MOVE_DELAY_MS); // small margin for acquisition
        }

        // Go back to (offset + range - 0.2) as the final Z position.
        double targetBackPos = Z_OFFSET_AFTER_HOME + Z_RANGE - 0.2; // fixed XY working height
        int backMotorPos = homeZ + relativeStepsZ(targetBackPos);
        std::cout << "Moving to Z = " << targetBackPos << " mm (motor pos: " << backMotorPos << ")... ";
        if (!moveMotorToPosition(serialZ, backMotorPos)) {
            std::cerr << "Failed to move to Z = " << targetBackPos << " mm!" << std::endl;
            goto ALL_CLEANUP;
        } else {
            std::cout << "OK" << std::endl;
        }
    }

    // -------- X/Y scan block --------
    // Move X/Y to initial 2.0 mm (no trigger), then run the interleaved scans.
    {
        int startX = homeX + relativeStepsX(X_OFFSET_AFTER_HOME);
        int startY = homeY + relativeStepsY(Y_OFFSET_AFTER_HOME);
        std::cout << "Move X to start (+" << X_OFFSET_AFTER_HOME << " mm) pos=" << startX << " ... ";
        if (!moveMotorToPosition(serialX, startX)) { std::cerr << "Failed.\n"; goto ALL_CLEANUP; }
        std::cout << "OK\n";
        std::cout << "Move Y to start (+" << Y_OFFSET_AFTER_HOME << " mm) pos=" << startY << " ... ";
        if (!moveMotorToPosition(serialY, startY)) { std::cerr << "Failed.\n"; goto ALL_CLEANUP; }
        std::cout << "OK\n";

        std::cout << "\n---- Sequence begin (X/Y) ----\n";

        // Keep the original interleaved order. Each step triggers Qt and waits.
        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.0, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.9, 3.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.0 -> 2.1\n";
        if (!scanX(serialX, homeX, 2.0, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 3.0, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.9, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.1 -> 2.2\n";
        if (!scanX(serialX, homeX, 2.1, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.0, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.9, 3.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.2 -> 2.3\n";
        if (!scanX(serialX, homeX, 2.2, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 3.0, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.9, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.3 -> 2.4\n";
        if (!scanX(serialX, homeX, 2.3, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.0, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.9, 3.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.4 -> 2.5\n";
        if (!scanX(serialX, homeX, 2.4, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 3.0, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.9, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.5 -> 2.6\n";
        if (!scanX(serialX, homeX, 2.5, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.0, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.9, 3.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.6 -> 2.7\n";
        if (!scanX(serialX, homeX, 2.6, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 3.0, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.9, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.7 -> 2.8\n";
        if (!scanX(serialX, homeX, 2.7, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.0, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.9, 3.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.8 -> 2.9\n";
        if (!scanX(serialX, homeX, 2.8, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 3.0, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.9, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 3.0 -> 2.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning X: 2.9 -> 3.0\n";
        if (!scanX(serialX, homeX, 2.9, 3.0)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.0, 2.1)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.1, 2.2)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.2, 2.3)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.3, 2.4)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.4, 2.5)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.5, 2.6)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.6, 2.7)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.7, 2.8)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.8, 2.9)) goto ALL_CLEANUP;

        std::cout << "\nScanning Y: 2.0 -> 3.0\n";
        if (!scanY(serialY, homeY, 2.9, 3.0)) goto ALL_CLEANUP;

        std::cout << "\n---- Sequence end (X/Y) ----\n";
    }
    scanResult = 0;

ALL_CLEANUP:
    // Stop polling and close all devices. Always do this before exit.
    CC_StopPolling(serialX);
    CC_StopPolling(serialY);
    CC_StopPolling(serialZ);
    CC_Close(serialX);
    CC_Close(serialY);
    CC_Close(serialZ);

    std::cout << "All device connections closed." << std::endl;
    std::cout << (scanResult == 0 ? "Scan complete." : "Scan aborted; no further point acquired.") << std::endl;
    return scanResult;
}

