#pragma once
#include <windows.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <thread>

// One outstanding point per session. Files are published atomically.
namespace ScanHandshake {
inline bool publish(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    { std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
      out << text; out.flush(); if (!out) return false; }
    for (int attempt = 0; attempt < 20; ++attempt) {
        if (MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
        Sleep(10);
    }
    return false;
}
inline bool cancelled(const std::string& dir) {
    return GetFileAttributesA((dir + "/cancel.txt").c_str()) != INVALID_FILE_ATTRIBUTES;
}
inline bool waitForAck(const std::string& dir, const std::string& run, unsigned point,
                       int timeoutMs = 30000) {
    auto lastContact = std::chrono::steady_clock::now();
    std::string previousHeartbeat;
    while (!cancelled(dir)) {
        std::ifstream in(dir + "/response.txt");
        std::string text; std::getline(in, text); in.close();
        std::istringstream row(text);
        std::string tag, actualRun, state, heartbeat; unsigned actualPoint = 0;
        if (row >> tag >> actualRun >> actualPoint >> state >> heartbeat) {
            if (tag == "ACK" && actualRun == run && actualPoint == point) {
                if (state == "OK") return true;
                if (state == "FAIL") return false;
                if ((state == "PAUSED" || state == "HOLD") && heartbeat != previousHeartbeat) {
                    previousHeartbeat = heartbeat;
                    lastContact = std::chrono::steady_clock::now();
                }
            }
        }
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - lastContact).count() >= timeoutMs) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}
}
