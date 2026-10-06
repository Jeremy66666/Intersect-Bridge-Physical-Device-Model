// Deliberately no motor SDK: exercises the REAL handshake helper, never a device.
#include "../hardware/motor/scan_handshake.h"
#include "../hardware/motor/line_scan_plan.h"
int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) return 2;
    const std::string mode = argc == 4 ? argv[3] : "2d";
    if (mode != "2d" && mode != "1d-x" && mode != "1d-y") return 2;
    const auto positions = lineScanPositions();
    const std::string dir = argv[1], run = argv[2];
    const unsigned count = mode == "2d" ? 3 : static_cast<unsigned>(positions.size());
    for (unsigned i = 1; i <= count; ++i) {
        const int x = mode == "1d-x" ? positions[i-1] : 69400;
        const int y = mode == "1d-y" ? positions[i-1] : 72870;
        if (!ScanHandshake::publish(dir + "/request.txt", "STEP " + run + " " +
                std::to_string(i) + " " + std::to_string(x) + " " + std::to_string(y) + " 131860\n")) return 3;
        if (!ScanHandshake::waitForAck(dir, run, i, 3000)) {
            Sleep(2500); // Model cleanup longer than the old 2-second Qt limit.
            return 4;
        }
        ScanHandshake::publish(dir + "/advanced.txt", std::to_string(i));
        Sleep(80); // Leave the same request visible to test duplicate handling.
    }
    return 0;
}
