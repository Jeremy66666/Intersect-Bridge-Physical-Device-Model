#define wmain motor_entry
#include "../hardware/motor/RamanAutofocusTest.cpp"
#undef wmain
#include <QCoreApplication>
#include <QTemporaryDir>
#include <atomic>

bool runPath(const wchar_t* mode, bool badLimit) {
    StubMotor::reset();
    if (badLimit) StubMotor::maximum = 694000;
    pointId = 0; sessionDir.clear(); sessionId.clear();
    QTemporaryDir tmp;
    const std::string dir = tmp.path().toStdString();
    std::wstring wdir = tmp.path().toStdWString(), wrun = L"path_test", wmode = mode, program = L"test";
    wchar_t* args[] = {program.data(), wdir.data(), wrun.data(), wmode.data()};
    std::atomic<bool> done{false};
    std::thread qt([&] {
        while (!done) {
            std::ifstream in(dir + "/request.txt");
            std::string text; std::getline(in,text); in.close();
            std::istringstream row(text); std::string tag, run; unsigned id = 0;
            if (row >> tag >> run >> id) ScanHandshake::publish(dir + "/response.txt",
                "ACK " + run + " " + std::to_string(id) + " OK 1\n");
            Sleep(10);
        }
    });
    const int result = motor_entry(4,args);
    done = true; qt.join();
    if (badLimit) return result != 0 && StubMotor::moves.empty() && StubMotor::homes.empty();
    const std::string selected = wmode == L"1d-x" ? "27266528" : "27266445";
    if (result != 0 || pointId != 13 || StubMotor::moves.size() != 13 ||
        StubMotor::homes.size() != 1 || StubMotor::homes[selected] != 1 ||
        StubMotor::positions["27266446"] != 131860) return false;
    for (int i = 0; i < 13; ++i)
        if (StubMotor::moves[i].first != selected || StubMotor::moves[i].second != i*69400) return false;
    return StubMotor::positions[selected == "27266528" ? "27266445" : "27266528"] ==
           (selected == "27266528" ? 72870 : 69400);
}
int main(int argc, char** argv) {
    QCoreApplication app(argc,argv);
    if (!runPath(L"1d-x",false) || !runPath(L"1d-y",false) || !runPath(L"1d-y",true)) return 1;
    std::cout << "PASS actual motor branches: selected-axis-only homing/moves, 13 captures, other axes fixed, invalid limits rejected before motion." << std::endl;
    return 0;
}
