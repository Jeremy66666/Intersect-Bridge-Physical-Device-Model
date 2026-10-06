"""Generate a separate Qt source copy using checked anchors. Originals are read only."""
import argparse
import hashlib
from pathlib import Path
from common import ROOT, atomic_json

def generate(mock=False):
    project = ROOT.parent
    source = project / "main.cpp"
    original = source.read_bytes()
    text = original.decode("utf-8-sig")
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError(f"Source changed: expected one anchor {old[:90]!r}; review before regenerating")
        text = text.replace(old, new, 1)
    replace('#include <QApplication>', '#include <QApplication>\n#include <QCheckBox>\n#include <functional>')
    replace('    friend struct HandshakeTest;', '    friend struct HandshakeTest;\n' + (ROOT / "qt_bridge_members.inc").read_text())
    anchor = '        connect(ramanStartButton, &QPushButton::clicked, this, [startScan]() { startScan("2d"); });'
    setup = '''        bridgeStartScan = startScan;
        bridgeArmed = new QCheckBox("Enable remote scans (existing travel paths)");
        controlsRow->addWidget(bridgeArmed);
        QDir().mkpath(bridgeRoot() + "/requests");
        QDir().mkpath(bridgeRoot() + "/responses");
        QDir().mkpath(bridgeRoot() + "/journal");
        auto* bridgeTimer = new QTimer(this);
        connect(bridgeTimer, &QTimer::timeout, this, [this](){ bridgeTick(); });
        bridgeTimer->start(100);
'''
    replace(anchor, setup + anchor)
    replace('            currentScanMode = mode;', '            bridgeScanState = "running";\n            currentScanMode = mode;')
    replace('            stopFocusMonitor();\n            stopMotorScan();', '            bridgeScanState = "cancelled";\n            stopFocusMonitor();\n            stopMotorScan();')
    replace('    void failScan(const QString& reason) {', '    void failScan(const QString& reason) {\n        bridgeScanState = "failed";')
    replace('                    stopRamanDataCollection();\n                });',
            '                    if (bridgeScanState == "running") bridgeScanState = (code == 0 && status == QProcess::NormalExit) ? "completed" : "failed";\n                    stopRamanDataCollection();\n                });')
    replace('                ramanOutput->append("[Error] Could not start: " + ramanExe);',
            '                bridgeScanState = "failed";\n                ramanOutput->append("[Error] Could not start: " + ramanExe);')
    if not mock:
        runtime = project / "build/Desktop_Qt_6_6_3_MSVC2019_64bit-Debug"
        replace('QCoreApplication::applicationDirPath() + "/handshake_v2/backend/SampleApl.exe"',
                'QString::fromUtf8(R"(' + (runtime / 'handshake_v2/backend/SampleApl.exe').as_posix() + ')")')
        replace('QLockFile controllerLock(QCoreApplication::applicationDirPath() + "/controller.lock");',
                'QLockFile controllerLock(QString::fromUtf8(R"(' + (runtime / 'controller.lock').as_posix() + ')"));')
    else:
        # Mock mode uses test executables only, while retaining an actual Qt application main().
        replace('#ifndef HANDSHAKE_TESTING\nint main(', '#if 1\nint main(')
        text = '#define HANDSHAKE_TESTING\n' + text
        replace('        bridgeTimer->start(100);', '        bridgeTimer->start(100);\n        laserOn = true; // mock only\n        bridgeArmed->setChecked(true);')
    folder = ROOT / ("generated_mock" if mock else "generated")
    folder.mkdir(exist_ok=True)
    (folder / "main.cpp").write_text(text, encoding="utf-8")
    (folder / "acquisition_protocol.h").write_bytes((project / "acquisition_protocol.h").read_bytes())
    cmake = '''cmake_minimum_required(VERSION 3.16)
project(RamanIntersect LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets Charts)
add_executable(raman_intersect main.cpp)
target_compile_options(raman_intersect PRIVATE /utf-8)
target_link_libraries(raman_intersect PRIVATE Qt6::Widgets Qt6::Charts)
set_target_properties(raman_intersect PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
'''
    if mock:
        tests = (project / "tests").as_posix()
        cmake += f'''add_executable(mock_backend "{tests}/mock_backend.cpp")
target_link_libraries(mock_backend PRIVATE Qt6::Core)
set_target_properties(mock_backend PROPERTIES OUTPUT_NAME SampleApl RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin/handshake_v2/backend")
add_executable(mock_motor "{tests}/mock_motor.cpp")
set_target_properties(mock_motor PROPERTIES OUTPUT_NAME RamanAutofocus RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin/handshake_v2/motor")
'''
    (folder / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    atomic_json(folder / "source_manifest.json", {"source":str(source), "sha256":hashlib.sha256(original).hexdigest(), "mock":mock})
    print(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    generate(parser.parse_args().mock)
