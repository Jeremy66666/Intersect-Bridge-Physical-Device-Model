#include "../main.cpp"
#include "../hardware/SampleApl/capture_timing.h"
#include "../hardware/motor/scan_handshake.h"
#include "../hardware/motor/line_scan_plan.h"
#include <QTemporaryDir>
#include <functional>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static QByteArray readFile(const QString& path) {
    QFile f(path); if (!f.open(QIODevice::ReadOnly)) return {};
    return f.readAll();
}
static bool waitUntil(const std::function<bool()>& done, int timeout = 4000) {
    QElapsedTimer timer; timer.start();
    while (!done() && timer.elapsed() < timeout) {
        QCoreApplication::processEvents(); QThread::msleep(5);
    }
    return done();
}
static void spin(int ms) {
    QElapsedTimer t; t.start(); waitUntil([&]{return t.elapsed() >= ms;}, ms + 1000);
}
struct HandshakeTest {
    static void darkStorage() {
        MainWindow w;
        const QVector<int> pixels(2048, 4321);
        const QString first = w.saveDarkMeasurementToTxt(pixels);
        const QString second = w.saveDarkMeasurementToTxt(pixels);
        check(!first.isEmpty() && !second.isEmpty() && first != second, "dark saves distinct files");
        check(QFileInfo(first).absolutePath() == QCoreApplication::applicationDirPath() + "/dark_measurements", "dark subfolder");
        const auto content = readFile(first);
        check(content.contains("# Points: 2048") && content.contains("index\traman_shift_cm-1\tintensity"), "dark original format retained");
        check(content.count('\n') == 2052, "complete dark rows");
    }
    static void lineScan(const QString& mode) {
        MainWindow w;
        w.laserOn = true;
        AcquisitionProtocol::publish(w.laserProcess->workingDirectory() + "/mock_mode.txt", "ok");
        w.lineAxisCombo->setCurrentIndex(w.lineAxisCombo->findData(mode));
        click(w, "1D Scan");
        check(waitUntil([&]{return w.lastCompletedPoint == 13 && !w.scanActive;}, 20000), "13-point line scan completes");
        check(readFile(w.sessionDir + "/advanced.txt") == "13", "13 motor acknowledgements");
        check(QFileInfo(w.sessionDir).fileName().startsWith(mode == "1d-x" ? "1d_x_scan_" : "1d_y_scan_"), "readable scan directory");
        const auto combined = readFile(w.sessionDir + "/laser_log_combined.txt");
        QList<QByteArray> frames;
        for (const auto& line : combined.split('\n'))
            if (line.startsWith("Image Data")) frames.append(line);
        check(frames.size() == 13, "combined TXT contains all frames");
        check(QDir(w.sessionDir).entryList({"point_*.txt"}, QDir::Files).isEmpty(), "no individual TXT files");
        for (int i = 1; i <= 13; ++i) {
            const auto p = QJsonDocument::fromJson(readFile(w.sessionDir + QString("/point_%1.json").arg(i,4,10,QChar('0')))).object();
            check(p["scan_mode"].toString() == mode, "correct recorded mode");
            QVector<int> txtPixels;
            check(w.parseImageDataLineToInts(QString::fromUtf8(frames[i-1]), txtPixels), "combined TXT frame readable");
            check(txtPixels.size() == 2048, "TXT complete frame");
            for (int pixel = 0; pixel < txtPixels.size(); ++pixel)
                check(txtPixels[pixel] == p["intensity"].toArray()[pixel].toInt(), "TXT and JSON values identical");
            check(p["controller_x_steps"].toInt() == (mode == "1d-x" ? (i-1)*69400 : 69400), "X coordinate");
            check(p["controller_y_steps"].toInt() == (mode == "1d-y" ? (i-1)*69400 : 72870), "Y coordinate");
            check(p["controller_z_steps"].toInt() == 131860, "Z held fixed");
        }
        check(w.lineAxisCombo->isEnabled(), "axis selector restored after scan");
        check(combined.count("Image Data") == 13 && combined.count("Stop Capture") == 13, "combined TXT contains all 13 frames");
        for (int i = 1; i <= 13; ++i)
            check(combined.contains(QString("# Point: %1\n").arg(i).toUtf8()), "combined retains point identifiers");
    }
    static void monitor(bool fail = false) {
        MainWindow w;
        w.monitorButton->click();
        check(!w.monitorActive, "monitor requires explicit Laser ON");
        w.laserOn = true;
        const QString dir = w.laserProcess->workingDirectory();
        AcquisitionProtocol::publish(dir + "/mock_mode.txt", fail ? "fail" : "ok");
        w.monitorButton->click();
        if (fail) {
            check(waitUntil([&]{return !w.monitorActive && !w.isCollecting;}), "preview error stops monitor");
            check(w.backendCaptureFault && w.monitorFrameCount == 0, "bad preview not displayed");
            return;
        }
        click(w, "Start Z scan then 2D scan");
        click(w, "Capture Dark");
        check(w.sessionId.isEmpty(), "monitor blocks motor scan");
        check(waitUntil([&]{return w.monitorFrameCount >= 3;}), "monitor continuously acquires");
        w.monitorButton->click();
        check(waitUntil([&]{return !w.isCollecting;}), "preview drains before another operation");
        const int count = readFile(dir + "/calls.txt").count('\n');
        spin(250);
        check(readFile(dir + "/calls.txt").count('\n') == count, "OFF stops new captures");
        check(w.rtSeries->count() == 2048, "monitor displays complete frame");
        check(w.ramanProcess->state() == QProcess::NotRunning, "monitor never starts motor");
        check(QDir(dir + "/capture_results").entryList({"*.txt"}, QDir::Files).isEmpty(), "no preview spectral files");
        check(!readFile(w.laserTxtFile.fileName()).contains("MONITOR_FRAME"), "preview pixels excluded from log");
        start(w);
        check(waitUntil([&]{return w.lastCompletedPoint == 3 && !w.scanActive;}), "normal scan still works after monitor");
    }
    static void duringCapture(MainWindow& w, const QString& button) {
        auto* timer = new QTimer(&w);
        timer->setInterval(5);
        QObject::connect(timer, &QTimer::timeout, &w, [&w, timer, button]{
            if (w.isCollecting) { timer->stop(); click(w, button); }
        });
        timer->start();
    }
    static void click(MainWindow& w, const QString& text) {
        for (auto* b : w.findChildren<QPushButton*>()) if (b->text() == text) { b->click(); return; }
        throw std::runtime_error("Button not found");
    }
    static void start(MainWindow& w, const QByteArray& mode = "ok") {
        check(w.laserProcess->state() == QProcess::Running, "Mock backend not running");
        check(AcquisitionProtocol::publish(w.laserProcess->workingDirectory() + "/mock_mode.txt", mode), "Set mock mode");
        w.laserOn = true; // Tests only: bypass physical laser controls, never call an SDK.
        click(w, "Start Z scan then 2D scan");
    }
    static void success(const QByteArray& mode = "ok") {
        MainWindow w; start(w, mode);
        check(waitUntil([&]{return w.lastCompletedPoint == 3 && !w.scanActive;}), "3 points must complete");
        check(readFile(w.sessionDir + "/advanced.txt") == "3", "motor advances after saved point");
        check(readFile(w.laserProcess->workingDirectory() + "/calls.txt").count('\n') == 3, "duplicates must not recapture");
        check(readFile(w.sessionDir + "/laser_log_combined.txt").count("Image Data") == 3, "combined TXT has no duplicate frames");
        const auto p = QJsonDocument::fromJson(readFile(w.sessionDir + "/point_0001.json")).object();
        check(p["intensity"].toArray().size() == 2048 && p["controller_z_steps"].toInt() == 131860, "saved spectrum and coordinates");
    }
    static void pause() {
        MainWindow w; start(w);
        duringCapture(w, "Pause");
        check(waitUntil([&]{return w.lastCompletedPoint == 1 && !w.isCollecting;}), "in-flight point saved during pause");
        spin(150);
        check(readFile(w.sessionDir + "/advanced.txt").isEmpty(), "no advance during pause");
        const auto logSize = QFileInfo(w.laserTxtFile.fileName()).size();
        click(w, "Resume");
        check(waitUntil([&]{return w.lastCompletedPoint == 3 && !w.scanActive;}), "resume completes");
        check(QFileInfo(w.laserTxtFile.fileName()).size() >= logSize, "resume does not clear logs");
        check(readFile(w.laserProcess->workingDirectory() + "/calls.txt").count('\n') == 3, "paused point not reacquired");
    }
    static void failure(const QByteArray& mode) {
        MainWindow w; start(w, mode);
        check(waitUntil([&]{return !w.scanActive && !w.isCollecting;}), "failure must stop scan");
        check(w.lastCompletedPoint == 0 && w.backendCaptureFault, "failed data not accepted");
        check(w.ramanProcess->exitStatus() == QProcess::NormalExit && w.ramanProcess->exitCode() == 4,
              "allow delayed motor cleanup without force-kill");
        check(!QFile::exists(w.sessionDir + "/point_0001.json"), "failure must not save success record");
        check(readFile(w.sessionDir + "/advanced.txt").isEmpty(), "motor must not advance after failure");
        const auto run = w.sessionId;
        click(w, "Start Z scan then 2D scan");
        check(w.sessionId == run, "backend fault blocks new scan");
        if (mode == "late") {
            spin(1000);
            check(w.lastCompletedPoint == 0, "late result cannot be accepted after timeout");
        }
    }
    static void stop() {
        MainWindow w; start(w);
        duringCapture(w, "Stop");
        check(waitUntil([&]{return !w.scanActive && !w.isCollecting;}), "stop completes");
        check(readFile(w.sessionDir + "/advanced.txt").isEmpty(), "no advance after Stop");
        check(w.lastCompletedPoint == 0, "in-flight cancelled capture not attributed to a scan point");
    }
    static void darkIsolation() {
        MainWindow w; start(w);
        duringCapture(w, "Capture Dark");
        check(waitUntil([&]{return w.lastCompletedPoint == 3 && !w.scanActive;}), "scan survives dark-button click");
        check(readFile(w.laserProcess->workingDirectory() + "/calls.txt").count('\n') == 3, "dark cannot overwrite scan capture command");
    }
};
static void protocolTests() {
    const auto positions = lineScanPositions();
    check(positions.size() == 13 && positions.front() == 0 && positions.back() == 832800, "0..24 mm inclusive");
    check(lineScanWithinLimits(positions, 0, 867500), "25 mm stage accepts plan");
    check(!lineScanWithinLimits(positions, 0, 694000), "20 mm stage rejects plan");
    check(!lineScanWithinLimits(positions, 0, 0), "unknown limits reject plan");
    using D = CaptureStatusDecision;
    check(captureStatusDecision(1, 0, -1, 5, 1250, 8000) == D::Wait, "rc1 means keep waiting for this status API");
    check(captureStatusDecision(1, 1, 0, 5, 1500, 8000) == D::Wait, "rc1 never accepts stale outputs");
    check(captureStatusDecision(0, 1, 0, 5, 2000, 8000) == D::Ready, "ready after pending polls");
    check(captureStatusDecision(1, 0, -1, 5, 8000, 8000) == D::Timeout, "persistent pending times out");
    check(captureStatusDecision(0, 1, 0, 5, 8001, 8000) == D::Timeout, "late success rejected");
    check(captureStatusDecision(3, 0, -1, 5, 1300, 8000) == D::Error, "access error must not retry");
    check(captureStatusDecision(41, 0, -1, 5, 1300, 8000) == D::Error, "capture error must not retry");
    check(captureStatusDecision(0, 6, 0, 5, 1300, 8000) == D::Error, "invalid metadata not success");
    check(firstCaptureStatusDelayMs(1000000, 1100000) == 1250, "restore known-working initial wait");
    check(firstCaptureStatusDelayMs(2000000, 2100000) == 2250, "wait scales with longer exposure/cycle");
    check(firstCaptureStatusDelayMs(2500001, 1100000) == 2651, "round exposure up to full milliseconds");
    check(firstAvailableCaptureIndex(1, 0, 5) == 0, "first complete frame");
    check(firstAvailableCaptureIndex(2, 1, 5) == 0, "read first frame when multiple ready");
    check(firstAvailableCaptureIndex(5, 4, 5) == 0, "full legacy buffer");
    check(firstAvailableCaptureIndex(3, 0, 5) == 3, "wrapped buffer index");
    check(firstAvailableCaptureIndex(0, 0, 5) == -1, "reject no frame");
    check(firstAvailableCaptureIndex(6, 0, 5) == -1, "reject invalid frame count");
    using namespace AcquisitionProtocol;
    Point point;
    check(parsePoint("STEP run 1 2 3 4", "run", point), "valid point");
    check(!parsePoint("STEP old 1 2 3 4", "run", point), "wrong session");
    check(!parsePoint("STEP run 1 2 3", "run", point), "partial point");
    QVector<int> values; QString error;
    check(parseSpectrum("CAPTURE c OK 2\n0001 FFFF\nEND c", "c", 2, values, error), "valid spectrum");
    check(!parseSpectrum("CAPTURE c OK 2\n0001\nEND c", "c", 2, values, error), "partial spectrum");
    check(!parseSpectrum("CAPTURE c OK 2\n0001 ERROR\nEND c", "c", 2, values, error), "nonhex spectrum");
    check(!parseSpectrum("CAPTURE old OK 2\n0001 0002\nEND old", "c", 2, values, error), "old spectrum");
    QTemporaryDir tmp;
    check(!publish(tmp.path() + "/missing/path.txt", "x"), "write failure detected");
    const auto dir = tmp.path().toStdString();
    check(ScanHandshake::publish(dir + "/response.txt", "ACK old 1 OK 1"), "publish stale ack");
    check(!ScanHandshake::waitForAck(dir, "run", 1, 100), "stale session must not advance");
    ScanHandshake::publish(dir + "/response.txt", "ACK run 2 OK 2");
    check(!ScanHandshake::waitForAck(dir, "run", 1, 100), "wrong point must not advance");
    ScanHandshake::publish(dir + "/response.txt", "ACK run 1 PAUSED 3");
    check(!ScanHandshake::waitForAck(dir, "run", 1, 100), "frozen heartbeat must time out");
    ScanHandshake::publish(dir + "/response.txt", "ACK run 1 FAIL 4");
    check(!ScanHandshake::waitForAck(dir, "run", 1, 100), "explicit failure");
    ScanHandshake::publish(dir + "/response.txt", "ACK run 1 OK 5");
    check(ScanHandshake::waitForAck(dir, "run", 1, 100), "matching saved ack advances");
    ScanHandshake::publish(dir + "/cancel.txt", "STOP");
    check(!ScanHandshake::waitForAck(dir, "run", 1, 100), "cancel wins over saved ack");
}
int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    try {
        protocolTests(); std::cout << "PASS protocol validation / stale ACK / cancellation / heartbeat\n";
        HandshakeTest::darkStorage(); std::cout << "PASS dark subfolder / unique names / complete TXT\n";
        HandshakeTest::monitor(); std::cout << "PASS live preview / no files / mode exclusion / return to scan\n";
        HandshakeTest::monitor(true); std::cout << "PASS preview failure stops loop\n";
        HandshakeTest::success(); std::cout << "PASS 3-point integration / duplicate request\n";
        HandshakeTest::success("locked_short"); std::cout << "PASS transient file lock / same capture command\n";
        HandshakeTest::success("locked_two"); std::cout << "PASS two failed reads / third read recovers\n";
        HandshakeTest::failure("locked_long"); std::cout << "PASS persistent file lock / stop without advancing\n";
        HandshakeTest::lineScan("1d-x"); std::cout << "PASS 1D X / 13 saved points / live display\n";
        HandshakeTest::lineScan("1d-y"); std::cout << "PASS 1D Y / 13 saved points / live display\n";
        HandshakeTest::success("not_ready_then_ready"); std::cout << "PASS repeated rc1 then ready / one capture command per point\n";
        HandshakeTest::pause(); std::cout << "PASS pause in exposure / resume / retained log\n";
        for (auto mode : {"fail", "partial", "wrong_id", "late", "not_ready_timeout"}) {
            HandshakeTest::failure(mode); std::cout << "PASS failure scenario: " << mode << '\n';
        }
        HandshakeTest::stop(); std::cout << "PASS stop during exposure\n";
        HandshakeTest::darkIsolation(); std::cout << "PASS dark/scan mutual exclusion\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL " << ex.what() << std::endl; return 1;
    }
}
