#include <QApplication>
#include <QMainWindow>
#include <QPushButton>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QElapsedTimer>
#include <QThread>
#include <QFile>
#include <QTextStream>
#include <QCoreApplication>
#include <QTimer>
#include <QProcess>

class LaserWindow : public QMainWindow {
    Q_OBJECT

public:
    LaserWindow(QWidget *parent = nullptr) : QMainWindow(parent), laserOn(false) {
        auto *central = new QWidget;
        auto *layout = new QVBoxLayout;
        auto *toggleButton = new QPushButton("Toggle Laser ON/OFF");
        outputBox = new QTextEdit;
        outputBox->setReadOnly(true);

        layout->addWidget(toggleButton);
        layout->addWidget(outputBox);
        central->setLayout(layout);
        setCentralWidget(central);
        setWindowTitle("Laser Toggle");

        backendProcess = new QProcess(this);
        QString backendDir = "C:\\Users\\Jeremy666\\Downloads\\portableRamanSpec_Software (1)\\SampleApl\\SampleApl\\x64\\Debug";
        QString backendExe = backendDir + "\\SampleApl.exe";
        backendProcess->setWorkingDirectory(backendDir);
        backendProcess->start(backendExe);
        if (!backendProcess->waitForStarted(3000)) {
            outputBox->append("[Error] Could not start backend process: " + backendExe);
        } else {
            outputBox->append("[Backend process started]");
        }
        connect(backendProcess, &QProcess::readyReadStandardOutput, this, [this]() {
            QString text = backendProcess->readAllStandardOutput();
            outputBox->append("[Backend] " + text.trimmed());
        });
        connect(backendProcess, &QProcess::readyReadStandardError, this, [this]() {
            QString text = backendProcess->readAllStandardError();
            outputBox->append("[Backend ERROR] " + text.trimmed());
        });

        connect(toggleButton, &QPushButton::clicked, this, [this]() {
            sendCommand("6");
            outputBox->append("Laser power toggle initiated...");
            QTimer::singleShot(300, this, [this]() {
                if (!laserOn) {
                    sendCommand("1");
                    outputBox->append("Laser turned ON");
                } else {
                    sendCommand("0");
                    outputBox->append("Laser turned OFF");
                }
                laserOn = !laserOn;
            });
        });
    }

    ~LaserWindow() override {
        if (backendProcess && backendProcess->state() == QProcess::Running) {
            backendProcess->terminate();
            backendProcess->waitForFinished(2000);
        }
    }

private:
    QTextEdit *outputBox;
    bool laserOn;
    QProcess *backendProcess = nullptr;

    bool sendCommand(const QString &cmd) {
        QString basePath = "C:\\Users\\Jeremy666\\Downloads\\portableRamanSpec_Software (1)\\SampleApl\\SampleApl\\x64\\Debug";
        QString commandFilePath = basePath + "/command.txt";
        QString completedFilePath = basePath + "/command_completed.txt";

        QFile completedFile(completedFilePath);
        if (completedFile.exists()) completedFile.remove();

        QFile file(commandFilePath);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            outputBox->append("[Error writing command file]");
            return false;
        }

        QTextStream out(&file);
        out << cmd;
        file.close();

        QElapsedTimer timer;
        timer.start();
        while (timer.elapsed() < 5000) {
            QCoreApplication::processEvents();
            if (QFile::exists(completedFilePath)) {
                QFile doneFile(completedFilePath);
                if (doneFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                    QString response = doneFile.readAll();
                    doneFile.close();
                    if (response.contains("done")) {
                        QFile::remove(completedFilePath);
                        return true;
                    }
                }
            }
            QThread::msleep(100);
        }

        outputBox->append("[Timeout waiting for command processing]");
        return false;
    }
};

#include "main.moc"  // needed if not using a separate header

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    LaserWindow window;
    window.resize(400, 200);
    window.show();
    return app.exec();
}
