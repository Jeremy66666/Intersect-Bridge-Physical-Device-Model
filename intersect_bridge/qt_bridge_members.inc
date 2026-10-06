    // Included only in the generated controller copy, never the original source.
    QString bridgeBootId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    QString bridgeScanState = "idle";
    bool bridgeBusy = false;
    QCheckBox* bridgeArmed = nullptr;
    std::function<void(const QString&)> bridgeStartScan;

    QString bridgeRoot() const { return QCoreApplication::applicationDirPath() + "/ipc"; }
    QJsonObject bridgeState() const {
        return QJsonObject{{"boot_id", bridgeBootId}, {"utc_ms", QDateTime::currentMSecsSinceEpoch()},
            {"session_id", sessionId}, {"session_dir", sessionDir}, {"scan_state", bridgeScanState},
            {"scan_active", scanActive}, {"last_completed_point", static_cast<qint64>(lastCompletedPoint)},
            {"laser_on", laserOn}, {"laser_state_uncertain", laserStateUncertain},
            {"backend_fault", backendCaptureFault}, {"collecting", isCollecting},
            {"remote_scan_enabled", bridgeArmed && bridgeArmed->isChecked()}};
    }
    void bridgeTick() {
        AcquisitionProtocol::publish(bridgeRoot() + "/status.json", QJsonDocument(bridgeState()).toJson());
        if (bridgeBusy) return;
        QScopedValueRollback<bool> guard(bridgeBusy, true);
        const QDir requests(bridgeRoot() + "/requests");
        for (const QString& name : requests.entryList({"*.json"}, QDir::Files, QDir::Name)) {
            const QString id = QFileInfo(name).completeBaseName();
            if (!QRegularExpression("^[A-Za-z0-9_-]{1,80}$").match(id).hasMatch()) continue;
            const QString output = bridgeRoot() + "/responses/" + name;
            if (QFile::exists(output)) continue;
            QFile file(requests.filePath(name));
            if (!file.open(QIODevice::ReadOnly) || file.size() > 16384) continue;
            const auto request = QJsonDocument::fromJson(file.readAll()).object();
            QJsonObject reply{{"request_id", id}, {"ok", false}};
            const QString marker = bridgeRoot() + "/journal/" + name;
            const QString op = request["operation"].toString();
            QString error;
            if (QFile::exists(marker)) error = "Previous attempt uncertain; inspect status. Command will not be replayed.";
            else if (request["request_id"].toString() != id || request["boot_id"].toString() != bridgeBootId)
                error = "Wrong request ID or obsolete Qt boot ID";
            else if (request["expires_ms"].toDouble() < QDateTime::currentMSecsSinceEpoch()) error = "Command expired";
            else if (op != "start_scan" && op != "cancel_scan") error = "Unsupported operation";
            else if (!AcquisitionProtocol::publish(marker, QJsonDocument(request).toJson())) error = "Cannot journal command";
            else if (op == "start_scan") {
                const QString mode = request["mode"].toString();
                if (!bridgeArmed->isChecked()) error = "Enable remote scans in the local Qt window first";
                else if (mode != "1d-x" && mode != "1d-y" && mode != "2d") error = "Unsupported scan mode";
                else if (scanActive || monitorActive || isCollecting || laserCommandBusy || !laserOn ||
                         laserStateUncertain || backendCaptureFault || ramanProcess->state() != QProcess::NotRunning)
                    error = "Device not ready: check local laser, monitor, scan and fault state";
                else {
                    bridgeStartScan(mode);
                    if (!scanActive || ramanProcess->state() != QProcess::Running) error = "Scan start failed; inspect Qt log";
                    else reply["ok"] = true;
                }
            } else {
                if (request["session_id"].toString() != sessionId || sessionId.isEmpty()) error = "Session mismatch";
                else {
                    bridgeScanState = "cancelled";
                    stopMotorScan();
                    reply["ok"] = true;
                    reply["note"] = "Scan cancelled; laser power is unchanged";
                }
            }
            if (!error.isEmpty()) reply["error"] = error;
            reply["session_id"] = sessionId;
            reply["session_dir"] = sessionDir;
            reply["state"] = bridgeState();
            AcquisitionProtocol::publish(output, QJsonDocument(reply).toJson());
        }
    }
