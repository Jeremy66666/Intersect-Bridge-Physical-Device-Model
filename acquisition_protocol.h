#pragma once
#include <QFile>
#include <QSaveFile>
#include <QRegularExpression>
#include <QStringList>
#include <QVector>
#include <QThread>

namespace AcquisitionProtocol {
inline bool publish(const QString& path, const QByteArray& bytes) {
    // Windows readers/OneDrive/antivirus may momentarily deny replacement.
    // Retry atomic publication, never fall back to a partial in-place write.
    for (int attempt = 0; attempt < 20; ++attempt) {
        QSaveFile file(path);
        if (file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit()) return true;
        QThread::msleep(10);
    }
    return false;
}
struct Point {
    quint64 id = 0;
    qint64 x = 0, y = 0, z = 0; // Controller-reported steps, not calibrated focus positions.
};
inline bool parsePoint(const QByteArray& bytes, const QString& run, Point& p) {
    const auto t = QString::fromUtf8(bytes).trimmed().split(QRegularExpression("\\s+"));
    if (t.size() != 6 || t[0] != "STEP" || t[1] != run) return false;
    bool a,b,c,d;
    p.id = t[2].toULongLong(&a); p.x = t[3].toLongLong(&b);
    p.y = t[4].toLongLong(&c); p.z = t[5].toLongLong(&d);
    return a && b && c && d && p.id > 0;
}
// A complete, atomically published result must have the expected ID and pixel count.
inline bool parseSpectrum(const QByteArray& bytes, const QString& id, int pixels,
                          QVector<int>& spectrum, QString& error) {
    spectrum.clear();
    const auto t = QString::fromUtf8(bytes).trimmed().split(QRegularExpression("\\s+"));
    if (t.size() < 6 || t[0] != "CAPTURE" || t[1] != id ||
        t[t.size()-2] != "END" || t.last() != id) {
        error = "Malformed or mismatched capture result"; return false;
    }
    if (t[2] != "OK") { error = "Backend capture failed: " + QString::fromUtf8(bytes).trimmed(); return false; }
    bool countOk = false;
    if (t[3].toInt(&countOk) != pixels || !countOk || t.size() != pixels + 6) {
        error = "Incomplete spectrum / unexpected pixel count"; return false;
    }
    for (int i = 0; i < pixels; ++i) {
        bool ok = false; const uint value = t[4+i].toUInt(&ok, 16);
        if (!ok || value > 65535 || !QRegularExpression("^[0-9A-Fa-f]{1,4}$").match(t[4+i]).hasMatch()) {
            spectrum.clear(); error = "Invalid pixel"; return false;
        }
        spectrum.append(static_cast<int>(value));
    }
    return true;
}
}
