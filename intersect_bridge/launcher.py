"""Windows GUI launcher. Double-click EXE starts software, never a scan or laser."""
import argparse
import ctypes
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QTimer, QLockFile
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QPlainTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QSplitter)
from launcher_core import LauncherCore

class Signals(QObject):
    log = Signal(str)
    done = Signal(str, object)
    failed = Signal(str)

class Window(QMainWindow):
    def __init__(self, root, preview=False):
        super().__init__()
        self.root = root
        self.preview = preview
        self.busy = False
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.signals = Signals()
        self.signals.log.connect(self.write_log)
        self.signals.done.connect(self.finished)
        self.signals.failed.connect(self.failed)
        self.core = LauncherCore(root, self.signals.log.emit)
        self.setWindowTitle("Raman INTERSECT - Experiment Launcher")
        self.resize(1120, 800)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 18)
        title = QLabel("Raman × INTERSECT")
        title.setStyleSheet("font-size:26px;font-weight:700;color:#16365c")
        layout.addWidget(title)
        layout.addWidget(QLabel("Automatic startup - Qt device control - PCA-SVM analysis"))
        self.summary = QLabel("Waiting to start..." if not preview else "Preview only: no services or devices are started")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("background:#e9f1fa;padding:14px;border-radius:8px;font-size:15px")
        layout.addWidget(self.summary)
        self.controls = []
        row = QHBoxLayout()
        self.start_btn = self.button("Start / Check System", lambda:self.submit("startup", self.core.startup), row)
        self.button("Check Status", lambda:self.submit("status", lambda:self.core.rpc("get_status")), row)
        self.button("Restart Analysis Service", lambda:self.submit("status", self.core.restart_service), row)
        self.button("Open Config", self.open_config, row, managed=False)
        self.button("Open Code Folder", lambda:os.startfile(str(root)), row, managed=False)
        row.addStretch()
        layout.addLayout(row)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Scan mode"))
        self.mode = QComboBox()
        self.mode.addItem("Y axis - 0-24 mm - 13 points", "1d-y")
        self.mode.addItem("X axis - 0-24 mm - 13 points", "1d-x")
        self.mode.addItem("Original Z + 2D scan", "2d")
        mode_row.addWidget(self.mode)
        self.scan_btn = self.button("Start Scan", self.start_scan, mode_row)
        self.button("Cancel Current Scan", lambda:self.submit("cancel", self.core.cancel), mode_row)
        mode_row.addStretch()
        layout.addLayout(mode_row)
        result_row = QHBoxLayout()
        result_row.addWidget(QLabel("Session ID"))
        self.session = QLineEdit(self.core.session_id)
        self.session.setPlaceholderText("Filled after starting a scan here, or paste an existing session ID")
        result_row.addWidget(self.session, 1)
        self.button("View SVM Results", self.get_results, result_row)
        self.button("Open Results Folder", self.open_results, result_row, managed=False)
        layout.addLayout(result_row)
        self.analysis = QLabel("Waiting for results. Model scores are not identification accuracy.")
        layout.addWidget(self.analysis)
        split = QSplitter()
        from PySide6.QtCore import Qt
        split.setOrientation(Qt.Vertical)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Point ID", "Y steps", "Class", "R6G", "MB", "Adenine", "Background / Reason"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        split.addWidget(self.table)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(1000)
        split.addWidget(self.logbox)
        split.setSizes([260, 180])
        layout.addWidget(split, 1)
        note = QLabel("Enable the laser and remote scans in Qt. Closing this launcher does not stop scans, turn off the laser, or close Qt / services.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#5b6573;font-size:12px")
        layout.addWidget(note)
        self.setStyleSheet("QWidget{font-family:'Microsoft YaHei UI';font-size:13px;} QPushButton{padding:8px 12px;} QLineEdit,QComboBox{padding:7px;} QTableWidget{background:white;alternate-background-color:#f4f7fa;}")
        self.table.setAlternatingRowColors(True)
        self.write_log("Code folder: " + str(root))
        self.write_log("Current dark spectrum: " + str(self.core.path("dark_file")))
        if preview:
            for button in self.controls: button.setEnabled(False)
        else:
            QTimer.singleShot(300, lambda:self.submit("startup", self.core.startup))

    def button(self, text, action, layout, managed=True):
        button = QPushButton(text)
        button.clicked.connect(action)
        layout.addWidget(button)
        if managed: self.controls.append(button)
        return button

    def write_log(self, text):
        from datetime import datetime
        self.logbox.appendPlainText(datetime.now().strftime("%H:%M:%S  ") + text)

    def submit(self, kind, function):
        if self.busy or self.preview: return
        self.busy = True
        for button in self.controls: button.setEnabled(False)
        self.summary.setText("Working, please wait... (Qt remains available)")
        def worker():
            try: self.signals.done.emit(kind, function())
            except Exception as exc:
                text = str(exc)
                try:
                    secret = self.core.cfg()["intersect"]["brokers"][0].get("password", "")
                except Exception:
                    secret = ""
                self.signals.failed.emit(text.replace(secret, "[password hidden]") if secret else text)
        self.pool.submit(worker)

    def unlock(self):
        self.busy = False
        for button in self.controls: button.setEnabled(True)
        self.scan_btn.setText("Retry Last Start" if self.core.pending_start else "Start Scan")

    def failed(self, text):
        self.unlock()
        self.summary.setText("Operation incomplete. See the log below.")
        self.write_log(text)

    def finished(self, kind, result):
        self.unlock()
        if kind in ("startup", "status"):
            qt = result.get("qt", {})
            if result.get("offline"):
                self.summary.setText("Qt status unavailable: " + str(result.get("error", "")))
            else:
                self.summary.setText("INTERSECT connected  |  Scan: {}  |  Points: {}  |  Laser: {}  |  Remote scans: {}".format(
                    qt.get("scan_state", "unknown"), qt.get("last_completed_point", 0),
                    "ON" if qt.get("laser_on") else "OFF", "enabled" if qt.get("remote_scan_enabled") else "disabled"))
            if result.get("analysis_error"): self.write_log("Analysis error: "+result["analysis_error"])
        elif kind == "scan":
            self.session.setText(result["session_id"])
            self.summary.setText("Scan submitted. SVM analyzes spectra saved by Qt. Check status or view results.")
        elif kind == "cancel":
            self.summary.setText("Scan cancellation completed. The laser is not turned off automatically.")
        elif kind == "results":
            rows = result["points"]
            self.table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                scores = row.get("scores", {})
                values = [row["point_id"], row.get("controller_y_steps", ""), row["label"]]
                values.extend(f"{scores[c]:.4f}" if c in scores else "—" for c in ("r6g", "methylene_blue", "adenine"))
                values.append((f"{scores['di_water']:.4f} " if "di_water" in scores else "") + row.get("reason", ""))
                for j, value in enumerate(values): self.table.setItem(i, j, QTableWidgetItem(str(value)))
            self.analysis.setText(f"Analyzed {result['analyzed_points']} points; analysis error: {result.get('analysis_error') or "none"}")
            self.summary.setText("Results updated. Click View SVM Results again during a scan to load new points.")
        self.write_log("Completed: " + kind)

    def start_scan(self):
        mode = self.mode.currentData()
        if self.core.pending_start:
            self.submit("scan", lambda:self.core.scan(mode))
            return
        description = "The selected axis will home, then measure 13 points from 0 to 24 mm." if mode != "2d" else "The original Z and XY scan paths will run."
        if QMessageBox.question(self, "Start Physical Scan", description+"\nAfter checking the sample, focus, and travel path, start the scan?") == QMessageBox.Yes:
            self.submit("scan", lambda:self.core.scan(mode))

    def get_results(self):
        session_id = self.session.text().strip()
        self.submit("results", lambda:self.core.results(session_id))

    def open_config(self):
        os.startfile(str(self.root/"config.json"))
        self.write_log("After changing the model or dark spectrum, finish acquisition and click Restart Analysis Service. Password changes must also match the broker account.")

    def open_results(self):
        path = self.core.path("results_dir")
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def closeEvent(self, event):
        if self.busy:
            QMessageBox.information(self, "Working", "Wait for the current operation to finish before closing. You can still stop the scan in Qt.")
            event.ignore()
            return
        self.pool.shutdown(wait=False)
        event.accept()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Display GUI only; never start services or hardware")
    parser.add_argument("--smoke-test", action="store_true", help="Render preview then exit, no services or hardware")
    parser.add_argument("--root")
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else (Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent)
    app = QApplication(sys.argv[:1])
    if os.name == "nt" and getattr(sys, "frozen", False):
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    lock = QLockFile(str(root/"runtime/launcher/gui.lock"))
    (root/"runtime/launcher").mkdir(parents=True, exist_ok=True)
    if not lock.tryLock(100):
        QMessageBox.information(None, "Launcher Already Running", "A launcher window is already open. Please switch to it.")
        return
    try:
        window = Window(root, preview=args.preview or args.smoke_test)
    except Exception as exc:
        QMessageBox.critical(None, "Cannot Open Launcher", "Keep the launcher in the intersect_bridge folder and check config.json.\n"+str(exc))
        return
    window.show()
    if args.smoke_test:
        def capture():
            window.grab().save(str(root/"runtime/launcher/preview.png"))
            window.close()
        QTimer.singleShot(1000, capture)
    app.exec()

if __name__ == "__main__":
    main()
