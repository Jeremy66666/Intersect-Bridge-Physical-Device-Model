import ast
import json
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from common import ROOT, atomic_json, configuration, read_json
from qt_client import QtClient
from svm_predictor import Predictor, preprocess

class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = configuration()
        cls.model = Predictor(cls.cfg["model_dir"], cls.cfg["dark_file"])
        session = ROOT.parent / "build/Desktop_Qt_6_6_3_MSVC2019_64bit-Debug/scan_sessions/1d_y_scan_20260921_152840_289"
        cls.point = read_json(session / "point_0001.json")

    def test_saved_model_and_real_point(self):
        result = self.model.predict(self.point)
        self.assertEqual(result["session_id"], self.point["session_id"])
        self.assertIn(result["label"], self.model.classes + ["UNKNOWN", "REJECTED"])
        self.assertEqual(len(result["model_sha256"]), 64)

    def test_saturation_and_wrong_length_rejected(self):
        for raw in ([1]*2047, [65535]*2048, [float("nan")]*2048):
            result = self.model.predict(dict(self.point, intensity=raw))
            self.assertEqual(result["label"], "REJECTED")
            self.assertEqual(result["scores"], {})

    def test_zero_signal_rejected(self):
        result = self.model.predict(dict(self.point, intensity=self.model.dark.tolist()))
        self.assertEqual(result["label"], "REJECTED")

    def test_preprocessing_matches_supplied_original(self):
        attachment = Path("C:/Users/Jeremy666/.codex/attachments/16babfeb-ba70-4ad6-9f32-a83018ad7ba2/Pasted text.txt")
        if not attachment.exists():
            self.skipTest("Original attachment is not available on this computer")
        tree = ast.parse(attachment.read_text(encoding="utf-8-sig"))
        names = {"moving_average", "rolling_minimum", "baseline_correction_fast", "normalize_min_max", "preprocess_two_versions"}
        # Execute only the five pure helpers, never training or hardware code.
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        namespace = dict(np=np, TARGET_LEN=2048, DROP_N=90, DO_BASELINE=True, BASELINE_WINDOW=15, BASELINE_SMOOTH=15)
        exec(compile(functions, "original_preprocessing_helpers", "exec"), namespace)
        _, expected, _ = namespace["preprocess_two_versions"](self.point["intensity"], self.model.dark)
        actual, *_ = preprocess(self.point["intensity"], self.model.dark)
        np.testing.assert_array_equal(actual, expected)

    def test_mailbox_cached_reply_and_conflicting_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            qt = QtClient(temp)
            request = dict(operation="start_scan", mode="1d-y", request_id="same", boot_id="old", expires_ms=0)
            atomic_json(Path(temp)/"requests/same.json", request)
            atomic_json(Path(temp)/"responses/same.json", dict(ok=True, session_id="old-session"))
            self.assertEqual(qt.request("start_scan", "same", mode="1d-y")["session_id"], "old-session")
            with self.assertRaises(ValueError): qt.request("start_scan", "same", mode="1d-x")
            with self.assertRaises(ValueError): qt.request("start_scan", "../outside", mode="1d-y")

    def test_offline_status(self):
        with tempfile.TemporaryDirectory() as temp:
            qt = QtClient(temp)
            with self.assertRaises(RuntimeError): qt.status()
            atomic_json(Path(temp)/"status.json", dict(utc_ms=(time.time()-60)*1000))
            with self.assertRaises(RuntimeError): qt.status()

if __name__ == "__main__":
    unittest.main(verbosity=2)
