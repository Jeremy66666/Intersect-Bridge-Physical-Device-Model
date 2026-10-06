"""Pure-class PCA-SVM, extracted from supplied training script; never retrains."""
import hashlib
from pathlib import Path
import joblib
import numpy as np
from common import read_json

def load_dark(path):
    values = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        columns = line.split("\t")
        if len(columns) >= 3:
            try:
                values.append(float(columns[2]))
            except ValueError:
                pass
    dark = np.asarray(values, dtype=np.float64)
    if dark.shape != (2048,) or not np.isfinite(dark).all():
        raise ValueError("A complete finite 2048-pixel dark spectrum is required")
    return dark

def preprocess(raw, dark):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (2048,) or not np.isfinite(raw).all() or (raw < 0).any() or (raw >= 65535).any():
        raise ValueError("Invalid, incomplete or saturated spectrum")
    corrected = (raw - dark)[90:]
    padded = np.pad(corrected, (7, 7), mode="edge")
    minimum = np.array([padded[i:i+15].min() for i in range(corrected.size)])
    baseline = np.convolve(np.pad(minimum, (7, 7), mode="edge"), np.ones(15)/15, mode="valid")
    signal = corrected - baseline
    spread = signal.max() - signal.min()
    normalized = np.zeros_like(signal) if spread < 1e-12 else (signal - signal.min()) / spread
    return normalized.astype(np.float32), signal, float(np.ptp(corrected)), float(np.std(corrected))

class Predictor:
    def __init__(self, model_dir, dark_file):
        root = Path(model_dir)
        self.meta = read_json(root / "meta.json")
        self.classes = self.meta["classes"]
        self.threshold = float(self.meta["pure_branch"]["confidence_threshold"])
        paths = [root / n for n in ("pure_scaler_minmax.joblib", "pure_pca_minmax.joblib", "pure_svm_minmax.joblib")]
        # Only load the user's trusted local model artifacts.
        self.scaler, self.pca, self.svm = [joblib.load(p) for p in paths]
        if self.meta["feature_dim"] != 1958 or self.scaler.n_features_in_ != 1958:
            raise ValueError("Model preprocessing does not match the 1958-feature pure branch")
        self.dark = load_dark(dark_file)
        self.dark_hash = hashlib.sha256(Path(dark_file).read_bytes()).hexdigest()
        self.version = hashlib.sha256(b"".join(p.read_bytes() for p in paths) + (root / "meta.json").read_bytes()).hexdigest()

    def predict(self, record):
        result = {k: record[k] for k in ("session_id", "point_id", "capture_id", "controller_x_steps", "controller_y_steps", "controller_z_steps")}
        result.update(label="REJECTED", reason="", scores={}, model_sha256=self.version,
                      dark_sha256=self.dark_hash, mode="pure", score_type="svm_predict_proba")
        try:
            if record.get("status") != "OK" or record.get("pixel_count") != 2048:
                raise ValueError("Point is not a successful 2048-pixel acquisition")
            x, signal, ptp, std = preprocess(record["intensity"], self.dark)
            if np.allclose(signal, 0):
                raise ValueError("Preprocessing failed: zero baseline-corrected signal")
            if ptp < 100 or std < 15:
                raise ValueError("Low raw signal")
        except ValueError as exc:
            result["reason"] = str(exc)
            return result
        native = self.svm.predict_proba(self.pca.transform(self.scaler.transform(x.reshape(1, -1))))[0]
        scores = {self.classes[int(cid)]: float(value) for cid, value in zip(self.svm.classes_, native)}
        winner = max(scores, key=scores.get)
        result.update(scores=scores, label=winner)
        if winner == "di_water":
            result.update(label="UNKNOWN", reason="Background-class response; material not identified")
        elif scores[winner] < self.threshold:
            result.update(label="UNKNOWN", reason="Low pure-class confidence")
        return result
