"""
CNN 2-class Classifier: R6G vs Methylene Blue (with UNKNOWN rejection)

Standalone Python script — NOT a Jupyter notebook.

Features:
  - Auto-finds the latest laser_log_*.txt (test data) and dark_*.txt in DATA_ROOT
  - Dark subtraction: raw - dark, then drop first 30 points
  - Trains on database files (also dark-subtracted)
  - Pure softmax-based open-set rejection (no embedding distance)
  - Prints clean results to stdout

Usage:
  python cnn_2class_auto.py                          # auto-find latest files
  python cnn_2class_auto.py  path/to/test.txt        # specify test file
  python cnn_2class_auto.py  test.txt  dark.txt      # specify both
"""

import os
import sys
import re
import glob
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIG
# =============================================================================
# Root directory where laser_log_*.txt and dark_*.txt are generated
DATA_ROOT = r"C:\Users\Jeremy666\OneDrive\Documents\laser_test\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug"

# Database files for training (these are the known reference spectra)
LIB_FILES = {
    "r6g":            os.path.join(DATA_ROOT, r"Portable_Raman_database\Database_r6g.txt"),
    "methylene_blue": os.path.join(DATA_ROOT, r"Portable_Raman_database\Database_mythylene_blue.txt"),
}

# Dark files corresponding to each database
LIB_DARK_FILES = {
    "r6g":            os.path.join(DATA_ROOT, r"Portable_Raman_database\dark_r6g.txt"),
    "methylene_blue": os.path.join(DATA_ROOT, r"Portable_Raman_database\dark_mythylene_blue.txt"),
}

# Patterns to find latest test data and dark files automatically
TEST_FILE_PATTERN = "laser_log_*.txt"     # glob pattern in DATA_ROOT
DARK_FILE_PATTERN = "dark_*.txt"          # glob pattern in DATA_ROOT

DROP_N = 30
TARGET_LEN = 2018
BATCH_SIZE = 32
EPOCHS = 60
LR = 8e-4
WEIGHT_DECAY = 1e-4
SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

DO_BASELINE = False
NUM_WORKERS = 0

# Open-set rejection
CONFIDENCE_THRESHOLD = 0.80

INFER_TEMPERATURE = 1.0
DROPOUT = 0.25
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 3.0

USE_AUGMENT = True
AUG_NOISE_STD = 0.015
AUG_SHIFT_MAX = 12
AUG_SCALE_LOW = 0.90
AUG_SCALE_HIGH = 1.10

OUTPUT_DIR = os.path.join(DATA_ROOT, "_CNN_2class_r6g_mb")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
META_PATH  = os.path.join(OUTPUT_DIR, "meta.json")
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_2class.pt")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE.startswith("cuda")


# =============================================================================
# AUTO-FIND LATEST FILES
# =============================================================================
def find_latest_file(directory, pattern):
    """Find the most recently modified file matching glob pattern."""
    search = os.path.join(directory, pattern)
    files = glob.glob(search)
    if not files:
        return None

    # Try to extract timestamp from filename first (e.g. laser_log_20260404_173347.txt)
    ts_re = re.compile(r"(\d{8}_\d{6})")
    dated = []
    for f in files:
        m = ts_re.search(os.path.basename(f))
        if m:
            try:
                dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                dated.append((dt, f))
            except ValueError:
                pass

    if dated:
        dated.sort(key=lambda x: x[0], reverse=True)
        return dated[0][1]

    # Fallback: most recently modified
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def resolve_test_and_dark_files():
    """
    Determine test file and dark file from CLI args or auto-detection.

    Priority:
      1. CLI args: python script.py <test_file> [dark_file]
      2. Auto-find latest laser_log_*.txt and dark_*.txt in DATA_ROOT
    """
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    # Test file
    if len(args) >= 1 and args[0].strip() and os.path.isfile(args[0].strip()):
        test_file = args[0].strip()
    else:
        test_file = find_latest_file(DATA_ROOT, TEST_FILE_PATTERN)
        if test_file is None:
            print(f"[ERROR] No test file found matching '{TEST_FILE_PATTERN}' in {DATA_ROOT}")
            sys.exit(1)

    # Dark file
    if len(args) >= 2 and args[1].strip() and os.path.isfile(args[1].strip()):
        dark_file = args[1].strip()
    else:
        dark_file = find_latest_file(DATA_ROOT, DARK_FILE_PATTERN)
        if dark_file is None:
            print("[WARN] No dark file found — proceeding without dark subtraction")
            dark_file = ""

    return test_file, dark_file


# =============================================================================
# UTILS
# =============================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def stratified_split_indices(Y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                             test_ratio=TEST_RATIO, seed=SEED):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for cls in np.unique(Y):
        idx = np.where(Y == cls)[0].tolist(); rng.shuffle(idx)
        n = len(idx); n_tr = int(n * train_ratio); n_va = int(n * val_ratio)
        idx_tr.extend(idx[:n_tr])
        idx_va.extend(idx[n_tr:n_tr + n_va])
        idx_te.extend(idx[n_tr + n_va:])
    rng.shuffle(idx_tr); rng.shuffle(idx_va); rng.shuffle(idx_te)
    return idx_tr, idx_va, idx_te


def zscore(x, eps=1e-8):
    x = np.nan_to_num(np.asarray(x, np.float32))
    m, s = float(np.mean(x)), float(np.std(x))
    if not np.isfinite(m) or not np.isfinite(s) or s < eps:
        return np.zeros_like(x, np.float32)
    return ((x - m) / (s + eps)).astype(np.float32)


def softmax_np(logits):
    z = np.asarray(logits, np.float64)
    if not np.isfinite(z).all() or z.size == 0:
        return np.zeros_like(z, np.float32)
    z -= z.max(); e = np.exp(z); s = e.sum()
    return (e / s).astype(np.float32) if s > 0 and np.isfinite(s) else np.zeros_like(z, np.float32)


# =============================================================================
# DARK / PARSING / PREPROCESSING
# =============================================================================
def load_dark_spectrum_from_file(dark_path):
    """Load dark spectrum from tab-separated file (index  raman_shift  intensity)."""
    if not dark_path or not Path(dark_path).exists():
        return np.array([], dtype=np.float64)
    intensities = []
    with open(dark_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("index"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    intensities.append(float(parts[2]))
                except ValueError:
                    continue
    return np.array(intensities, dtype=np.float64)


ERROR_PHRASE = "image data error code 0x0001"


def strip_prefix(line):
    line = line.strip()
    line = re.sub(r"^\[[^\]]+\]\s+\[Backend\]\s*", "", line)
    line = re.sub(r"^\[[^\]]+\]\s*", "", line)
    return line.strip()


def parse_portable_log_to_spectra(log_file):
    """Parse portable Raman log -> list of raw intensity arrays."""
    lines = open(log_file, "r", encoding="utf-8", errors="ignore").read().splitlines()
    spectra, failed = [], 0
    i = 0
    while i < len(lines):
        line_raw = lines[i]; low = line_raw.lower(); line = strip_prefix(line_raw)
        if ERROR_PHRASE in low:
            failed += 1; i += 1; continue
        if line.startswith("Image Data"):
            tokens = list(re.findall(r"\b[0-9A-Fa-f]{2,4}\b", line[len("Image Data"):].strip()))
            j = i + 1; block_err = False
            while j < len(lines):
                nxt_raw = lines[j]; nxt = strip_prefix(nxt_raw)
                if ERROR_PHRASE in nxt_raw.lower():
                    block_err = True
                if nxt.startswith("Stop Capture") or (nxt.startswith("Image Data") and j != i):
                    break
                hexes = re.findall(r"\b[0-9A-Fa-f]{2,4}\b", nxt)
                if hexes:
                    tokens.extend(hexes)
                j += 1
            if block_err or not tokens:
                failed += 1
            else:
                vals = np.array([int(t, 16) for t in tokens], dtype=np.int64)
                if vals.size % 2 == 0 and vals.max(initial=0) <= 255:
                    le = vals[0::2] + (vals[1::2] << 8)
                    be = (vals[0::2] << 8) + vals[1::2]
                    v16 = le if le.max(initial=0) >= be.max(initial=0) else be
                    if v16.max(initial=0) > 255:
                        vals = v16.astype(np.int64)
                spectra.append(vals.astype(np.float32))
            i = j + 1; continue
        i += 1
    return spectra, failed


def apply_dark_and_trim(spectrum, dark):
    """Subtract dark spectrum, then drop first DROP_N points."""
    s = np.asarray(spectrum, np.float64)
    if dark.size > 0:
        ml = min(s.size, dark.size)
        s = s[:ml] - dark[:ml]
    if s.size > DROP_N:
        s = s[DROP_N:]
    return s.astype(np.float32)


def preprocess_intensity_to_vec(y, target_len, do_baseline=False):
    """Interpolate to target_len, then z-score normalize."""
    y = np.nan_to_num(np.asarray(y, np.float32))
    if y.size < 4:
        out = np.zeros(target_len, np.float32)
        out[:min(target_len, y.size)] = y[:min(target_len, y.size)]
        return out
    if do_baseline:
        xl = np.linspace(0, 1, y.size)
        y = y - np.polyval(np.polyfit(xl, y, 3), xl).astype(np.float32)
    x = np.arange(y.size, dtype=np.float64)
    xn = np.linspace(x.min(), x.max(), target_len, dtype=np.float64)
    return zscore(np.interp(xn, x, y.astype(np.float64)).astype(np.float32))


# =============================================================================
# BUILD DATASET (dark-subtracted)
# =============================================================================
class_names = list(LIB_FILES.keys())
class_to_id = {c: i for i, c in enumerate(class_names)}
id_to_class = {i: c for c, i in class_to_id.items()}
NUM_CLASSES = len(class_names)


def build_dataset():
    """Load all database spectra, apply dark subtraction, preprocess."""
    X, Y = [], []
    print("\n===== Building dataset (dark-subtracted) =====")
    for cls, fp in LIB_FILES.items():
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Database file not found: {fp}")

        dark_raw = load_dark_spectrum_from_file(LIB_DARK_FILES.get(cls, ""))
        if dark_raw.size > 0:
            print(f"  [DARK] {cls}: loaded {dark_raw.size} points")
        else:
            print(f"  [DARK] {cls}: NO dark file — using raw spectra")

        spectra, failed = parse_portable_log_to_spectra(fp)
        print(f"  [LIB]  {cls:16s}  spectra={len(spectra):3d}  failed={failed}")

        for s in spectra:
            v = preprocess_intensity_to_vec(
                apply_dark_and_trim(s, dark_raw), TARGET_LEN, DO_BASELINE
            )
            if not np.allclose(v, 0):
                X.append(v)
                Y.append(class_to_id[cls])

    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.int64)
    counts = {id_to_class[i]: int((Y == i).sum()) for i in range(NUM_CLASSES)}
    print(f"  Total: {X.shape}, classes: {counts}")
    return X, Y


# =============================================================================
# DATASET + AUGMENTATION
# =============================================================================
def augment_spectrum_1d(x):
    x = x.copy()
    x *= np.float32(np.random.uniform(AUG_SCALE_LOW, AUG_SCALE_HIGH))
    shift = int(np.random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1))
    if shift:
        x = np.roll(x, shift)
    x += np.random.normal(0, AUG_NOISE_STD, x.shape).astype(np.float32)
    return zscore(x)


class RamanDataset(Dataset):
    def __init__(self, X, Y, indices, augment=False):
        self.X, self.Y, self.indices, self.augment = X, Y, indices, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]; x = self.X[i].copy()
        if self.augment:
            x = augment_spectrum_1d(x)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(int(self.Y[i]), dtype=torch.long)


# =============================================================================
# MODEL
# =============================================================================
class SEBlock1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        h = max(4, ch // r)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(ch, h), nn.ReLU(True), nn.Linear(h, ch), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1)


class ResBlock1D(nn.Module):
    def __init__(self, ic, oc, ks=3, stride=1, drop=0.0):
        super().__init__()
        p = ks // 2
        self.c1 = nn.Conv1d(ic, oc, ks, stride, p, bias=False); self.b1 = nn.BatchNorm1d(oc)
        self.c2 = nn.Conv1d(oc, oc, ks, 1, p, bias=False);      self.b2 = nn.BatchNorm1d(oc)
        self.se = SEBlock1D(oc); self.drop = nn.Dropout(drop)
        self.sc = (nn.Sequential(nn.Conv1d(ic, oc, 1, stride, bias=False), nn.BatchNorm1d(oc))
                   if stride != 1 or ic != oc else nn.Identity())
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.act(self.b1(self.c1(x)))
        out = self.se(self.drop(self.b2(self.c2(out))))
        return self.act(out + self.sc(x))


class SERSResCNN1D(nn.Module):
    def __init__(self, in_ch=1, base=32, nc=2, drop=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3, bias=False), nn.BatchNorm1d(base), nn.ReLU(True),
            nn.Conv1d(base, base, 5, padding=2, bias=False),  nn.BatchNorm1d(base), nn.ReLU(True))
        self.s1 = nn.Sequential(ResBlock1D(base, base, 7, 1, drop * 0.5), nn.MaxPool1d(2))
        self.s2 = nn.Sequential(ResBlock1D(base, base * 2, 5, 1, drop),   nn.MaxPool1d(2))
        self.s3 = nn.Sequential(ResBlock1D(base * 2, base * 4, 3, 1, drop), nn.MaxPool1d(2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(base * 4, 128)
        self.fc1_act = nn.ReLU(True)
        self.fc1_drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(128, nc)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.s3(self.s2(self.s1(self.stem(x))))
        x = self.pool(x).flatten(1)
        x = self.fc1_drop(self.fc1_act(self.fc1(x)))
        return self.fc2(x)


# =============================================================================
# TRAIN
# =============================================================================
@torch.no_grad()
def eval_acc(model, loader):
    model.eval(); total = correct = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)


def train_model(X, Y, idx_tr, idx_va, idx_te):
    train_loader = DataLoader(RamanDataset(X, Y, idx_tr, True),  BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(RamanDataset(X, Y, idx_va, False), BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(RamanDataset(X, Y, idx_te, False), BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = SERSResCNN1D(nc=NUM_CLASSES, drop=DROPOUT).to(DEVICE)

    counts = np.array([(Y == i).sum() for i in range(NUM_CLASSES)], np.float32)
    w = torch.tensor(counts.mean() / np.maximum(counts, 1), dtype=torch.float32, device=DEVICE)
    ce = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", 0.5, 4, min_lr=1e-6)

    best_val, best_state, bad = -1.0, None, 0
    print("\n===== Training =====")
    for ep in range(1, EPOCHS + 1):
        model.train(); losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step(); losses.append(loss.item())
        va = eval_acc(model, val_loader); sched.step(va)
        print(f"  epoch {ep:02d}/{EPOCHS}  loss={np.mean(losses):.4f}  "
              f"val_acc={va:.3f}  lr={opt.param_groups[0]['lr']:.2e}")
        if va > best_val + 1e-6:
            best_val = va
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 14:
                print("  early stop"); break

    if best_state:
        model.load_state_dict(best_state)
    te_acc = eval_acc(model, test_loader)
    print(f"  TEST acc = {te_acc:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")
    return model


# =============================================================================
# PREDICT (pure softmax, dark-subtracted)
# =============================================================================
@torch.no_grad()
def predict_single_file(model, spectrum_path, dark_path="",
                        scan_all=True, debug=True):
    """
    Classify all spectra in a file.

    Pipeline per spectrum:
      raw -> subtract dark -> drop first 30 -> interpolate -> zscore -> CNN

    Decision:
      avg_softmax >= CONFIDENCE_THRESHOLD  ->  label = top class
      avg_softmax <  CONFIDENCE_THRESHOLD  ->  label = UNKNOWN
    """
    if not os.path.exists(spectrum_path):
        raise FileNotFoundError(f"Not found: {spectrum_path}")

    model.to(DEVICE); model.eval()
    dark_raw = load_dark_spectrum_from_file(dark_path) if dark_path else np.array([])

    if dark_raw.size > 0 and debug:
        print(f"  [DARK] Loaded test dark: {dark_raw.size} points from {os.path.basename(dark_path)}")

    spectra, failed = parse_portable_log_to_spectra(spectrum_path)
    already_pp = False

    # Fallback: try raw hex parsing if no "Image Data" blocks found
    if not spectra:
        text = open(spectrum_path, "r", encoding="utf-8", errors="ignore").read()
        tokens = re.findall(r"\b(?:0x)?[0-9A-Fa-f]{2,4}\b", text)
        tokens = [t[2:] if t.lower().startswith("0x") else t for t in tokens]
        if len(tokens) >= 32:
            vals = np.array([int(t, 16) for t in tokens], np.int64)
            if vals.size % 2 == 0 and vals.max(initial=0) <= 255:
                le = vals[0::2] + (vals[1::2] << 8)
                be = (vals[0::2] << 8) + vals[1::2]
                vals = le if le.max(initial=0) >= be.max(initial=0) else be
            v = preprocess_intensity_to_vec(
                apply_dark_and_trim(vals.astype(np.float32), dark_raw), TARGET_LEN)
            spectra = [v]; already_pp = True
        else:
            return {"matched": False, "label": "UNKNOWN", "reason": "no spectra parsed",
                    "num_spectra": 0, "num_used": 0}

    n_scan = len(spectra) if scan_all else 1
    per_block = []

    for idx in range(n_scan):
        raw = spectra[idx]
        if already_pp:
            v = raw.astype(np.float32)
        else:
            v = preprocess_intensity_to_vec(
                apply_dark_and_trim(raw, dark_raw), TARGET_LEN)
        if np.allclose(v, 0):
            continue

        xt = torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = model(xt)
        logits_np = logits.squeeze(0).cpu().numpy()
        prob = softmax_np(logits_np / INFER_TEMPERATURE)

        cnn_pred = int(np.argmax(prob))
        cnn_conf = float(prob[cnn_pred])

        per_block.append({
            "idx": idx, "prob": prob, "cnn_pred": cnn_pred, "cnn_conf": cnn_conf,
        })

        if debug:
            prob_str = "  ".join(f"{id_to_class[k]}={prob[k]:.4f}" for k in range(NUM_CLASSES))
            print(f"  [SPEC] idx={idx:3d}  -> {id_to_class[cnn_pred]}  "
                  f"conf={cnn_conf:.4f}  [{prob_str}]")

    if not per_block:
        return {"matched": False, "label": "UNKNOWN", "reason": "all blocks zero",
                "num_spectra": len(spectra), "num_used": 0}

    # Aggregate: average softmax across all spectra
    avg_prob = np.mean(np.stack([b["prob"] for b in per_block]), axis=0)
    avg_pred = int(np.argmax(avg_prob))
    avg_conf = float(avg_prob[avg_pred])

    # Count per-spectrum votes
    votes = {}
    for b in per_block:
        cls_name = id_to_class[b["cnn_pred"]]
        votes[cls_name] = votes.get(cls_name, 0) + 1

    # Decision
    matched = avg_conf >= CONFIDENCE_THRESHOLD
    label = id_to_class[avg_pred] if matched else "UNKNOWN"
    reason = "OK" if matched else f"avg_softmax={avg_conf:.4f} < {CONFIDENCE_THRESHOLD}"

    prob_dict = {id_to_class[i]: float(avg_prob[i]) for i in range(NUM_CLASSES)}

    return {
        "matched": matched,
        "label": label,
        "reason": reason,
        "cnn_confidence": avg_conf,
        "votes": votes,
        "num_spectra": len(spectra),
        "num_used": len(per_block),
        "failed_blocks": failed,
        "probs": dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  CNN 2-Class Raman Classifier (R6G vs Methylene Blue)")
    print("  Dark subtraction enabled | Auto file detection")
    print("=" * 70)
    print(f"DEVICE = {DEVICE}")
    print(f"MODEL_PATH = {MODEL_PATH}")

    # ---- Resolve test and dark files ----
    test_file, dark_file = resolve_test_and_dark_files()
    print(f"\nTest file:  {test_file}")
    print(f"Dark file:  {dark_file if dark_file else '(none)'}")

    # ---- Build dataset ----
    X, Y = build_dataset()
    idx_tr, idx_va, idx_te = stratified_split_indices(Y)
    print(f"  Split: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")

    # ---- Train ----
    model = train_model(X, Y, idx_tr, idx_va, idx_te)

    # ---- Save meta ----
    meta = {
        "classes": class_names,
        "target_len": TARGET_LEN,
        "drop_n": DROP_N,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- Load and predict ----
    model2 = SERSResCNN1D(nc=NUM_CLASSES, drop=DROPOUT)
    model2.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model2.eval()
    print(f"\nLoaded: {MODEL_PATH}")

    print(f"\n===== Predicting: {os.path.basename(test_file)} =====")
    res = predict_single_file(model2, test_file,
                              dark_path=dark_file,
                              scan_all=True, debug=True)

    # ---- Print results ----
    print("\n" + "=" * 70)
    print(f"TEST FILE:  {test_file}")
    print(f"DARK FILE:  {dark_file if dark_file else '(none)'}")
    print(f"Spectra: {res['num_spectra']}  used={res['num_used']}  "
          f"failed={res.get('failed_blocks', 0)}")

    if res.get("votes"):
        print(f"Votes: {res['votes']}")

    print(f"\nSoftmax probabilities:")
    for k, v in res["probs"].items():
        bar = "#" * int(v * 50)
        print(f"  {k:16s}  {v:.4f}  {bar}")

    print(f"\nConfidence: {res.get('cnn_confidence', 0):.4f}  "
          f"(threshold={CONFIDENCE_THRESHOLD})")

    print(f"\nFINAL: {res['label']}")
    if res["matched"]:
        print(f"  -> MATCHED: {res['label']}  "
              f"(confidence={res['cnn_confidence']:.4f})")
    else:
        print(f"  -> UNKNOWN  (reason: {res['reason']})")

    print("=" * 70)


if __name__ == "__main__":
    main()