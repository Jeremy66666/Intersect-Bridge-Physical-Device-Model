"""
CNN Raman classifier v3 — multi-label mixture detection
Model architecture (SE-ResNet1D) is IDENTICAL to v2.
Changes: multi-label sigmoid output, synthetic mixture training, OvR-style inference.
"""
import os, sys, re, json, csv, time, math, random
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = r"C:\Users\Jeremy666\OneDrive\Documents\laser_test\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug"
DATA_DIR = os.path.join(DATA_ROOT, "data_base_code_part")

TRAINING_MINMAX_NPY_FILES = {
    "r6g": os.path.join(DATA_DIR, "training_spectra_r6g_minmax.npy"),
    "methylene_blue": os.path.join(DATA_DIR, "training_spectra_methylene_blue_minmax.npy"),
    "adenine": os.path.join(DATA_DIR, "training_spectra_adenine_minmax.npy"),
    "di_water": os.path.join(DATA_DIR, "training_spectra_di_water_minmax.npy"),
}
TRAINING_RAW_NPY_FILES = {
    "r6g": os.path.join(DATA_DIR, "training_spectra_r6g_baseline_raw.npy"),
    "methylene_blue": os.path.join(DATA_DIR, "training_spectra_methylene_blue_baseline_raw.npy"),
    "adenine": os.path.join(DATA_DIR, "training_spectra_adenine_baseline_raw.npy"),
    "di_water": os.path.join(DATA_DIR, "training_spectra_di_water_baseline_raw.npy"),
}

# TEST_FILE = r"C:\Users\Jeremy666\OneDrive\Documents\laser_test\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug\portable_raman_test_data\laser_log_20260513_150549.txt"
# TEST_DARK_FILE = r"C:\Users\Jeremy666\OneDrive\Documents\laser_test\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug\portable_raman_test_data\dark_20260513_150739.txt"
TEST_FILE = r"C:\Users\Jeremy666\OneDrive\Documents\laser_test\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug\scan_sessions\1d_y_scan_20260921_152840_289\laser_log_combined.txt"

TEST_DARK_FILE = r"C:\Users\Jeremy666\OneDrive\Documents\laser_test\build\Desktop_Qt_6_6_3_MSVC2019_64bit-Debug\dark_measurements\dark_20260921_153016_700.txt"

# RETRAIN_MODEL = True
RETRAIN_MODEL = False

DROP_N = 90; TARGET_LEN = 2048; DO_BASELINE = True
BASELINE_WINDOW = 15; BASELINE_SMOOTH = 15

ENABLE_QUALITY_FILTER = True; RAW_MIN_PTP = 100.0; RAW_MIN_STD = 15.0

BATCH_SIZE = 32; EPOCHS = 80; LR = 5e-4; WEIGHT_DECAY = 5e-4; SEED = 42
TRAIN_RATIO = 0.70; VAL_RATIO = 0.10; TEST_RATIO = 0.20; NUM_WORKERS = 0

USE_AUGMENT = True
AUG_NOISE_STD = 0.02; AUG_SHIFT_MAX = 10
AUG_SCALE_LOW = 0.95; AUG_SCALE_HIGH = 1.05; AUG_BASELINE_DRIFT = 0.12

DROPOUT = 0.35; GRAD_CLIP_NORM = 3.0; EMBED_DIM = 128

# Synthetic mixture generation
N_SYNTHETIC_PER_COMBO = 200; N_SYNTHETIC_TERNARY = 100
MIXTURE_RATIO_MIN = 0.10; MIXTURE_RATIO_MAX = 0.90
SYNTH_NOISE_STD_RATIO = 0.01
ENABLE_EXTRA_LOW_R6G_MB = True; N_EXTRA_LOW_R6G_MB = 300
LOW_R6G_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20]

# Benchmark and inference thresholds
BENCHMARK_THRESHOLD = 0.50
OVR_FINAL_THRESHOLD = 0.40
R6G_WEAK_THRESHOLD_WHEN_MB_DOMINANT = 0.25

TARGET_ANALYTES = ["r6g", "methylene_blue", "adenine"]
BACKGROUND_CLASSES = ["di_water"]

OUTPUT_DIR = os.path.join(DATA_ROOT, "_CNN_raman_v3_mixture")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "cnn_mixture.pt")
META_PATH = os.path.join(OUTPUT_DIR, "meta.json")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE.startswith("cuda")
print(f"DEVICE={DEVICE}  RETRAIN={RETRAIN_MODEL}")


# =========================================================
# UTILS
# =========================================================
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

def _is_jupyter():
    try:
        from IPython import get_ipython
        if get_ipython().__class__.__name__ in ("ZMQInteractiveShell","TerminalInteractiveShell"): return True
    except: pass
    return len(sys.argv)>0 and "jupyter" in sys.argv[0].lower()

def _parse_cli_args():
    if _is_jupyter(): return TEST_FILE, TEST_DARK_FILE
    args=[a for a in sys.argv[1:] if not a.startswith("-")]
    tf=args[0].strip() if len(args)>=1 and args[0].strip() else TEST_FILE
    td=args[1].strip() if len(args)>=2 and args[1].strip() else TEST_DARK_FILE
    return tf,td

ERROR_PHRASE="image data error code 0x0001"
def strip_prefix(line):
    line=line.strip(); line=re.sub(r"^\[[^\]]+\]\s+\[Backend\]\s*","",line); line=re.sub(r"^\[[^\]]+\]\s*","",line); return line.strip()

def load_dark_spectrum_from_file(dark_path):
    if not dark_path or not Path(dark_path).exists(): return np.array([],dtype=np.float64)
    intensities=[]
    with open(dark_path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or line.startswith("index"): continue
            parts=line.split("\t")
            if len(parts)>=3:
                try: intensities.append(float(parts[2]))
                except: continue
    return np.array(intensities,dtype=np.float64)

def parse_portable_log_to_spectra(log_file):
    # Keep one entry per spectrum block, including invalid blocks.
    with open(log_file, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = f.read().splitlines()
    spectra, failed = [], 0
    i = 0
    hex_pattern = re.compile(r"[0-9A-Fa-f]{2,4}(?:\s+[0-9A-Fa-f]{2,4})*")
    while i < len(lines):
        line = strip_prefix(lines[i]).strip()
        if not line.startswith("Image Data"):
            i += 1
            continue
        payload = line[len("Image Data"):].strip()
        tokens = payload.split() if hex_pattern.fullmatch(payload) else []
        bad = bool(payload) and not bool(tokens)
        j = i + 1
        while j < len(lines):
            nxt = strip_prefix(lines[j]).strip()
            if nxt.startswith("Image Data") or nxt.startswith("Stop Capture"):
                break
            if nxt and not nxt.startswith("#"):
                if ERROR_PHRASE and str(ERROR_PHRASE).lower() in lines[j].lower():
                    bad = True
                if hex_pattern.fullmatch(nxt):
                    tokens.extend(nxt.split())
            j += 1
        # Explicitly accept 2048 words; do not guess byte order.
        if bad or len(tokens) != 2048:
            spectra.append(np.array([], dtype=np.float64))
            failed += 1
        else:
            spectra.append(np.array([int(t, 16) for t in tokens], dtype=np.float64))
        i = j
    return spectra, failed


def classify_target_scores(scores, threshold):
    # This is a detection label, not proof of chemical purity.
    if any(not np.isfinite(v) or v < 0 or v > 1 for v in scores.values()):
        return "UNKNOWN", []
    detected = [name for name, value in scores.items() if value >= threshold]
    if len(detected) == 0:
        return "UNKNOWN", []
    if len(detected) == 1:
        return "SINGLE_TARGET", detected
    return "MULTI_TARGET_CANDIDATE", detected


def export_scan_analysis(spec_path, dark_path, spectra, per_block, rejected_blocks, model_name):
    from datetime import datetime
    import tempfile
    target_names = [id_to_class[c] for c in TARGET_CIDS]
    accepted = {b["idx"]: b for b in per_block}
    rejected = {b["idx"]: b["reason"] for b in rejected_blocks}
    rows = []
    # Recover coordinates only when the sidecar spectrum matches exactly.
    single_match = re.fullmatch(r"point_(\d+)", Path(spec_path).stem)
    for idx, raw in enumerate(spectra):
        point_id = int(single_match.group(1)) if single_match else idx + 1
        row = {
            "spectrum_index": idx, "point_id": point_id,
            "x_steps": None, "y_steps": None, "z_steps": None,
            "position_verified": False, "label": "REJECTED",
            "detected": [], "scores": None,
            "reason": rejected.get(idx, "invalid spectrum or preprocessing"),
            "full_scale_pixels": int(np.sum(raw == 65535)),
        }
        sidecar = Path(spec_path).parent / f"point_{point_id:04d}.json"
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if raw.size == 2048 and np.array_equal(raw, np.asarray(meta["intensity"])):
                    row.update(
                        x_steps=meta.get("controller_x_steps"),
                        y_steps=meta.get("controller_y_steps"),
                        z_steps=meta.get("controller_z_steps"),
                        position_verified=True,
                    )
            except (ValueError, KeyError, OSError):
                pass
        if idx in accepted:
            block = accepted[idx]
            values = block["ovr_probs"] if "ovr_probs" in block else block["probs"]
            scores = {id_to_class[c]: float(values[c]) for c in TARGET_CIDS}
            label, detected = classify_target_scores(scores, OVR_FINAL_THRESHOLD)
            row.update(label=label, detected=detected,
                       scores={k: v if np.isfinite(v) else None for k, v in scores.items()},
                       reason="" if label != "UNKNOWN" else "no confident target or invalid scores")
        rows.append(row)
        print(f"[POINT {point_id:02d}] {row['label']}: {', '.join(row['detected']) or 'none'}"
              f" | scores={row['scores']} | {row['reason']}")

    counts = {label: sum(r["label"] == label for r in rows) for label in
              ("SINGLE_TARGET", "MULTI_TARGET_CANDIDATE", "UNKNOWN", "REJECTED")}
    summary = {
        "model": model_name,
        "source_file": str(Path(spec_path).resolve()),
        "dark_file": str(Path(dark_path).resolve()) if dark_path else None,
        "threshold": OVR_FINAL_THRESHOLD,
        "weak_r6g_rule_enabled": False,
        "total_blocks": len(rows),
        "counts": counts,
        "targets_across_scan": sorted({n for r in rows for n in r["detected"]}),
        "note": "Targets at different positions do not prove a mixture. Multi-target labels require validation.",
        "points": rows,
    }
    parent = Path(OUTPUT_DIR) / "scan_analysis"
    parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(
        prefix=Path(spec_path).stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_",
        dir=str(parent)
    ))
    (out_dir / "point_predictions.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (out_dir / "point_predictions.txt").open("w", encoding="utf-8") as f:
        f.write("point_id\tx_steps\ty_steps\tz_steps\tlabel\tdetected\t"
                + "\t".join(target_names) + "\treason\tfull_scale_pixels\n")
        for row in rows:
            scores = row["scores"] or {}
            values = [str(row["point_id"])] + [
                "" if row[k] is None else str(row[k]) for k in ("x_steps", "y_steps", "z_steps")
            ] + [row["label"], "+".join(row["detected"])] + [
                "" if scores.get(n) is None else f"{scores[n]:.6f}" for n in target_names
            ] + [row["reason"], str(row["full_scale_pixels"])]
            f.write("\t".join(values) + "\n")
    print("\nSCAN SUMMARY:", counts)
    print("Targets across positions:", summary["targets_across_scan"])
    print(summary["note"])
    print("Saved scan analysis:", out_dir)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in target_names:
            ys = [r["scores"].get(name) if r["scores"] else np.nan for r in rows]
            ax.plot([r["point_id"] for r in rows], ys, marker="o", label=name)
        ax.axhline(OVR_FINAL_THRESHOLD, color="gray", linestyle="--", label="Point threshold")
        ax.set(xlabel="Original point ID (1-based)", ylabel="Target score",
               ylim=(0, 1.05), title=model_name + " point-level target scores")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "point_probabilities.png", dpi=200)
        plt.show()
    except Exception as exc:
        print("Plot warning:", exc)
    return summary



def moving_average(y,window):
    window=int(max(3,window));
    if window%2==0: window+=1
    pad=window//2; ypad=np.pad(y,(pad,pad),mode="edge")
    return np.convolve(ypad,np.ones(window,dtype=np.float64)/window,mode="valid")

def rolling_minimum(y,window):
    window=int(max(3,window))
    if window%2==0: window+=1
    pad=window//2; ypad=np.pad(y,(pad,pad),mode="edge")
    out=np.empty_like(y,dtype=np.float64)
    for i in range(y.size): out[i]=np.min(ypad[i:i+window])
    return out

def baseline_correction_fast(y,window=15,smooth=15):
    y=np.asarray(y,dtype=np.float64)
    if y.size<5: return y.copy()
    return y-moving_average(rolling_minimum(y,window),smooth)

def normalize_min_max(y):
    y=np.asarray(y,dtype=np.float64); ymin,ymax=y.min(),y.max(); d=ymax-ymin
    return np.zeros_like(y) if d<1e-12 else (y-ymin)/d

def preprocess_two_versions(raw_spectrum,dark_raw):
    s=np.asarray(raw_spectrum,np.float64)
    if s.size<TARGET_LEN: return None,None,None
    s2048=s[:TARGET_LEN]
    d2048=dark_raw[:TARGET_LEN] if dark_raw.size>=TARGET_LEN else dark_raw
    raw_dark=s2048-d2048 if d2048.size>=TARGET_LEN else s2048.copy()
    if DROP_N>=raw_dark.size: return None,None,None
    raw_dark=raw_dark[DROP_N:]
    quality={"ptp":float(np.ptp(raw_dark)),"std":float(np.std(raw_dark))}
    bl=baseline_correction_fast(raw_dark,BASELINE_WINDOW,BASELINE_SMOOTH) if DO_BASELINE else raw_dark.copy()
    mm=normalize_min_max(bl)
    return bl.astype(np.float32),mm.astype(np.float32),quality


# =========================================================
# STEP 1: Load data + generate synthetic mixtures
# =========================================================
class_names=list(TRAINING_MINMAX_NPY_FILES.keys())
class_to_id={c:i for i,c in enumerate(class_names)}
id_to_class={i:c for c,i in class_to_id.items()}
NUM_CLASSES=len(class_names)
TARGET_CIDS=[class_to_id[c] for c in TARGET_ANALYTES]

print("="*70); print("Step 1: Load data + generate synthetic mixtures"); print("="*70)

pure_data_raw={}
X_mm_parts,X_raw_parts,Y_parts=[],[],[]
for cls in class_names:
    dm=np.load(TRAINING_MINMAX_NPY_FILES[cls]).astype(np.float32)
    dr=np.load(TRAINING_RAW_NPY_FILES[cls]).astype(np.float32)
    pure_data_raw[cls]=dr
    X_mm_parts.append(dm); X_raw_parts.append(dr)
    Y_parts.append(np.full(dm.shape[0],class_to_id[cls],dtype=np.int64))
    print(f"  {cls:20s} shape={dm.shape}")

X_pure_mm=np.vstack(X_mm_parts).astype(np.float32)
Y_pure=np.concatenate(Y_parts)
FEATURE_DIM=X_pure_mm.shape[1]
n_pure=X_pure_mm.shape[0]

# Generate synthetic mixtures in raw space, then normalize
rng_mix=np.random.RandomState(SEED)
X_all_list=[X_pure_mm.copy()]  # start with pure (already normalized)
Y_ml_list=[]
for i in range(n_pure):
    label=np.zeros(NUM_CLASSES,dtype=np.float32); label[Y_pure[i]]=1.0; Y_ml_list.append(label)

n_synth=0
for cid_a,cid_b in combinations(range(NUM_CLASSES),2):
    da,db=pure_data_raw[id_to_class[cid_a]],pure_data_raw[id_to_class[cid_b]]
    for _ in range(N_SYNTHETIC_PER_COMBO):
        r=rng_mix.uniform(MIXTURE_RATIO_MIN,MIXTURE_RATIO_MAX)
        mix=r*da[rng_mix.randint(len(da))]+(1-r)*db[rng_mix.randint(len(db))]
        if SYNTH_NOISE_STD_RATIO>0: mix+=rng_mix.normal(0,SYNTH_NOISE_STD_RATIO*max(float(np.std(mix)),1e-6),mix.shape).astype(np.float32)
        mix_mm=normalize_min_max(mix).astype(np.float32)
        X_all_list.append(mix_mm.reshape(1,-1))
        lb=np.zeros(NUM_CLASSES,dtype=np.float32); lb[cid_a]=1.0; lb[cid_b]=1.0; Y_ml_list.append(lb); n_synth+=1

# Extra low R6G+MB
if ENABLE_EXTRA_LOW_R6G_MB and "r6g" in class_to_id and "methylene_blue" in class_to_id:
    dr6g,dmb=pure_data_raw["r6g"],pure_data_raw["methylene_blue"]
    cid_r,cid_m=class_to_id["r6g"],class_to_id["methylene_blue"]
    per_r=max(1,N_EXTRA_LOW_R6G_MB//len(LOW_R6G_RATIOS))
    for ratio in LOW_R6G_RATIOS:
        for _ in range(per_r):
            mix=ratio*dr6g[rng_mix.randint(len(dr6g))]+(1-ratio)*dmb[rng_mix.randint(len(dmb))]
            if SYNTH_NOISE_STD_RATIO>0: mix+=rng_mix.normal(0,SYNTH_NOISE_STD_RATIO*max(float(np.std(mix)),1e-6),mix.shape).astype(np.float32)
            X_all_list.append(normalize_min_max(mix).astype(np.float32).reshape(1,-1))
            lb=np.zeros(NUM_CLASSES,dtype=np.float32); lb[cid_r]=1.0; lb[cid_m]=1.0; Y_ml_list.append(lb); n_synth+=1

# Ternary
for combo in combinations(range(NUM_CLASSES),3):
    for _ in range(N_SYNTHETIC_TERNARY):
        specs=[pure_data_raw[id_to_class[c]][rng_mix.randint(len(pure_data_raw[id_to_class[c]]))] for c in combo]
        while True:
            ratios=rng_mix.dirichlet([1.0]*len(combo))
            if all(r>=MIXTURE_RATIO_MIN for r in ratios): break
        mix=sum(r*s for r,s in zip(ratios,specs))
        if SYNTH_NOISE_STD_RATIO>0: mix+=rng_mix.normal(0,SYNTH_NOISE_STD_RATIO*max(float(np.std(mix)),1e-6),mix.shape).astype(np.float32)
        X_all_list.append(normalize_min_max(mix).astype(np.float32).reshape(1,-1))
        lb=np.zeros(NUM_CLASSES,dtype=np.float32)
        for c in combo: lb[c]=1.0
        Y_ml_list.append(lb); n_synth+=1

X_all=np.vstack(X_all_list).astype(np.float32)
Y_ml=np.stack(Y_ml_list).astype(np.float32)
print(f"\n  Pure={n_pure}  Synthetic={n_synth}  Total={X_all.shape[0]}")


# =========================================================
# STEP 2: Split
# =========================================================
print("\n"+"="*70); print("Step 2: Split"); print("="*70)
all_idx=list(range(X_all.shape[0]))
rng_s=random.Random(SEED); rng_s.shuffle(all_idx)
n_tr=int(len(all_idx)*TRAIN_RATIO); n_va=int(len(all_idx)*VAL_RATIO)
idx_tr=all_idx[:n_tr]; idx_va=all_idx[n_tr:n_tr+n_va]; idx_te=all_idx[n_tr+n_va:]
print(f"  train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)}")


# =========================================================
# DATASET
# =========================================================
def augment_spectrum_1d(x):
    x=x.copy()*np.float32(np.random.uniform(AUG_SCALE_LOW,AUG_SCALE_HIGH))
    shift=int(np.random.randint(-AUG_SHIFT_MAX,AUG_SHIFT_MAX+1))
    if shift: x=np.roll(x,shift)
    x+=np.random.normal(0,AUG_NOISE_STD,x.shape).astype(np.float32)
    x+=np.float32(np.random.uniform(-AUG_BASELINE_DRIFT,AUG_BASELINE_DRIFT))*np.linspace(-1,1,x.size,dtype=np.float32)
    return normalize_min_max(x).astype(np.float32)

class RamanDatasetML(Dataset):
    def __init__(self,X,Y_ml,indices,augment=False):
        self.X=X; self.Y_ml=Y_ml; self.indices=indices; self.augment=augment
    def __len__(self): return len(self.indices)
    def __getitem__(self,k):
        i=self.indices[k]; x=self.X[i].copy()
        if self.augment: x=augment_spectrum_1d(x)
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(self.Y_ml[i])

train_loader=DataLoader(RamanDatasetML(X_all,Y_ml,idx_tr,USE_AUGMENT),BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
val_loader=DataLoader(RamanDatasetML(X_all,Y_ml,idx_va,False),BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
test_loader=DataLoader(RamanDatasetML(X_all,Y_ml,idx_te,False),BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)


# =========================================================
# MODEL (IDENTICAL to v2)
# =========================================================
class SEBlock1D(nn.Module):
    def __init__(self,ch,r=8):
        super().__init__(); h=max(4,ch//r)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Sequential(nn.Linear(ch,h),nn.ReLU(True),nn.Linear(h,ch),nn.Sigmoid())
    def forward(self,x):
        b,c,_=x.shape; return x*self.fc(self.pool(x).view(b,c)).view(b,c,1)

class ResBlock1D(nn.Module):
    def __init__(self,ic,oc,ks,stride,drop):
        super().__init__(); p=ks//2
        self.c1=nn.Conv1d(ic,oc,ks,stride,p,bias=False); self.b1=nn.BatchNorm1d(oc)
        self.c2=nn.Conv1d(oc,oc,ks,1,p,bias=False); self.b2=nn.BatchNorm1d(oc)
        self.se=SEBlock1D(oc); self.drop=nn.Dropout(drop)
        self.sc=nn.Sequential(nn.Conv1d(ic,oc,1,stride,bias=False),nn.BatchNorm1d(oc)) if stride!=1 or ic!=oc else nn.Identity()
        self.act=nn.GELU()
    def forward(self,x):
        out=self.act(self.b1(self.c1(x))); out=self.se(self.drop(self.b2(self.c2(out))))
        return self.act(out+self.sc(x))

class RamanNet(nn.Module):
    def __init__(self,nc=4,base=32,drop=0.35):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(1,base,7,padding=3,bias=False),nn.BatchNorm1d(base),nn.GELU(),
                                nn.Conv1d(base,base,5,padding=2,bias=False),nn.BatchNorm1d(base),nn.GELU())
        self.s1=nn.Sequential(ResBlock1D(base,base,7,1,drop*0.5),nn.MaxPool1d(2))
        self.s2=nn.Sequential(ResBlock1D(base,base*2,5,1,drop),nn.MaxPool1d(2))
        self.s3=nn.Sequential(ResBlock1D(base*2,base*4,3,1,drop),nn.MaxPool1d(2))
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Linear(base*4,EMBED_DIM),nn.GELU(),nn.Dropout(drop))
        self.cls=nn.Linear(EMBED_DIM,nc)  # nc outputs, one per class
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d): nn.init.kaiming_normal_(m.weight,nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm1d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward_embedding(self,x):
        return self.head(self.pool(self.s3(self.s2(self.s1(self.stem(x))))).flatten(1))
    def forward(self,x):
        emb=self.forward_embedding(x); return self.cls(emb),emb



# =========================================================
# PLOTTING HELPERS (test-set multi-label + real-file probabilities)
# =========================================================
def _plot_safe_name(prefix):
    return re.sub(r"[^A-Za-z0-9_]+", "_", prefix.lower()).strip("_")

def _target_label_name(row):
    names=[id_to_class[c] for c in TARGET_CIDS if row[c] >= 0.5]
    return "+".join(names) if names else "none/bg"

def _all_target_label_names():
    names=[id_to_class[c] for c in TARGET_CIDS]
    labels=["none/bg"]
    for r in range(1,len(names)+1):
        for combo in combinations(names,r):
            labels.append("+".join(combo))
    return labels

def _annotate_matrix(ax, cm):
    vmax=max(float(np.max(cm)),1.0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(int(cm[i,j])),ha="center",va="center",
                    color="white" if cm[i,j] > 0.55*vmax else "black",fontsize=10)

def plot_multilabel_test_figures(all_true, all_preds, prefix="MODEL", save_dir=OUTPUT_DIR):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
        os.makedirs(save_dir, exist_ok=True)
        slug=_plot_safe_name(prefix)

        y_true_t=all_true[:,TARGET_CIDS].astype(int)
        y_pred_t=all_preds[:,TARGET_CIDS].astype(int)
        mcm=multilabel_confusion_matrix(y_true_t,y_pred_t)

        fig,axes=plt.subplots(1,len(TARGET_CIDS),figsize=(4.2*len(TARGET_CIDS),3.6))
        if len(TARGET_CIDS)==1:
            axes=[axes]
        for ax,cm,cid in zip(axes,mcm,TARGET_CIDS):
            ax.imshow(cm,cmap="Blues")
            ax.set_title(f"{prefix}: {id_to_class[cid]}")
            ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0","Pred 1"])
            ax.set_yticks([0,1]); ax.set_yticklabels(["True 0","True 1"])
            _annotate_matrix(ax,cm)
        fig.suptitle(f"{prefix} target-wise multi-label confusion matrices")
        fig.tight_layout()
        out1=os.path.join(save_dir,f"{slug}_target_confusion_matrices.png")
        fig.savefig(out1,dpi=200,bbox_inches="tight")
        plt.show()
        print(f"  Saved plot: {out1}")

        true_names=[_target_label_name(row) for row in all_true]
        pred_names=[_target_label_name(row) for row in all_preds]
        labels=_all_target_label_names()
        cm=confusion_matrix(true_names,pred_names,labels=labels)
        fig,ax=plt.subplots(figsize=(max(7,0.8*len(labels)),max(6,0.7*len(labels))))
        ax.imshow(cm,cmap="Purples")
        ax.set_title(f"{prefix} label-set confusion matrix")
        ax.set_xlabel("Predicted target set")
        ax.set_ylabel("True target set")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=35,ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        _annotate_matrix(ax,cm)
        fig.tight_layout()
        out2=os.path.join(save_dir,f"{slug}_labelset_confusion_matrix.png")
        fig.savefig(out2,dpi=200,bbox_inches="tight")
        plt.show()
        print(f"  Saved plot: {out2}")
    except Exception as e:
        print(f"  [Plot warning] Could not draw test-set matrices: {e}")

def plot_real_file_probabilities(per_block, avg_probs, prefix="MODEL", save_dir=OUTPUT_DIR):
    try:
        import matplotlib.pyplot as plt
        if not per_block:
            return
        os.makedirs(save_dir, exist_ok=True)
        slug=_plot_safe_name(prefix)
        xs=[b["idx"] for b in per_block]

        plt.figure(figsize=(12,5))
        for cid in TARGET_CIDS:
            ys=[float(b["probs"][cid]) for b in per_block]
            plt.plot(xs,ys,marker="o",markersize=3,label=id_to_class[cid],alpha=0.8)
        plt.axhline(OVR_FINAL_THRESHOLD,linestyle="--",alpha=0.55,label=f"threshold={OVR_FINAL_THRESHOLD}")
        plt.axhline(R6G_WEAK_THRESHOLD_WHEN_MB_DOMINANT,linestyle=":",alpha=0.55,label=f"weak r6g={R6G_WEAK_THRESHOLD_WHEN_MB_DOMINANT}")
        plt.xlabel("Spectrum index")
        plt.ylabel("Sigmoid / OvR probability")
        plt.title(f"{prefix} real-file target probabilities")
        plt.ylim(0,1.05)
        plt.legend(loc="center right")
        plt.tight_layout()
        out1=os.path.join(save_dir,f"{slug}_real_file_probabilities.png")
        plt.savefig(out1,dpi=200,bbox_inches="tight")
        plt.show()
        print(f"  Saved plot: {out1}")

        names=[id_to_class[c] for c in TARGET_CIDS]
        vals=[float(avg_probs[c]) for c in TARGET_CIDS]
        plt.figure(figsize=(7.5,4.5))
        bars=plt.bar(names,[v*100 for v in vals],edgecolor="black",alpha=0.85)
        plt.axhline(OVR_FINAL_THRESHOLD*100,linestyle="--",alpha=0.55,label=f"threshold={OVR_FINAL_THRESHOLD}")
        plt.axhline(R6G_WEAK_THRESHOLD_WHEN_MB_DOMINANT*100,linestyle=":",alpha=0.55,label=f"weak r6g={R6G_WEAK_THRESHOLD_WHEN_MB_DOMINANT}")
        plt.ylabel("Average probability (%)")
        plt.title(f"{prefix} final target probabilities")
        plt.ylim(0,105)
        plt.legend()
        for bar,val in zip(bars,vals):
            plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{val*100:.1f}%",ha="center",va="bottom")
        plt.tight_layout()
        out2=os.path.join(save_dir,f"{slug}_real_file_avg_probabilities.png")
        plt.savefig(out2,dpi=200,bbox_inches="tight")
        plt.show()
        print(f"  Saved plot: {out2}")
    except Exception as e:
        print(f"  [Plot warning] Could not draw real-file probability plots: {e}")


# =========================================================
# TRAINING (multi-label: BCEWithLogitsLoss)
# =========================================================
@torch.no_grad()
def eval_multilabel(model,loader,threshold=BENCHMARK_THRESHOLD):
    model.eval(); total=0; correct=0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        logits,_=model(x); preds=(torch.sigmoid(logits)>=threshold).float()
        correct+=int((preds==y).all(dim=1).sum().item()); total+=y.shape[0]
    return correct/max(total,1)

def train_model():
    model=RamanNet(nc=NUM_CLASSES,drop=DROPOUT).to(DEVICE)
    # Compute pos_weight for imbalanced multi-label
    pos_counts=Y_ml[idx_tr].sum(axis=0)
    neg_counts=len(idx_tr)-pos_counts
    pos_weight=torch.tensor(neg_counts/np.maximum(pos_counts,1),dtype=torch.float32,device=DEVICE)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=5,min_lr=1e-6)

    best_val=-1.0; best_state=None; bad=0
    print("\n"+"="*70); print("Step 3: Training CNN (multi-label)"); print("="*70)
    for ep in range(1,EPOCHS+1):
        model.train(); losses=[]; ep_t0=time.perf_counter()
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits,_=model(x); loss=criterion(logits,y)
            if not torch.isfinite(loss): continue
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP_NORM); opt.step()
            losses.append(float(loss.item()))
        va=eval_multilabel(model,val_loader)
        sched.step(va)
        print(f"epoch {ep:02d}/{EPOCHS} loss={np.mean(losses):.4f} val_exact_match={va:.3f} lr={opt.param_groups[0]['lr']:.2e} time={time.perf_counter()-ep_t0:.1f}s")
        if va>best_val+1e-6: best_val=va; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=16: print("early stop"); break
    if best_state: model.load_state_dict(best_state)

    # Test evaluation
    te=eval_multilabel(model,test_loader)
    print(f"\nBest val exact_match: {best_val:.4f}")
    print(f"TEST exact_match:     {te:.4f}")

    # Per-class metrics on test set
    model.eval()
    all_probs,all_true=[],[]
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(DEVICE); logits,_=model(x)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(y.numpy())
    all_probs=np.vstack(all_probs); all_true=np.vstack(all_true).astype(np.int64)
    all_preds=(all_probs>=BENCHMARK_THRESHOLD).astype(np.int64)
    print("\nPer-class metrics (test set):")
    for cid in range(NUM_CLASSES):
        tp=int(((all_preds[:,cid]==1)&(all_true[:,cid]==1)).sum())
        fp=int(((all_preds[:,cid]==1)&(all_true[:,cid]==0)).sum())
        fn=int(((all_preds[:,cid]==0)&(all_true[:,cid]==1)).sum())
        prec=tp/(tp+fp) if tp+fp>0 else 0; rec=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
        auc=roc_auc_score(all_true[:,cid],all_probs[:,cid]) if np.unique(all_true[:,cid]).size==2 else float("nan")
        print(f"  {id_to_class[cid]:20s} P={prec:.4f} R={rec:.4f} F1={f1:.4f} AUC={auc:.4f}  TP={tp} FP={fp} FN={fn}")

    # Standardized Figure 4 exports: target-only metrics use R6G, MB, and adenine.
    y_true_target=all_true[:,TARGET_CIDS]
    y_prob_target=all_probs[:,TARGET_CIDS]
    y_pred_target=all_preds[:,TARGET_CIDS]
    mixture_mask=y_true_target.sum(axis=1)>1
    n_mixture=int(mixture_mask.sum())
    per_target_rows=[]
    for j,cid in enumerate(TARGET_CIDS):
        yt=y_true_target[:,j]; yp=y_pred_target[:,j]; pr=y_prob_target[:,j]
        tp=int(((yp==1)&(yt==1)).sum()); fp=int(((yp==1)&(yt==0)).sum())
        fn=int(((yp==0)&(yt==1)).sum()); tn=int(((yp==0)&(yt==0)).sum())
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        auc=float(roc_auc_score(yt,pr)) if np.unique(yt).size==2 else float("nan")
        per_target_rows.append({
            "model":"CNN", "target":id_to_class[cid],
            "precision":precision, "recall":recall, "f1":f1, "roc_auc":auc,
            "tp":tp, "fp":fp, "fn":fn, "tn":tn,
            "positive_support":int(yt.sum()), "n_test":int(len(yt)),
            "benchmark_threshold":BENCHMARK_THRESHOLD,
        })
    target_macro_f1=float(np.mean([row["f1"] for row in per_target_rows]))
    target_macro_roc_auc=float(np.mean([row["roc_auc"] for row in per_target_rows]))
    r6g_recall=next(row["recall"] for row in per_target_rows if row["target"]=="r6g")
    mixture_exact=(float(np.mean(np.all(y_pred_target[mixture_mask]==y_true_target[mixture_mask],axis=1))) if n_mixture else float("nan"))
    target_exact=float(np.mean(np.all(y_pred_target==y_true_target,axis=1)))
    all_label_exact=float(np.mean(np.all(all_preds==all_true,axis=1)))
    overall_row={
        "model":"CNN", "n_test":int(len(all_true)), "n_mixture":n_mixture,
        "target_macro_f1":target_macro_f1, "target_macro_roc_auc":target_macro_roc_auc,
        "r6g_recall":r6g_recall, "mixture_exact_match":mixture_exact,
        "target_exact_match":target_exact, "all_label_exact_match":all_label_exact,
        "benchmark_threshold":BENCHMARK_THRESHOLD,
    }
    overall_path=os.path.join(OUTPUT_DIR,"figure4_overall_metrics.csv")
    per_target_path=os.path.join(OUTPUT_DIR,"figure4_per_target_metrics.csv")
    predictions_path=os.path.join(OUTPUT_DIR,"figure4_test_predictions.npz")
    with open(overall_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(overall_row.keys())); writer.writeheader(); writer.writerow(overall_row)
    with open(per_target_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(per_target_rows[0].keys())); writer.writeheader(); writer.writerows(per_target_rows)
    np.savez_compressed(predictions_path,test_indices=np.asarray(idx_te),target_names=np.asarray([id_to_class[c] for c in TARGET_CIDS]),y_true=y_true_target,y_prob=y_prob_target,y_pred=y_pred_target)
    print(f"  Target macro-F1={target_macro_f1:.4f}  Target macro ROC-AUC={target_macro_roc_auc:.4f}")
    print(f"  Multi-target mixture exact-match={mixture_exact:.4f} (n={n_mixture})")
    print(f"  Saved Figure 4 metrics: {overall_path}")

    plot_multilabel_test_figures(all_true, all_preds, prefix="CNN", save_dir=OUTPUT_DIR)

    torch.save(model.state_dict(),MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")
    return model


# =========================================================
# INFERENCE (multi-label, same logic as SVM)
# =========================================================
@torch.no_grad()
def predict_file(model, spec_path, dark_path):
    model.eval()
    dark_raw = load_dark_spectrum_from_file(dark_path) if dark_path else np.array([])
    if dark_path and dark_raw.size != 2048:
        raise ValueError("Dark file must contain 2048 valid intensities")
    spectra, failed = parse_portable_log_to_spectra(spec_path)
    print(f"Parsed: {len(spectra)} spectrum blocks, {failed} invalid")
    print("Pixel counts:", [len(s) for s in spectra])
    per_block, rejected_blocks = [], []
    for idx, raw in enumerate(spectra):
        if raw.size != 2048:
            rejected_blocks.append({"idx": idx, "reason": "invalid pixel count"})
            continue
        _, v_mm, quality = preprocess_two_versions(raw, dark_raw)
        if v_mm is None or np.allclose(v_mm, 0):
            rejected_blocks.append({"idx": idx, "reason": "preprocessing failed"})
            continue
        if ENABLE_QUALITY_FILTER and (quality["ptp"] < RAW_MIN_PTP or quality["std"] < RAW_MIN_STD):
            rejected_blocks.append({"idx": idx, "reason": "low signal"})
            continue
        xt = torch.from_numpy(v_mm).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits, _ = model(xt)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        per_block.append({"idx": idx, "probs": probs, "quality": quality})
    return export_scan_analysis(
        spec_path, dark_path, spectra, per_block, rejected_blocks, "CNN"
    )


# =========================================================
# RUN
# =========================================================
RUNTIME_TEST_FILE,RUNTIME_TEST_DARK=_parse_cli_args()
print(f"\nTest: {RUNTIME_TEST_FILE}")

if RETRAIN_MODEL or not os.path.exists(MODEL_PATH):
    model=train_model()
else:
    model=RamanNet(nc=NUM_CLASSES,drop=DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model=model.to(DEVICE); model.eval()
    print(f"Loaded: {MODEL_PATH}")

print("\n"+"="*70); print("Inference"); print("="*70)
predict_file(model,RUNTIME_TEST_FILE,RUNTIME_TEST_DARK)
