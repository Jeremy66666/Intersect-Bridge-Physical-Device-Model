"""
ViT-1D Raman classifier v2 — multi-label mixture detection
Transformer architecture is IDENTICAL to v1.
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
from sklearn.metrics import roc_auc_score

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

DROP_N = 90; TARGET_LEN_RAW = 2048
FEATURE_DIM_AFTER_DROP = TARGET_LEN_RAW - DROP_N  # 1958
TARGET_LEN = 2048  # pad to 2048 for clean patch division
PATCH_SIZE = 64; NUM_PATCHES = TARGET_LEN // PATCH_SIZE  # 32

DO_BASELINE = True; BASELINE_WINDOW = 15; BASELINE_SMOOTH = 15

ENABLE_QUALITY_FILTER = True; RAW_MIN_PTP = 100.0; RAW_MIN_STD = 15.0

BATCH_SIZE = 32; EPOCHS = 100; LR = 5e-4; WEIGHT_DECAY = 1e-4; SEED = 42
WARMUP_EPOCHS = 5
TRAIN_RATIO = 0.70; VAL_RATIO = 0.10; TEST_RATIO = 0.20; NUM_WORKERS = 0

USE_AUGMENT = True
AUG_NOISE_STD = 0.02; AUG_SHIFT_MAX = 10
AUG_SCALE_LOW = 0.95; AUG_SCALE_HIGH = 1.05; AUG_BASELINE_DRIFT = 0.12

D_MODEL = 128; NUM_HEADS = 4; DEPTH = 4; MLP_RATIO = 4.0
DROPOUT = 0.15; ATTN_DROP = 0.10; DROP_PATH = 0.10
GRAD_CLIP_NORM = 3.0

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

OUTPUT_DIR = os.path.join(DATA_ROOT, "_ViT1D_raman_v2_mixture")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "vit1d_mixture.pt")
META_PATH = os.path.join(OUTPUT_DIR, "meta.json")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE.startswith("cuda")
print(f"DEVICE={DEVICE}  RETRAIN={RETRAIN_MODEL}")
print(f"ViT-1D  d_model={D_MODEL} heads={NUM_HEADS} depth={DEPTH} patch={PATCH_SIZE} tokens={NUM_PATCHES}")


# =========================================================
# UTILS (same as CNN version)
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
    with open(
        log_file,
        "r",
        encoding="utf-8-sig",
        errors="ignore"
    ) as f:
        lines = f.read().splitlines()

    spectra = []
    failed = 0
    i = 0

    error_phrase = str(ERROR_PHRASE).lower()

    # Match lines containing only hex values.
    hex_line_pattern = re.compile(
        r"[0-9A-Fa-f]{2,4}(?:\s+[0-9A-Fa-f]{2,4})*"
    )

    while i < len(lines):
        line_raw = lines[i]
        line = strip_prefix(line_raw).strip()

        # Skip blank lines and comments.
        if not line or line.startswith("#"):
            i += 1
            continue

        if error_phrase and error_phrase in line_raw.lower():
            failed += 1
            i += 1
            continue

        # Find the next spectrum.
        if not line.startswith("Image Data"):
            i += 1
            continue

        payload = line[len("Image Data"):].strip()
        tokens = []
        block_err = False

        if payload:
            if hex_line_pattern.fullmatch(payload):
                tokens.extend(payload.split())
            else:
                block_err = True

        j = i + 1

        # Read any continuation lines.
        while j < len(lines):
            nxt_raw = lines[j]
            nxt = strip_prefix(nxt_raw).strip()

            # Keep the next spectrum for the outer loop.
            if nxt.startswith("Image Data"):
                break

            if nxt.startswith("Stop Capture"):
                break

            if not nxt or nxt.startswith("#"):
                j += 1
                continue

            if error_phrase and error_phrase in nxt_raw.lower():
                block_err = True

            # Ignore non-data lines.
            if hex_line_pattern.fullmatch(nxt):
                tokens.extend(nxt.split())

            j += 1

        if block_err or not tokens:
            failed += 1
        else:
            vals = np.array(
                [int(token, 16) for token in tokens],
                dtype=np.int64
            )

            # Keep legacy byte-pair conversion.
            if vals.size % 2 == 0 and vals.max(initial=0) <= 255:
                le = vals[0::2] + (vals[1::2] << 8)
                be = (vals[0::2] << 8) + vals[1::2]

                if le.max(initial=0) >= be.max(initial=0):
                    v16 = le
                else:
                    v16 = be

                if v16.max(initial=0) > 255:
                    vals = v16.astype(np.int64)

            spectra.append(vals.astype(np.float64))

        # Advance without skipping the next spectrum.
        i = j

    return spectra, failed

def moving_average(y,w):
    w=int(max(3,w));
    if w%2==0: w+=1
    p=w//2; yp=np.pad(y,(p,p),mode="edge")
    return np.convolve(yp,np.ones(w,dtype=np.float64)/w,mode="valid")

def rolling_minimum(y,w):
    w=int(max(3,w))
    if w%2==0: w+=1
    p=w//2; yp=np.pad(y,(p,p),mode="edge")
    out=np.empty_like(y,dtype=np.float64)
    for i in range(y.size): out[i]=np.min(yp[i:i+w])
    return out

def baseline_correction_fast(y,window=15,smooth=15):
    y=np.asarray(y,dtype=np.float64)
    if y.size<5: return y.copy()
    return y-moving_average(rolling_minimum(y,window),smooth)

def normalize_min_max(y):
    y=np.asarray(y,dtype=np.float64); mn,mx=y.min(),y.max(); d=mx-mn
    return np.zeros_like(y) if d<1e-12 else (y-mn)/d

def pad_or_trim(x,tgt):
    x=np.asarray(x,dtype=np.float32)
    if x.size==tgt: return x
    if x.size>tgt: return x[:tgt]
    out=np.zeros(tgt,dtype=np.float32); out[:x.size]=x; return out

def preprocess_two_versions(raw_spectrum,dark_raw):
    s=np.asarray(raw_spectrum,np.float64)
    if s.size<TARGET_LEN_RAW: return None,None,None
    s2048=s[:TARGET_LEN_RAW]
    d2048=dark_raw[:TARGET_LEN_RAW] if dark_raw.size>=TARGET_LEN_RAW else dark_raw
    raw_dark=s2048-d2048 if d2048.size>=TARGET_LEN_RAW else s2048.copy()
    if DROP_N>=raw_dark.size: return None,None,None
    raw_dark=raw_dark[DROP_N:]
    quality={"ptp":float(np.ptp(raw_dark)),"std":float(np.std(raw_dark))}
    bl=baseline_correction_fast(raw_dark,BASELINE_WINDOW,BASELINE_SMOOTH) if DO_BASELINE else raw_dark.copy()
    mm=normalize_min_max(bl).astype(np.float32)
    mm_padded=pad_or_trim(mm,TARGET_LEN)
    return bl.astype(np.float32),mm_padded,quality


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
X_mm_parts,Y_parts=[],[]
for cls in class_names:
    dm=np.load(TRAINING_MINMAX_NPY_FILES[cls]).astype(np.float32)
    dr=np.load(TRAINING_RAW_NPY_FILES[cls]).astype(np.float32)
    pure_data_raw[cls]=dr
    # Pad minmax to TARGET_LEN for ViT
    dm_padded=np.zeros((dm.shape[0],TARGET_LEN),dtype=np.float32)
    dm_padded[:,:dm.shape[1]]=dm
    X_mm_parts.append(dm_padded)
    Y_parts.append(np.full(dm.shape[0],class_to_id[cls],dtype=np.int64))
    print(f"  {cls:20s} raw={dr.shape} padded_mm=({dm.shape[0]},{TARGET_LEN})")

X_pure_mm=np.vstack(X_mm_parts).astype(np.float32)
Y_pure=np.concatenate(Y_parts)
n_pure=X_pure_mm.shape[0]

# Generate synthetic mixtures in raw space → normalize → pad
rng_mix=np.random.RandomState(SEED)
X_all_list=[X_pure_mm.copy()]
Y_ml_list=[]
for i in range(n_pure):
    lb=np.zeros(NUM_CLASSES,dtype=np.float32); lb[Y_pure[i]]=1.0; Y_ml_list.append(lb)

def make_mix_and_append(mix_raw, label):
    if SYNTH_NOISE_STD_RATIO>0:
        mix_raw=mix_raw+rng_mix.normal(0,SYNTH_NOISE_STD_RATIO*max(float(np.std(mix_raw)),1e-6),mix_raw.shape).astype(np.float32)
    mm=normalize_min_max(mix_raw).astype(np.float32)
    mm_padded=pad_or_trim(mm,TARGET_LEN)
    X_all_list.append(mm_padded.reshape(1,-1))
    Y_ml_list.append(label)

n_synth=0
# Binary
for cid_a,cid_b in combinations(range(NUM_CLASSES),2):
    da,db=pure_data_raw[id_to_class[cid_a]],pure_data_raw[id_to_class[cid_b]]
    for _ in range(N_SYNTHETIC_PER_COMBO):
        r=rng_mix.uniform(MIXTURE_RATIO_MIN,MIXTURE_RATIO_MAX)
        mix=r*da[rng_mix.randint(len(da))]+(1-r)*db[rng_mix.randint(len(db))]
        lb=np.zeros(NUM_CLASSES,dtype=np.float32); lb[cid_a]=1.0; lb[cid_b]=1.0
        make_mix_and_append(mix,lb); n_synth+=1

# Extra low R6G+MB
if ENABLE_EXTRA_LOW_R6G_MB and "r6g" in class_to_id and "methylene_blue" in class_to_id:
    dr6g,dmb=pure_data_raw["r6g"],pure_data_raw["methylene_blue"]
    cid_r,cid_m=class_to_id["r6g"],class_to_id["methylene_blue"]
    per_r=max(1,N_EXTRA_LOW_R6G_MB//len(LOW_R6G_RATIOS))
    for ratio in LOW_R6G_RATIOS:
        for _ in range(per_r):
            mix=ratio*dr6g[rng_mix.randint(len(dr6g))]+(1-ratio)*dmb[rng_mix.randint(len(dmb))]
            lb=np.zeros(NUM_CLASSES,dtype=np.float32); lb[cid_r]=1.0; lb[cid_m]=1.0
            make_mix_and_append(mix,lb); n_synth+=1

# Ternary
for combo in combinations(range(NUM_CLASSES),3):
    for _ in range(N_SYNTHETIC_TERNARY):
        specs=[pure_data_raw[id_to_class[c]][rng_mix.randint(len(pure_data_raw[id_to_class[c]]))] for c in combo]
        while True:
            ratios=rng_mix.dirichlet([1.0]*len(combo))
            if all(r>=MIXTURE_RATIO_MIN for r in ratios): break
        mix=sum(r*s for r,s in zip(ratios,specs))
        lb=np.zeros(NUM_CLASSES,dtype=np.float32)
        for c in combo: lb[c]=1.0
        make_mix_and_append(mix,lb); n_synth+=1

X_all=np.vstack(X_all_list).astype(np.float32)
Y_ml=np.stack(Y_ml_list).astype(np.float32)
print(f"\n  Pure={n_pure} Synthetic={n_synth} Total={X_all.shape[0]}")


# =========================================================
# STEP 2: Split
# =========================================================
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
        # ViT takes flat 1D input (no channel dim)
        return torch.from_numpy(x), torch.from_numpy(self.Y_ml[i])

train_loader=DataLoader(RamanDatasetML(X_all,Y_ml,idx_tr,USE_AUGMENT),BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
val_loader=DataLoader(RamanDatasetML(X_all,Y_ml,idx_va,False),BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
test_loader=DataLoader(RamanDatasetML(X_all,Y_ml,idx_te,False),BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)


# =========================================================
# MODEL (IDENTICAL architecture to v1)
# =========================================================
class DropPath(nn.Module):
    def __init__(self,p=0.0): super().__init__(); self.p=p
    def forward(self,x):
        if self.p==0.0 or not self.training: return x
        keep=1.0-self.p; mask=torch.floor(torch.rand((x.shape[0],)+(1,)*(x.ndim-1),dtype=x.dtype,device=x.device)+keep)
        return x*mask/keep

class PatchEmbedding1D(nn.Module):
    def __init__(self,seq_len,patch_size,d_model):
        super().__init__()
        assert seq_len%patch_size==0
        self.patch_size=patch_size; self.num_patches=seq_len//patch_size
        self.proj=nn.Linear(patch_size,d_model,bias=False); self.norm=nn.LayerNorm(d_model)
    def forward(self,x):
        B,L=x.shape; return self.norm(self.proj(x.view(B,self.num_patches,self.patch_size)))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,attn_drop=0.0):
        super().__init__()
        assert d_model%num_heads==0
        self.num_heads=num_heads; self.head_dim=d_model//num_heads; self.scale=self.head_dim**-0.5
        self.qkv=nn.Linear(d_model,d_model*3,bias=False); self.proj=nn.Linear(d_model,d_model,bias=False)
        self.attn_drop=nn.Dropout(attn_drop)
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        attn=self.attn_drop((q@k.transpose(-2,-1)*self.scale).softmax(dim=-1))
        return self.proj((attn@v).transpose(1,2).reshape(B,N,C))

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,mlp_ratio=4.0,dropout=0.0,attn_drop=0.0,drop_path=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(d_model); self.attn=MultiHeadSelfAttention(d_model,num_heads,attn_drop)
        self.dp1=DropPath(drop_path); self.norm2=nn.LayerNorm(d_model)
        h=int(d_model*mlp_ratio)
        self.ffn=nn.Sequential(nn.Linear(d_model,h),nn.GELU(),nn.Dropout(dropout),nn.Linear(h,d_model),nn.Dropout(dropout))
        self.dp2=DropPath(drop_path)
    def forward(self,x):
        x=x+self.dp1(self.attn(self.norm1(x))); return x+self.dp2(self.ffn(self.norm2(x)))

class RamanViT1D(nn.Module):
    def __init__(self,seq_len=TARGET_LEN,patch_size=PATCH_SIZE,d_model=D_MODEL,
                 num_heads=NUM_HEADS,depth=DEPTH,mlp_ratio=MLP_RATIO,
                 num_classes=NUM_CLASSES,dropout=DROPOUT,attn_drop=ATTN_DROP,drop_path=DROP_PATH):
        super().__init__(); self.d_model=d_model
        self.patch_embed=PatchEmbedding1D(seq_len,patch_size,d_model)
        np_=seq_len//patch_size
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_embed=nn.Parameter(torch.zeros(1,np_+1,d_model))
        self.pos_drop=nn.Dropout(dropout)
        dpr=[x.item() for x in torch.linspace(0,drop_path,depth)]
        self.blocks=nn.ModuleList([TransformerBlock(d_model,num_heads,mlp_ratio,dropout,attn_drop,dpr[i]) for i in range(depth)])
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Sequential(nn.Linear(d_model,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,num_classes))
        self._init_weights()
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed,std=0.02); nn.init.trunc_normal_(self.cls_token,std=0.02)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight,std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward_embedding(self,x):
        B=x.shape[0]; x=self.patch_embed(x)
        x=torch.cat([self.cls_token.expand(B,-1,-1),x],dim=1)+self.pos_embed
        x=self.pos_drop(x)
        for blk in self.blocks: x=blk(x)
        return self.norm(x)[:,0]
    def forward(self,x):
        emb=self.forward_embedding(x); return self.head(emb),emb



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
    model=RamanViT1D().to(DEVICE)
    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters: {n_params:,}")

    pos_counts=Y_ml[idx_tr].sum(axis=0)
    neg_counts=len(idx_tr)-pos_counts
    pos_weight=torch.tensor(neg_counts/np.maximum(pos_counts,1),dtype=torch.float32,device=DEVICE)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,betas=(0.9,0.999))
    def lr_lambda(ep):
        if ep<WARMUP_EPOCHS: return float(ep+1)/float(WARMUP_EPOCHS)
        progress=float(ep-WARMUP_EPOCHS)/float(max(1,EPOCHS-WARMUP_EPOCHS))
        return max(0.05,0.5*(1.0+math.cos(math.pi*progress)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)

    best_val=-1.0; best_state=None; bad=0
    print("\n"+"="*70); print("Step 3: Training ViT-1D (multi-label)"); print("="*70)
    for ep in range(1,EPOCHS+1):
        model.train(); losses=[]; ep_t0=time.perf_counter()
        for x,y in train_loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits,_=model(x); loss=criterion(logits,y)
            if not torch.isfinite(loss): continue
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP_NORM); opt.step()
            losses.append(float(loss.item()))
        sched.step()
        va=eval_multilabel(model,val_loader)
        print(f"epoch {ep:02d}/{EPOCHS} loss={np.mean(losses):.4f} val_exact={va:.3f} lr={opt.param_groups[0]['lr']:.2e} time={time.perf_counter()-ep_t0:.1f}s")
        if va>best_val+1e-6: best_val=va; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=20: print("early stop"); break
    if best_state: model.load_state_dict(best_state)

    te=eval_multilabel(model,test_loader)
    print(f"\nBest val exact_match: {best_val:.4f}")
    print(f"TEST exact_match:     {te:.4f}")

    # Per-class metrics
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
            "model":"ViT-1D", "target":id_to_class[cid],
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
        "model":"ViT-1D", "n_test":int(len(all_true)), "n_mixture":n_mixture,
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

    plot_multilabel_test_figures(all_true, all_preds, prefix="ViT-1D", save_dir=OUTPUT_DIR)

    torch.save(model.state_dict(),MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")
    return model


# =========================================================
# INFERENCE (multi-label, same logic as SVM & CNN)
# =========================================================
def print_spatial_component_summary(spec_path, all_spectra, accepted, rejected):
    from datetime import datetime
    import tempfile

    # Point-level labels use the main threshold, not the weak R6G rule.
    groups = {}
    rows = []
    by_index = {b["idx"]: b for b in accepted}
    reasons = {b["idx"]: b["reason"] for b in rejected}
    single_file = re.fullmatch(r"point_(\d+)", Path(spec_path).stem)
    for idx, raw in enumerate(all_spectra):
        point_id = int(single_file.group(1)) if single_file else idx + 1
        xyz = None
        y_mm = None
        sidecar = Path(spec_path).parent / f"point_{point_id:04d}.json"
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if len(raw) == 2048 and np.array_equal(raw, np.asarray(meta["intensity"])):
                    xyz = tuple(meta[k] for k in ("controller_x_steps", "controller_y_steps", "controller_z_steps"))
                    if meta.get("scan_mode") == "1d-y":
                        y_mm = xyz[1] / 34700.0
            except (OSError, ValueError, KeyError, TypeError):
                pass
        scores = None
        if idx not in by_index:
            label = "REJECTED"
        else:
            scores = {id_to_class[c]: float(by_index[idx]["probs"][c]) for c in TARGET_CIDS}
            valid = all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values())
            detected = sorted(n for n, v in scores.items() if v >= OVR_FINAL_THRESHOLD) if valid else []
            if not detected:
                label = "UNKNOWN"
            elif len(detected) == 1:
                label = detected[0] + " only"
            else:
                label = " + ".join(detected) + " co-detection candidate"
        row = {"index": idx, "point_id": point_id, "xyz": xyz, "y_mm": y_mm,
               "label": label, "scores": scores, "reason": reasons.get(idx, "")}
        rows.append(row)
        groups.setdefault(label, []).append(row)

    r6g = groups.get("r6g only", [])
    mb = groups.get("methylene_blue only", [])
    verified_distinct = any(a["xyz"] is not None and b["xyz"] is not None and a["xyz"] != b["xyz"]
                            for a in r6g for b in mb)
    lines = ["", "=" * 70, "SCAN DISTRIBUTION SUMMARY"]
    if r6g and mb:
        lines.append("Spatially distinct R6G-only and MB-only target responses detected."
                     if verified_distinct else
                     "R6G-only and MB-only responses detected in different spectra; positions not verified.")
    elif any("co-detection candidate" in label for label in groups):
        lines.append("Multiple target responses detected at some points; separation is not established.")
    elif r6g or mb or any(label.endswith(" only") for label in groups):
        lines.append("Single-target responses detected; R6G/MB separation is not established.")
    else:
        lines.append("No confident target distribution established.")
    if any("co-detection candidate" in label for label in groups):
        lines.append("Some positions also show multiple target responses.")
    lines.append(f"Point threshold: {OVR_FINAL_THRESHOLD:.2f}; weak R6G rule NOT used for these notes.")
    lines.append("Position notes: spectrum index starts at 0; point ID starts at 1.")
    for label, points in groups.items():
        lines.append(f"  {label}: indices={[r['index'] for r in points]}; point IDs={[r['point_id'] for r in points]}")
    lines.append("Point details:")
    for row in rows:
        pos = f"Y={row['y_mm']:.3f} mm" if row["y_mm"] is not None else f"XYZ steps={row['xyz']}"
        lines.append(f"  Index {row['index']:02d} / point {row['point_id']:02d}: {pos}; {row['label']}; {row['reason']}")
    lines.extend([
        "Note: 'only' means only one target exceeded the threshold, not proof of chemical purity.",
        "Co-detection does not by itself confirm a physical mixture. Compare with the known sample layout.",
        "Y coordinates use verified sidecars and the existing 34700 steps/mm calibration.",
        "=" * 70,
    ])
    text = "\n".join(lines)
    print(text)
    parent = Path(OUTPUT_DIR) / "scan_analysis"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="vit1d_spatial_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_", dir=str(parent)))
    (folder / "spatial_summary.txt").write_text(text + "\n", encoding="utf-8")
    print("Saved position notes:", folder / "spatial_summary.txt")
    return rows



@torch.no_grad()
def predict_file(model,spec_path,dark_path):
    model.eval()
    dark_raw=load_dark_spectrum_from_file(dark_path) if dark_path else np.array([])
    spectra,failed=parse_portable_log_to_spectra(spec_path)
    print(f"\n  Parsed: {len(spectra)} spectra, {failed} failed")

    per_block=[]
    rejected_blocks=[]
    for idx,raw in enumerate(spectra):
        _,v_mm,quality=preprocess_two_versions(raw,dark_raw)
        if v_mm is None or np.allclose(v_mm,0):
            rejected_blocks.append({"idx":idx, "reason":"preprocessing failed"})
            continue
        if ENABLE_QUALITY_FILTER and (quality["ptp"]<RAW_MIN_PTP or quality["std"]<RAW_MIN_STD):
            rejected_blocks.append({"idx":idx, "reason":"low signal"})
            continue

        xt=torch.from_numpy(v_mm).unsqueeze(0).to(DEVICE)  # (1, TARGET_LEN)
        logits,_=model(xt)
        probs=torch.sigmoid(logits).squeeze(0).cpu().numpy()

        per_block.append({"idx":idx,"probs":probs,"quality":quality})
        ovr_str="  ".join(f"{id_to_class[k]}={probs[k]:.3f}" for k in range(NUM_CLASSES))
        print(f"  [SPEC {idx:3d}] [{ovr_str}]")

    print("\n"+"="*70); print("SCAN COMPONENT SUMMARY (ViT-1D average scores)"); print("="*70)
    n_acc=len(per_block)
    print(f"  Accepted: {n_acc}")

    if n_acc==0:
        print("  >>> RESULT: UNKNOWN")
        print_spatial_component_summary(spec_path, spectra, per_block, rejected_blocks)
        return

    avg_probs=np.mean(np.stack([b["probs"] for b in per_block]),axis=0)

    print(f"\n  === ViT OvR probabilities (avg over {n_acc} spectra) ===")
    for k in range(NUM_CLASSES):
        marker=" ***" if avg_probs[k]>=OVR_FINAL_THRESHOLD else ""
        print(f"    {id_to_class[k]:20s}  {avg_probs[k]:.4f}{marker}")

    plot_real_file_probabilities(per_block, avg_probs, prefix="ViT-1D", save_dir=OUTPUT_DIR)

    detected=set()
    for cid in TARGET_CIDS:
        if avg_probs[cid]>=OVR_FINAL_THRESHOLD: detected.add(cid)

    if class_to_id["methylene_blue"] in detected and avg_probs[class_to_id["r6g"]]>=R6G_WEAK_THRESHOLD_WHEN_MB_DOMINANT:
        detected.add(class_to_id["r6g"])

    print(f"\n  Detected: {[id_to_class[c] for c in sorted(detected)]}")
    if len(detected)==0:
        print("  >>> RESULT: UNKNOWN")
    elif len(detected)==1:
        cid=list(detected)[0]; print(f"  >>> RESULT: {id_to_class[cid]}  (OvR={avg_probs[cid]:.4f})")
    else:
        comps=sorted(detected,key=lambda c:avg_probs[c],reverse=True)
        names=[id_to_class[c] for c in comps]
        print(f"  >>> Components detected across scan: {', '.join(names)}")
        for c in comps:
            weak=" (weak)" if id_to_class[c]=="r6g" and avg_probs[c]<OVR_FINAL_THRESHOLD else ""
            print(f"      {id_to_class[c]:20s} OvR={avg_probs[c]:.4f}{weak}")
    print("="*70)
    print_spatial_component_summary(spec_path, spectra, per_block, rejected_blocks)


# =========================================================
# RUN
# =========================================================
RUNTIME_TEST_FILE,RUNTIME_TEST_DARK=_parse_cli_args()
print(f"\nTest: {RUNTIME_TEST_FILE}")

if RETRAIN_MODEL or not os.path.exists(MODEL_PATH):
    model=train_model()
else:
    model=RamanViT1D()
    model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
    model=model.to(DEVICE); model.eval()
    print(f"Loaded: {MODEL_PATH}")

print("\n"+"="*70); print("Inference"); print("="*70)
predict_file(model,RUNTIME_TEST_FILE,RUNTIME_TEST_DARK)
