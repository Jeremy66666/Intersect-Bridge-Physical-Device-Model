import os
import re
import io
import json
import time
import sys
import random
import math
from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Cell 1/10: config
# =========================================================
DATA_ROOT = r"C:\Users\Jeremy666\OneDrive\Desktop\Portable_Raman_Paper_Machine_Learning\Portable Raman数据"

LIB_FILES = {
    "pyo":     r"C:\Users\Jeremy666\OneDrive\Desktop\Portable_Raman_Paper_Machine_Learning\Portable Raman数据\pyo_1s_50mw_1mM_10x10_portable.txt",
    "adamite": r"C:\Users\Jeremy666\OneDrive\Desktop\Portable_Raman_Paper_Machine_Learning\Portable Raman数据\Admine_1s_50mw_10mM_10x10cycle_portable.txt",
    "si":      r"C:\Users\Jeremy666\OneDrive\Desktop\Portable_Raman_Paper_Machine_Learning\Portable Raman数据\Si_1s_50mw_10x10_2D_Portable.txt",
    "r6g":     r"C:\Users\Jeremy666\OneDrive\Desktop\Portable_Raman_Paper_Machine_Learning\Portable Raman数据\r6g_1s_50mw_1mM_10x10_2D_portable.txt",
}

TEST_FILE = r"C:\Users\Jeremy666\OneDrive\Desktop\text.txt"

SUPPORTED_EXTS = {".txt", ".csv", ".tsv", ".dat", ".asc", ".prn"}

# ----------------- 关键参数 -----------------
TARGET_LEN = 2048   # 2045 -> 插值到 2048（几乎无损，且是 2 的幂，对 patch 分割友好）
PATCH_SIZE = 64     # 每个 patch 覆盖 64 个光谱点；2048/64 = 32 个 token
                    # 可选组合：PATCH_SIZE=32 -> 64 token；PATCH_SIZE=128 -> 16 token
# --------------------------------------------

BATCH_SIZE = 32
EPOCHS     = 80     # Transformer 通常需要比 CNN 多训练几个 epoch
LR         = 5e-4   # Transformer 对学习率更敏感，稍低一些
WEIGHT_DECAY = 1e-4
SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

DO_BASELINE = False
NUM_WORKERS = 0

# file-level matching threshold
MATCH_THRESHOLD = 0.90
MATCH_MARGIN    = 0.08

# softmax temperature
INFER_TEMPERATURE = 1.0

# options: "mean_logits", "vote", "max"
INFER_AGGREGATION = "mean_logits"

# training improvements
DROPOUT         = 0.15          # Transformer 内部 attention/ffn dropout
DROP_PATH       = 0.10          # Stochastic Depth
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM  = 3.0

USE_AUGMENT    = True
AUG_NOISE_STD  = 0.015
AUG_SHIFT_MAX  = 12
AUG_SCALE_LOW  = 0.90
AUG_SCALE_HIGH = 1.10

OUTPUT_DIR = os.path.join(DATA_ROOT, "_ViT1D_4class_r6g_si_adamite_pyo")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
META_PATH  = os.path.join(OUTPUT_DIR, "meta.json")
MODEL_PATH = os.path.join(MODELS_DIR, "vit1d_4class.pt")

os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE.startswith("cuda")

print("DEVICE     =", DEVICE)
print("MODEL_PATH =", MODEL_PATH)
print(f"TARGET_LEN={TARGET_LEN}  PATCH_SIZE={PATCH_SIZE}  num_tokens={TARGET_LEN // PATCH_SIZE}")


# =========================================================
# Cell 2/10: utils  (与原版完全相同)
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def stratified_split_indices(
    Y: np.ndarray,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=SEED
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)

    idx_tr, idx_va, idx_te = [], [], []
    for cls in np.unique(Y):
        idx = np.where(Y == cls)[0].tolist()
        rng.shuffle(idx)
        n = len(idx)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        tr = idx[:n_train]
        va = idx[n_train:n_train + n_val]
        te = idx[n_train + n_val:]

        idx_tr.extend(tr)
        idx_va.extend(va)
        idx_te.extend(te)

    rng.shuffle(idx_tr)
    rng.shuffle(idx_va)
    rng.shuffle(idx_te)
    return idx_tr, idx_va, idx_te


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.mean(x))
    s = float(np.std(x))
    if (not np.isfinite(m)) or (not np.isfinite(s)) or (s < eps):
        return np.zeros_like(x, dtype=np.float32)
    out = (x - m) / (s + eps)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def simple_baseline_remove(y: np.ndarray, poly_order: int = 3) -> np.ndarray:
    n = y.size
    x = np.linspace(0, 1, n)
    coeff = np.polyfit(x, y, deg=poly_order)
    base  = np.polyval(coeff, x)
    return y - base


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.size == 0 or (not np.isfinite(z).all()):
        return np.zeros_like(z, dtype=np.float32)
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0 or (not np.isfinite(s)):
        return np.zeros_like(z, dtype=np.float32)
    out = e / s
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def pearson_corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-8 or db < 1e-8:
        return 0.0
    corr = float(np.dot(a, b) / (da * db + 1e-8))
    corr = max(-1.0, min(1.0, corr))
    return (corr + 1.0) * 0.5


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-8 or db < 1e-8:
        return 0.0
    cs = float(np.dot(a, b) / (da * db + 1e-8))
    cs = max(-1.0, min(1.0, cs))
    return (cs + 1.0) * 0.5


def combined_match_score(a: np.ndarray, b: np.ndarray) -> float:
    p = pearson_corr_score(a, b)
    c = cosine_score(a, b)
    score = 0.7 * p + 0.3 * c
    return float(max(0.0, min(1.0, score)))


# =========================================================
# Cell 3/10: parsing  (与原版完全相同)
# =========================================================
_BAD_FILES   = set()
ERROR_PHRASE = "image data error code 0x0001"


def _read_text_any_encoding(path: str) -> Tuple[str, str]:
    b = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "utf-16le", "utf-16be", "gbk", "cp1252", "latin1"):
        try:
            return b.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="ignore"), "utf-8-ignore"


def strip_prefix(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\[[^\]]+\]\s+\[Backend\]\s*", "", line)
    line = re.sub(r"^\[[^\]]+\]\s*", "", line)
    return line.strip()


def parse_portable_log_to_spectra(log_file: str) -> Tuple[List[np.ndarray], int]:
    lines = open(log_file, "r", encoding="utf-8", errors="ignore").read().splitlines()

    spectra       = []
    failed_blocks = 0

    i = 0
    while i < len(lines):
        line_raw = lines[i]
        low  = line_raw.lower()
        line = strip_prefix(line_raw)

        if ERROR_PHRASE in low:
            failed_blocks += 1
            i += 1
            continue

        if line.startswith("Image Data"):
            tokens     = []
            first_part = line[len("Image Data"):].strip()
            tokens.extend(re.findall(r"\b[0-9A-Fa-f]{2,4}\b", first_part))

            j = i + 1
            block_has_error = False

            while j < len(lines):
                nxt_raw = lines[j]
                nxt_low = nxt_raw.lower()
                nxt     = strip_prefix(nxt_raw)

                if ERROR_PHRASE in nxt_low:
                    block_has_error = True

                if nxt.startswith("Stop Capture"):
                    break
                if nxt.startswith("Image Data") and j != i:
                    break

                hexes = re.findall(r"\b[0-9A-Fa-f]{2,4}\b", nxt)
                if hexes:
                    tokens.extend(hexes)

                j += 1

            if block_has_error or len(tokens) == 0:
                failed_blocks += 1
            else:
                vals = np.array([int(t, 16) for t in tokens], dtype=np.int64)

                if vals.size % 2 == 0 and vals.max(initial=0) <= 255:
                    le     = vals[0::2] + (vals[1::2] << 8)
                    be     = (vals[0::2] << 8) + vals[1::2]
                    vals16 = le if le.max(initial=0) >= be.max(initial=0) else be
                    if vals16.max(initial=0) > 255:
                        vals = vals16.astype(np.int64)

                spectra.append(vals.astype(np.float32))

            i = j + 1
            continue

        i += 1

    return spectra, failed_blocks


def _parse_hex_tokens_to_vals(tokens: List[str]) -> np.ndarray:
    vals = np.array([int(t, 16) for t in tokens], dtype=np.int64)

    if vals.size % 2 == 0 and vals.max(initial=0) <= 255:
        le     = vals[0::2] + (vals[1::2] << 8)
        be     = (vals[0::2] << 8) + vals[1::2]
        vals16 = le if le.max(initial=0) >= be.max(initial=0) else be
        if vals16.max(initial=0) > 255:
            vals = vals16.astype(np.int64)

    return vals.astype(np.float32)


# =========================================================
# Cell 4/10: preprocess  (与原版完全相同)
# =========================================================
def preprocess_intensity_to_vec(y: np.ndarray, target_len: int, do_baseline: bool) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if y.size < 4:
        out = np.zeros(target_len, dtype=np.float32)
        out[:min(target_len, y.size)] = y[:min(target_len, y.size)]
        return out

    if do_baseline:
        y = simple_baseline_remove(y.astype(np.float64), poly_order=3).astype(np.float32)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    x     = np.arange(y.size, dtype=np.float64)
    x_new = np.linspace(float(x.min()), float(x.max()), target_len, dtype=np.float64)
    y_new = np.interp(x_new, x, y.astype(np.float64)).astype(np.float32)
    y_new = np.nan_to_num(y_new, nan=0.0, posinf=0.0, neginf=0.0)
    y_new = zscore(y_new).astype(np.float32)
    return y_new


def load_spectrum_text(path: str, target_len: int, do_baseline: bool) -> np.ndarray:
    try:
        text, _ = _read_text_any_encoding(path)
        low     = text.lower()

        if ("image data" in low) or ("stop capture" in low):
            spectra, failed = parse_portable_log_to_spectra(path)
            if len(spectra) == 0:
                raise ValueError(f"portable log parsed 0 spectra (failed_blocks={failed})")
            v = preprocess_intensity_to_vec(spectra[0], target_len, do_baseline)
            return v.astype(np.float32)

        tokens = re.findall(r"\b(?:0x)?[0-9A-Fa-f]{2,4}\b", text)
        tokens = [t[2:] if t.lower().startswith("0x") else t for t in tokens]
        if len(tokens) >= 32:
            vals = _parse_hex_tokens_to_vals(tokens)
            v    = preprocess_intensity_to_vec(vals, target_len, do_baseline)
            return v.astype(np.float32)

        head      = "\n".join(text.splitlines()[:50])
        delimiter = "," if "," in head else ("\t" if "\t" in head else None)

        data = np.genfromtxt(io.StringIO(text), delimiter=delimiter, dtype=np.float64, invalid_raise=False)
        if data is None or (isinstance(data, float) and np.isnan(data)):
            raise ValueError("genfromtxt returned NaN/None")

        if np.isscalar(data):
            y = np.array([float(data)], dtype=np.float64)
        elif data.ndim == 1:
            y = data.astype(np.float64)
        else:
            y = (data[:, 1] if data.shape[1] >= 2 else data[:, 0]).astype(np.float64)

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        v = preprocess_intensity_to_vec(y, target_len, do_baseline)
        return v.astype(np.float32)

    except Exception as e:
        if path not in _BAD_FILES:
            _BAD_FILES.add(path)
            print(f"[WARN] Failed to parse spectrum: {path}\n       Reason: {repr(e)}")
        return np.zeros((target_len,), dtype=np.float32)


# =========================================================
# Cell 5/10: build dataset  (与原版完全相同)
# =========================================================
class_names = list(LIB_FILES.keys())
class_to_id = {c: i for i, c in enumerate(class_names)}
id_to_class = {i: c for c, i in class_to_id.items()}

X        = []
Y        = []
RAW_LENS = []

for cls, fp in LIB_FILES.items():
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    spectra, failed = parse_portable_log_to_spectra(fp)
    print(f"[LIB] {cls:8s}  spectra={len(spectra)}  failed_blocks={failed}  file={os.path.basename(fp)}")

    for s in spectra:
        v = preprocess_intensity_to_vec(s, TARGET_LEN, DO_BASELINE)
        if np.allclose(v, 0):
            continue
        RAW_LENS.append(int(s.size))
        X.append(v)
        Y.append(class_to_id[cls])

X = np.stack(X, axis=0).astype(np.float32)
Y = np.array(Y, dtype=np.int64)

print("Total samples:", X.shape, "labels:", Y.shape)
print("Class counts:", {id_to_class[i]: int((Y == i).sum()) for i in range(len(class_names))})

RAW_LEN_MIN, RAW_LEN_MAX = 0, 10**9
if len(RAW_LENS) > 0:
    arr         = np.array(RAW_LENS, dtype=np.int64)
    q10, q90    = np.percentile(arr, [10, 90]).astype(np.float64)
    RAW_LEN_MIN = int(max(16, np.floor(q10 * 0.80)))
    RAW_LEN_MAX = int(np.ceil(q90 * 1.20))
print(f"RAW_LEN_RANGE (from LIB): [{RAW_LEN_MIN}, {RAW_LEN_MAX}]  (q10/q90 based)")

idx_tr, idx_va, idx_te = stratified_split_indices(Y, seed=SEED)
print("Split sizes:", len(idx_tr), len(idx_va), len(idx_te))

CLASS_PROTOTYPES = {}
for cls_name, cls_id in class_to_id.items():
    cls_x = X[Y == cls_id]
    if cls_x.shape[0] == 0:
        CLASS_PROTOTYPES[cls_name] = np.zeros((TARGET_LEN,), dtype=np.float32)
    else:
        proto = np.mean(cls_x, axis=0).astype(np.float32)
        CLASS_PROTOTYPES[cls_name] = zscore(proto).astype(np.float32)


def compute_match_rates_from_vec(v: np.ndarray) -> np.ndarray:
    rates = []
    for cls_name in class_names:
        proto = CLASS_PROTOTYPES[cls_name]
        rate  = combined_match_score(v, proto)
        rates.append(rate)
    rates = np.asarray(rates, dtype=np.float32)
    rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
    return rates


# =========================================================
# Cell 6/10: dataset loader + augmentation  (与原版完全相同)
# =========================================================
def augment_spectrum_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()

    scale = np.random.uniform(AUG_SCALE_LOW, AUG_SCALE_HIGH)
    x = x * np.float32(scale)

    shift = int(np.random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1))
    if shift != 0:
        x = np.roll(x, shift)

    noise = np.random.normal(0.0, AUG_NOISE_STD, size=x.shape).astype(np.float32)
    x = x + noise

    x = zscore(x).astype(np.float32)
    return x


class RamanMultiClassDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, indices: List[int], augment: bool = False):
        self.X       = X
        self.Y       = Y
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k: int):
        i = self.indices[k]
        x = self.X[i].copy()

        if self.augment:
            x = augment_spectrum_1d(x)

        # ViT 版本：输出 [TARGET_LEN] 的 1D 向量，不加 channel 维度
        # PatchEmbedding 层会在模型内部做切分
        x = torch.from_numpy(x)                            # shape: (TARGET_LEN,)
        y = torch.tensor(int(self.Y[i]), dtype=torch.long)
        return x, y


train_ds = RamanMultiClassDataset(X, Y, idx_tr, augment=USE_AUGMENT)
val_ds   = RamanMultiClassDataset(X, Y, idx_va, augment=False)
test_ds  = RamanMultiClassDataset(X, Y, idx_te, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


# =========================================================
# Cell 7/10: Patch-based 1D Vision Transformer  <- 核心改动
#
# 整体架构（以 TARGET_LEN=2048, PATCH_SIZE=64 为例）：
#
#   输入光谱  [B, 2048]
#       |
#       v  PatchEmbedding1D
#   32 个 patch，每 patch=64 点 -> 线性映射到 d_model=128
#   [B, 32, 128]
#       |
#       v  拼接可学习 [CLS] token
#   [B, 33, 128]
#       |
#       v  加可学习位置编码 + Dropout
#   [B, 33, 128]
#       |
#       v  TransformerBlock x 4 (Pre-LN + MHSA + DropPath + FFN)
#   [B, 33, 128]
#       |
#       v  取 CLS token (index=0) + LayerNorm
#   [B, 128]
#       |
#       v  分类头 (Linear -> GELU -> Dropout -> Linear)
#   [B, 4]  logits
#
# 参数量约 200K，适合 100-500 样本的小数据集
# =========================================================

class DropPath(nn.Module):
    """
    Stochastic Depth 随机深度正则化。
    训练时以概率 drop_prob 把整条残差路径清零（相当于跳过这一层）。
    推理时自动关闭，行为等价于恒等映射。
    和 Dropout 的区别：Dropout 随机置零单个神经元，DropPath 随机置零整个 batch 里某些样本的残差。
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob     = 1.0 - self.drop_prob
        shape         = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, 1, ...]
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)   # 0 or 1
        return x * random_tensor / keep_prob


class PatchEmbedding1D(nn.Module):
    """
    把 1D 光谱切成不重叠的小块（patch），再线性投影到 d_model 维度。

    例：seq_len=2048, patch_size=64
        -> 2048 / 64 = 32 个 patch
        每个 patch 是原始 64 个光谱强度值
        线性层把 64 维映射到 d_model=128 维

    为什么要切 patch？
        Transformer 的自注意力复杂度是 O(N^2)，如果把每个光谱点当一个 token（N=2048），
        计算量太大。把 64 个点打包成一个 token，N 降到 32，计算快 64 倍，
        同时每个 token 还保留了局部光谱形状信息。
    """
    def __init__(self, seq_len: int, patch_size: int, d_model: int):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"seq_len ({seq_len}) 必须能被 patch_size ({patch_size}) 整除"
        self.patch_size  = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Linear(patch_size, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len]
        B, L = x.shape
        # 切 patch: [B, num_patches, patch_size]
        x = x.view(B, self.num_patches, self.patch_size)
        # 线性投影: [B, num_patches, d_model]
        x = self.proj(x)
        x = self.norm(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力（MHSA）。

    核心思想：
        对序列里每一个 token，计算它和其他所有 token 的"相关性"分数（注意力权重），
        然后按权重聚合所有 token 的值（Value）。多头 = 同时跑多个独立的注意力，
        每个头关注不同的特征方向，最后拼接。

    对 Raman 光谱的意义：
        一个 patch（如 700-800 cm-1 的峰区）可以关注另一个 patch（如 1400-1500 cm-1），
        捕捉跨峰的相关性，这是 CNN 做不到的（CNN 只有局部感受野）。
    """
    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5   # 缩放因子，防止点积过大

        # Q/K/V 合并成一个线性层，推理时再 split
        self.qkv       = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj      = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # qkv: [B, N, 3C] -> [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # 各自 [B, heads, N, head_dim]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                               # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)    # [B, N, C]
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    """
    单个 Transformer 编码器层，采用 Pre-LayerNorm 结构（比原始的 Post-LN 训练更稳定）。

    结构：
        x -> LayerNorm -> MHSA -> DropPath -> (+残差) -> x
        x -> LayerNorm -> FFN  -> DropPath -> (+残差) -> x

    FFN（前馈网络）= Linear(d) -> GELU -> Dropout -> Linear(d)
    FFN 的隐层宽度 = d_model * mlp_ratio（默认 4 倍），用于学习非线性特征变换。
    """
    def __init__(self,
                 d_model:   int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout:   float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, num_heads, attn_drop)
        self.dp1   = DropPath(drop_path)

        self.norm2  = nn.LayerNorm(d_model)
        hidden_dim  = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))   # 注意力残差
        x = x + self.dp2(self.ffn(self.norm2(x)))    # FFN 残差
        return x


class RamanViT1D(nn.Module):
    """
    Patch-based 1D Vision Transformer，专为便携式 Raman/SERS 光谱分类设计。

    关键超参数：
        seq_len    : TARGET_LEN（光谱点数，必须是 patch_size 的整数倍）
        patch_size : PATCH_SIZE（每个 token 覆盖多少个光谱点）
        d_model    : token 的嵌入维度（Transformer 内部宽度）
        num_heads  : 自注意力头数（d_model 须整除 num_heads）
        depth      : Transformer 编码器层数
        mlp_ratio  : FFN 隐层倍率

    默认配置（d_model=128, depth=4）约 200K 参数：
        适合 100-500 样本/类。
    数据量大时（>1000 样本/类）可调大：
        d_model=192, depth=6, num_heads=6
    """
    def __init__(self,
                 seq_len:     int   = TARGET_LEN,
                 patch_size:  int   = PATCH_SIZE,
                 d_model:     int   = 128,
                 num_heads:   int   = 4,
                 depth:       int   = 4,
                 mlp_ratio:   float = 4.0,
                 num_classes: int   = 4,
                 dropout:     float = 0.15,
                 attn_drop:   float = 0.10,
                 drop_path:   float = DROP_PATH):
        super().__init__()

        # ---- Patch Embedding ----
        self.patch_embed = PatchEmbedding1D(seq_len, patch_size, d_model)
        num_patches      = seq_len // patch_size

        # ---- 可学习的 [CLS] token ----
        # 和 ViT 原文一样，用于汇总全局信息，最终接分类头
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ---- 可学习位置编码 ----
        # 让模型知道每个 patch 在光谱里的相对位置（哪个 Raman 位移区段）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop  = nn.Dropout(dropout)

        # ---- Transformer 编码器堆叠 ----
        # 越深的层 DropPath 概率越高（线性递增），越靠近输出越需要正则化
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model   = d_model,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                dropout   = dropout,
                attn_drop = attn_drop,
                drop_path = dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)

        # ---- 分类头 ----
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # 截断正态初始化位置编码和 CLS token（ViT 标准做法）
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, TARGET_LEN]
        B = x.shape[0]

        # Step 1: 切 patch + 线性投影 -> [B, num_patches, d_model]
        x = self.patch_embed(x)

        # Step 2: 把 CLS token 拼到最前面 -> [B, num_patches+1, d_model]
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, d_model]
        x   = torch.cat([cls, x], dim=1)

        # Step 3: 加位置编码（让模型知道每个 patch 的位置）
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Step 4: 过 depth 个 Transformer 层
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Step 5: 取 CLS token（index 0）做分类
        cls_out = x[:, 0]         # [B, d_model]
        logits  = self.head(cls_out)  # [B, num_classes]
        return logits


# =========================================================
# Cell 8/10: train  (仅模型初始化和学习率调度与原版不同)
# =========================================================
@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x      = x.to(DEVICE, non_blocking=True)
        y      = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        pred   = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total   += int(y.numel())
    return correct / max(total, 1)


def train_model():
    model = RamanViT1D(
        seq_len     = TARGET_LEN,
        patch_size  = PATCH_SIZE,
        d_model     = 128,      # token 宽度；样本多时可改 192/256
        num_heads   = 4,        # 注意力头数；d_model 必须整除 num_heads
        depth       = 4,        # Transformer 层数；样本多时可改 6
        mlp_ratio   = 4.0,
        num_classes = len(class_names),
        dropout     = DROPOUT,
        attn_drop   = 0.10,
        drop_path   = DROP_PATH,
    ).to(DEVICE)

    # 打印参数量，方便对比 CNN 版本
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ViT1D] trainable params: {n_params:,}")
    print(f"[ViT1D] num_patches={TARGET_LEN // PATCH_SIZE}  d_model=128  depth=4  heads=4")

    class_counts  = np.array([(Y == i).sum() for i in range(len(class_names))], dtype=np.float32)
    class_weights = class_counts.mean() / np.maximum(class_counts, 1.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # Transformer 标配：AdamW（比 SGD 收敛快，且对位置编码友好）
    opt = torch.optim.AdamW(
        model.parameters(),
        lr           = LR,
        weight_decay = WEIGHT_DECAY,
        betas        = (0.9, 0.999),
    )

    # Warmup（5 epoch 线性升温）+ Cosine 退火
    # Transformer 冷启动时梯度不稳定，warmup 是标准做法
    warmup_epochs = 5
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = float(ep - warmup_epochs) / float(max(1, EPOCHS - warmup_epochs))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val   = -1.0
    best_state = None
    patience   = 16   # Transformer 比 CNN 需要更多耐心
    bad        = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = ce(logits, y)

            if not torch.isfinite(loss):
                print(f"[WARN] skip non-finite loss at epoch {ep}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            losses.append(float(loss.item()))

        scheduler.step()

        tr_loss    = float(np.mean(losses)) if losses else 0.0
        va         = eval_acc(model, val_loader)
        current_lr = opt.param_groups[0]["lr"]
        print(f"epoch {ep:02d}/{EPOCHS}  train_loss={tr_loss:.4f}  val_acc={va:.3f}  lr={current_lr:.2e}")

        if va > best_val + 1e-6:
            best_val   = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad        = 0
        else:
            bad += 1
            if bad >= patience:
                print("early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te = eval_acc(model, test_loader)
    print(f"TEST acc = {te:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)

    meta = {
        "classes"          : class_names,
        "target_len"       : TARGET_LEN,
        "patch_size"       : PATCH_SIZE,
        "do_baseline"      : DO_BASELINE,
        "match_threshold"  : MATCH_THRESHOLD,
        "infer_temperature": INFER_TEMPERATURE,
        "infer_aggregation": INFER_AGGREGATION,
        "raw_len_range"    : [RAW_LEN_MIN, RAW_LEN_MAX],
        "timestamp"        : time.strftime("%Y-%m-%d %H:%M:%S"),
        "lib_files"        : LIB_FILES,
        "model_name"       : "RamanViT1D",
        "d_model"          : 128,
        "num_heads"        : 4,
        "depth"            : 4,
        "dropout"          : DROPOUT,
        "drop_path"        : DROP_PATH,
        "label_smoothing"  : LABEL_SMOOTHING,
        "use_augment"      : USE_AUGMENT,
        "weight_decay"     : WEIGHT_DECAY,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved model:", MODEL_PATH)
    print("Saved meta :", META_PATH)
    return model


# =========================================================
# Cell 9/10: load model + predict helper  (与原版基本相同)
# =========================================================
def load_trained_model():
    m = RamanViT1D(
        seq_len     = TARGET_LEN,
        patch_size  = PATCH_SIZE,
        d_model     = 128,
        num_heads   = 4,
        depth       = 4,
        mlp_ratio   = 4.0,
        num_classes = len(class_names),
        dropout     = DROPOUT,
        attn_drop   = 0.10,
        drop_path   = DROP_PATH,
    )
    sd = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def _forward_logits(model: nn.Module, v: np.ndarray) -> np.ndarray:
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ViT 版：输入 [1, TARGET_LEN]，不需要额外 channel 维度
    x      = torch.from_numpy(v).unsqueeze(0).to(DEVICE)   # [1, TARGET_LEN]
    logits = model(x)

    logits = logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return logits


def _filter_by_raw_len(spectra: List[np.ndarray], raw_min: int, raw_max: int) -> Tuple[List[np.ndarray], int]:
    if len(spectra) == 0:
        return spectra, 0
    kept    = []
    dropped = 0
    for s in spectra:
        n = int(getattr(s, "size", 0))
        if raw_min <= n <= raw_max:
            kept.append(s)
        else:
            dropped += 1
    if len(kept) == 0:
        return spectra, 0
    return kept, dropped


# =========================================================
# Cell 10/10: file-level predict  (与原版完全相同)
# =========================================================
@torch.no_grad()
def predict_single_file(model: nn.Module, spectrum_path: str,
                        scan_all_spectra: bool = True, debug: bool = True):
    if not os.path.exists(spectrum_path):
        raise FileNotFoundError(spectrum_path)

    ext = os.path.splitext(spectrum_path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported ext: {ext}")

    model = model.to(DEVICE)
    model.eval()

    spectra, failed      = parse_portable_log_to_spectra(spectrum_path)
    already_preprocessed = False

    if len(spectra) == 0:
        v_single = load_spectrum_text(spectrum_path, TARGET_LEN, DO_BASELINE).astype(np.float32)
        spectra  = [v_single]
        already_preprocessed = True
        dropped_len = 0
    else:
        spectra, dropped_len = _filter_by_raw_len(spectra, RAW_LEN_MIN, RAW_LEN_MAX)

    n_scan      = len(spectra) if scan_all_spectra else 1
    per_block   = []
    vote_counts = {c: 0 for c in class_names}

    for idx in range(n_scan):
        raw = spectra[idx]

        if already_preprocessed:
            v       = raw.astype(np.float32)
            raw_len = int(v.size)
        else:
            raw_len = int(raw.size)
            v = preprocess_intensity_to_vec(raw, TARGET_LEN, DO_BASELINE).astype(np.float32)

        if np.allclose(v, 0):
            continue

        logits = _forward_logits(model, v)
        prob   = softmax_np(logits / float(INFER_TEMPERATURE))

        match_rates = compute_match_rates_from_vec(v)
        top_id   = int(np.argmax(match_rates))
        top_name = id_to_class[top_id]
        top_rate = float(match_rates[top_id])

        vote_counts[top_name] += 1
        per_block.append({
            "idx"        : idx,
            "raw_len"    : raw_len,
            "logits"     : logits,
            "prob"       : prob,
            "match_rates": match_rates,
            "top_id"     : top_id,
            "top_name"   : top_name,
            "top_rate"   : top_rate,
        })

        if debug and match_rates.size >= 2:
            sorted_idx = np.argsort(match_rates)[::-1]
            top1 = id_to_class[int(sorted_idx[0])]
            top2 = id_to_class[int(sorted_idx[1])]
            r1   = float(match_rates[sorted_idx[0]])
            r2   = float(match_rates[sorted_idx[1]])
            print(f"[DEBUG] idx={idx:3d} raw_len={raw_len:4d} "
                  f"top1={top1}({r1:.4f}) top2={top2}({r2:.4f}) margin={(r1-r2):.4f}")

    if len(per_block) == 0:
        return {
            "matched"      : False,
            "label"        : "NOT MATCHED",
            "top_prob"     : 0.0,
            "top2_prob"    : 0.0,
            "margin"       : 0.0,
            "best_idx"     : 0,
            "num_spectra"  : int(len(spectra)),
            "num_used"     : 0,
            "dropped_len"  : int(dropped_len),
            "failed_blocks": int(failed),
            "probs"        : {c: 0.0 for c in class_names},
            "votes"        : vote_counts,
        }

    agg_rates = np.mean(
        np.stack([d["match_rates"] for d in per_block], axis=0), axis=0
    ).astype(np.float32)

    sorted_idx = np.argsort(agg_rates)[::-1]
    top1_id    = int(sorted_idx[0])
    top2_id    = int(sorted_idx[1])
    top1_name  = id_to_class[top1_id]
    top2_name  = id_to_class[top2_id]
    top1_prob  = float(agg_rates[top1_id])
    top2_prob  = float(agg_rates[top2_id])
    margin     = float(top1_prob - top2_prob)

    best_idx = max(per_block, key=lambda d: float(d["match_rates"][top1_id]))["idx"]

    if debug:
        print("\n[DEBUG] ===== FILE-LEVEL MATCH RATE AGGREGATION =====")
        print("[DEBUG] aggregation: mean_match_rates")
        print("[DEBUG] blocks used:", len(per_block), "/", n_scan, f"(dropped_len={dropped_len})")
        print("[DEBUG] votes:", vote_counts)
        print(f"[DEBUG] top1={top1_name}({top1_prob:.4f}) top2={top2_name}({top2_prob:.4f}) margin={margin:.4f}")
        sup = [d for d in per_block if d["idx"] == best_idx][0]
        print(f"[DEBUG] best_idx_for_top1={best_idx}  raw_len={sup['raw_len']}  "
              f"block_top1={sup['top_name']}({sup['top_rate']:.4f})")

    matched = (top1_prob >= MATCH_THRESHOLD) and (margin >= MATCH_MARGIN)

    prob_dict   = {id_to_class[i]: float(agg_rates[i]) for i in range(len(agg_rates))}
    prob_sorted = dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "matched"      : matched,
        "label"        : top1_name if matched else "NOT MATCHED",
        "top_prob"     : top1_prob,
        "top2_prob"    : top2_prob,
        "margin"       : margin,
        "best_idx"     : int(best_idx),
        "num_spectra"  : int(len(spectra)),
        "num_used"     : int(len(per_block)),
        "dropped_len"  : int(dropped_len),
        "failed_blocks": int(failed),
        "probs"        : prob_sorted,
        "votes"        : vote_counts,
    }


# =========================================================
# run
# =========================================================
RUNTIME_TEST_FILE = sys.argv[1] if len(sys.argv) > 1 and str(sys.argv[1]).strip() else TEST_FILE

model  = train_model()
model2 = load_trained_model()
print("Loaded model from:", MODEL_PATH)

res = predict_single_file(model2, RUNTIME_TEST_FILE, scan_all_spectra=True, debug=True)

print("\n" + "=" * 90)
print("TEST FILE:", RUNTIME_TEST_FILE)
print(f"Parsed spectra: {res['num_spectra']}  used={res['num_used']}  best_idx={res['best_idx']}  "
      f"failed_blocks={res.get('failed_blocks', 0)}  dropped_len={res.get('dropped_len', 0)}")
print(f"top1_prob={res['top_prob']:.4f}  top2_prob={res['top2_prob']:.4f}  margin={res['margin']:.4f}")

print("\nProbabilities (Raman ViT-1D, file-level):")
for k, v in res["probs"].items():
    print(f"  - {k:10s}  {v:.4f}")

print("\nFINAL:")
if not res["matched"]:
    print(f"  -> NOT MATCHED (top1={res['top_prob']:.4f}, margin={res['margin']:.4f})")
else:
    print(f"  -> MATCHED: {res['label']}  (score={res['top_prob']:.4f})")