import os
import argparse
import sys
import json
import time
import warnings
import gc
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception as _e:
    matplotlib = None
    plt = None
    _HAS_MPL = False
    print(f"[WARN] matplotlib unavailable ({_e}); plotting will be skipped.", flush=True)
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    sns = None
    _HAS_SNS = False
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize, MaxAbsScaler
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch import amp as torch_amp
    def autocast_ctx(enabled: bool):
        return torch_amp.autocast(device_type="cuda", enabled=enabled)
    def make_grad_scaler(enabled: bool):
        return torch_amp.GradScaler("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _cuda_autocast, GradScaler as _CudaGradScaler
    def autocast_ctx(enabled: bool):
        return _cuda_autocast(enabled=enabled)
    def make_grad_scaler(enabled: bool):
        return _CudaGradScaler(enabled=enabled)
TRAINING_FOLDER_PATH = r"J:\白酒数据集\train1"
TEST_FOLDER_PATH     = r"J:\白酒数据集\test1"
OUTPUT_FOLDER_PATH   = r"J:\白酒数据集\result_cnn概率校正"
RNG = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MZ_MIN = 50.0
MZ_MAX = 500.0
MZ_EPS = 1e-3
MZ_LABEL_DECIMALS = 1
USE_TIC_NORMALIZE = True
USE_LOG1P = True
USE_SCALER = True
SCALER_TYPE = "MaxAbs"
UNKNOWN_THRESHOLD = 0.60
GROUP_UNKNOWN_THRESHOLD = 0.0
EPOCHS = 60
EARLY_STOP_PATIENCE = 10
MIN_EPOCHS = 10
BATCH_SIZE = 32
GRAD_CLIP_NORM = 5.0
ENABLE_OPTUNA = True
OPTUNA_N_TRIALS = 10
OPTUNA_TIMEOUT = None
OPTUNA_CV_SPLITS = 5
OPTUNA_EPOCHS = 20
OPTUNA_NUM_WORKERS = 0
OPTUNA_VERBOSE = True
CHANNEL_OPTIONS = [
    (16, 32, 64),
    (32, 64, 128),
    (32, 64, 128, 256)
]
NUM_WORKERS = 0
PIN_MEMORY = False
ENABLE_SHAP = True
SHAP_BACKGROUND_SIZE = 50
SHAP_PLOT_SIZE = 150
SHAP_MAX_DISPLAY = 20
ENABLE_LEARNING_CURVE = True
LEARNING_CURVE_SIZES = [0.1, 0.3, 0.5, 0.7, 1.0]
LEARNING_CURVE_SCORING = "accuracy"
LEARNING_CURVE_CV_SPLITS = 5
LEARNING_CURVE_EPOCHS = 30
FIG_DPI = 300
SAVE_PDF = True
SAVE_PNG = True
ENABLE_ROC = True
MM_TO_INCH = 1.0 / 25.4
ELSEVIER_SINGLE_COL_MM = 90
ELSEVIER_DOUBLE_COL_MM = 190
def mm_to_in(mm: float) -> float:
    return float(mm) * MM_TO_INCH
def journal_figsize(width_mm: float, aspect: float = 0.75):
    w = mm_to_in(width_mm)
    return (w, w * float(aspect))
if _HAS_MPL and (plt is not None):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "savefig.dpi": FIG_DPI,
    })
    if _HAS_SNS:
        sns.set_theme(style="white", context="paper")
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
def _resolve_output_dir(path_str: str) -> Path:
    p = Path(path_str or ".").expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception as e:
        fallback = Path.cwd() / "output"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] Cannot create OUTPUT_FOLDER_PATH={p} ({e}); using {fallback}", flush=True)
        return fallback
output_dir = _resolve_output_dir(OUTPUT_FOLDER_PATH)
fig_dir = output_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
if os.environ.get("DISABLE_FILE_LOG", "0") != "1":
    sys.stdout = DualLogger(output_dir / "training_log.txt")
model_save_path = output_dir / "cnn1d_model.pkl"
labels_save_path = output_dir / "label_mapping.csv"
test_pred_save_path = output_dir / "test_predictions.csv"
group_test_pred_path = output_dir / "group_test_predictions.csv"
optuna_best_path = output_dir / "optuna_best.json"
print(f"[INFO] Output Directory: {output_dir.resolve()}")
print(f"[INFO] Device: {DEVICE}")
def free_mem():
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def mz_label(x: float, decimals: int = 1) -> str:
    return f"{float(x):.{int(decimals)}f}"
def mz_to_grid_idx(mz: np.ndarray, mz0: float, mz_step: float) -> np.ndarray:
    return np.rint((mz - float(mz0)) / float(mz_step)).astype(int)
def load_and_transpose_csv_grid(file_path: Path, label: int, mz0: float, mz_step: float):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[WARN] 读取失败 {file_path}: {e}")
        return None
    if "m/z" not in df.columns:
        print(f"[WARN] {file_path.name} 缺少 m/z 列")
        return None
    df["m/z"] = pd.to_numeric(df["m/z"], errors="coerce")
    df = df.dropna(subset=["m/z"])
    df = df[(df["m/z"] >= MZ_MIN - MZ_EPS) & (df["m/z"] <= MZ_MAX + MZ_EPS)]
    if df.empty:
        return None
    int_cols = [c for c in df.columns if c != "m/z"]
    mz = df["m/z"].values.astype(float)
    idx = mz_to_grid_idx(mz, mz0=mz0, mz_step=mz_step)
    df2 = df[int_cols].copy()
    df2["_idx"] = idx
    df2 = df2.groupby("_idx", as_index=True).sum(numeric_only=True)
    df_T = df2.T.copy()
    df_T["sample_id"] = df_T.index.astype(str)
    df_T["group_id"] = df_T["sample_id"].str.split("|", n=1).str[0].str.strip()
    df_T["label"] = int(label)
    df_T.reset_index(drop=True, inplace=True)
    return df_T
def get_aligned_data(df: pd.DataFrame, feature_idx_all):
    meta = df[["label", "sample_id", "group_id"]].copy()
    data_df = df.drop(columns=["label", "sample_id", "group_id"], errors="ignore").copy()
    cols_map = {}
    for c in data_df.columns:
        try:
            cols_map[c] = int(c)
        except Exception:
            pass
    data_df = data_df.rename(columns=cols_map)
    target_cols = list(map(int, feature_idx_all))
    df_aligned = data_df.reindex(columns=target_cols, fill_value=0.0)
    return df_aligned.values.astype(np.float32), meta
def _save_fig(fig, outpath_no_ext: Path):
    if (not _HAS_MPL) or (plt is None):
        print(f"[WARN] Skip saving figure (matplotlib unavailable): {outpath_no_ext}")
        return
    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(str(outpath_no_ext) + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(str(outpath_no_ext) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
def preprocess_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if USE_TIC_NORMALIZE:
        sums = X.sum(axis=1, keepdims=True)
        X = X / (sums + 1e-10)
    if USE_LOG1P:
        X = np.log1p(X)
    return X
class SpectraDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]
class CNN1D(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 1,
                 channels=(32, 64, 128),
                 kernel_size: int = 7,
                 dropout: float = 0.3,
                 fc_dim: int = 128,
                 input_len: int = 1000):
        super().__init__()
        ks = int(kernel_size)
        pad = ks // 2
        layers = []
        ch_in = in_ch
        curr_len = input_len
        for ch_out in channels:
            layers.append(nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding=pad, bias=False))
            layers.append(nn.BatchNorm1d(ch_out))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(p=float(dropout)))
            ch_in = ch_out
            curr_len = curr_len // 2
        self.feature = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(ch_in, int(fc_dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.out = nn.Linear(int(fc_dim), n_classes)
    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x).squeeze(-1)
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            with autocast_ctx(False):
                x = F.relu(self.fc1(x.float()))
                x = self.drop(x)
                x = self.out(x)
            return x
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.out(x)
def train_epoch(model, loader, optimizer, criterion, amp_scaler=None, amp_enabled=False):
    model.train()
    sum_loss, correct, total = 0.0, 0, 0
    if amp_scaler is None:
        amp_scaler = make_grad_scaler(enabled=amp_enabled)
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(amp_enabled):
            logits = model(Xb)
            loss = criterion(logits, yb)
        if amp_enabled:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        sum_loss += float(loss.item()) * Xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return sum_loss / max(total, 1), correct / max(total, 1)
@torch.no_grad()
def eval_epoch(model, loader, criterion, amp_enabled=False, temperature: float = 1.0):
    model.eval()
    sum_loss, correct, total = 0.0, 0, 0
    probs_list, preds_list, true_list = [], [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        with autocast_ctx(amp_enabled):
            logits = model(Xb)
            loss = criterion(logits, yb)
        sum_loss += float(loss.item()) * Xb.size(0)
        T = float(temperature) if (temperature is not None and float(temperature) > 0) else 1.0
        probs = F.softmax(logits.float() / T, dim=1)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        probs_list.append(probs.detach().cpu().numpy())
        preds_list.append(preds.detach().cpu().numpy())
        true_list.append(yb.detach().cpu().numpy())
    return (sum_loss / max(total, 1),
            correct / max(total, 1),
            np.concatenate(probs_list) if probs_list else np.zeros((0, 0)),
            np.concatenate(preds_list) if preds_list else np.zeros((0,), dtype=int),
            np.concatenate(true_list) if true_list else np.zeros((0,), dtype=int))
@torch.no_grad()
def eval_logits_epoch(model, loader, amp_enabled=False):
    model.eval()
    logits_list, true_list = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        with autocast_ctx(amp_enabled):
            logits = model(Xb)
        logits_list.append(logits.float().detach().cpu().numpy())
        true_list.append(yb.detach().cpu().numpy())
    if logits_list:
        return np.concatenate(logits_list, axis=0), np.concatenate(true_list, axis=0)
    return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=int)
def fit_temperature_from_logits(logits_np: np.ndarray, y_np: np.ndarray, max_iter: int = 100):
    logits = torch.tensor(np.asarray(logits_np, dtype=np.float32))
    y = torch.tensor(np.asarray(y_np, dtype=np.int64))
    if logits.ndim != 2 or logits.shape[0] != y.shape[0]:
        raise ValueError(f"Bad logits/y shapes: logits={tuple(logits.shape)} y={tuple(y.shape)}")
    if logits.shape[0] == 0:
        raise ValueError("Empty logits for temperature fitting")
    logT = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([logT], lr=0.2, max_iter=int(max_iter), line_search_fn="strong_wolfe")
    def _nll(temp: torch.Tensor):
        return F.cross_entropy(logits / temp, y)
    with torch.no_grad():
        nll_raw = float(F.cross_entropy(logits, y).item())
    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT) + 1e-6
        loss = _nll(T)
        loss.backward()
        return loss
    try:
        optimizer.step(closure)
        success = True
    except Exception:
        success = False
        opt2 = torch.optim.Adam([logT], lr=0.05)
        for _ in range(int(max_iter)):
            opt2.zero_grad()
            T = torch.exp(logT) + 1e-6
            loss = _nll(T)
            loss.backward()
            opt2.step()
    with torch.no_grad():
        T = float((torch.exp(logT) + 1e-6).item())
        nll_cal = float(_nll(torch.tensor(T)).item())
    return T, nll_raw, nll_cal, success
def collect_oof_logits_for_temp(best_params: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                n_classes: int, cv_splits: int = 5, calib_epochs: int = 20):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int).ravel()
    groups = np.asarray(groups)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=int(cv_splits), shuffle=True, random_state=RNG)
        splits = list(cv.split(X, y, groups))
        cv_name = "StratifiedGroupKFold"
    except Exception:
        cv = GroupKFold(n_splits=int(cv_splits))
        splits = list(cv.split(X, y, groups))
        cv_name = "GroupKFold"
    print(f"[CAL] Collecting OOF logits via {cv_name} (splits={len(splits)}) | epochs={calib_epochs}", flush=True)
    oof_logits = np.zeros((len(y), n_classes), dtype=np.float32)
    oof_mask = np.zeros((len(y),), dtype=bool)
    amp_enabled = (DEVICE == "cuda")
    for fold_i, (tr_idx, va_idx) in enumerate(splits, start=1):
        Xt, Xv = X[tr_idx], X[va_idx]
        yt, yv = y[tr_idx], y[va_idx]
        acc, model_f, scaler_f = run_training(
            best_params, Xt, yt, Xv, yv, n_classes,
            max_epochs=int(calib_epochs),
            verbose=False,
            num_workers=0
        )
        Xv2 = scaler_f.transform(Xv) if scaler_f is not None else Xv
        dl = DataLoader(SpectraDataset(Xv2, yv), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
        logits_v, y_v = eval_logits_epoch(model_f, dl, amp_enabled=amp_enabled)
        if logits_v.shape[0] != len(va_idx):
            raise RuntimeError(f"OOF logits size mismatch: got {logits_v.shape[0]} expected {len(va_idx)}")
        oof_logits[va_idx] = logits_v
        oof_mask[va_idx] = True
        print(f"[CAL] fold {fold_i}/{len(splits)} done | val={len(va_idx)} | acc={acc:.4f}", flush=True)
        free_mem()
    if not bool(oof_mask.all()):
        miss = int((~oof_mask).sum())
        print(f"[WARN] OOF collection incomplete: missing={miss}", flush=True)
    return oof_logits[oof_mask], y[oof_mask]
def run_training(params, X_train, y_train, X_val, y_val, n_classes, max_epochs=EPOCHS, verbose=False, num_workers=None):
    scaler = None
    if USE_SCALER:
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = scaler.transform(X_val)
    train_ds = SpectraDataset(X_train, y_train)
    nw = NUM_WORKERS if num_workers is None else int(num_workers)
    drop_last = len(train_ds) >= int(BATCH_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=drop_last,
        num_workers=nw, pin_memory=PIN_MEMORY
    )
    val_loader = None
    if X_val is not None:
        val_ds = SpectraDataset(X_val, y_val)
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=nw, pin_memory=PIN_MEMORY
        )
    model = CNN1D(
        n_classes=n_classes,
        input_len=X_train.shape[1],
        channels=params["channels"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        fc_dim=params["fc_dim"],
    ).to(DEVICE)
    weight_decay = float(params.get("weight_decay", 1e-4))
    label_smoothing = float(params.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=weight_decay)
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing > 0:
            print("[WARN] 当前 PyTorch 版本不支持 CrossEntropyLoss(label_smoothing=...), 已自动忽略 label_smoothing。", flush=True)
        criterion = nn.CrossEntropyLoss()
    amp_enabled = (DEVICE == "cuda")
    amp_scaler = make_grad_scaler(enabled=amp_enabled)
    best_acc = 0.0
    patience = 0
    best_state = None
    for ep in range(int(max_epochs)):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, amp_scaler, amp_enabled)
        if val_loader is not None:
            v_loss, v_acc, _, _, _ = eval_epoch(model, val_loader, criterion, amp_enabled)
            score = float(v_acc)
        else:
            v_acc = 0.0
            score = float(t_acc)
        if score > best_acc:
            best_acc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if verbose and (ep % 5 == 0):
            print(f"  Ep {ep:03d}: T={t_acc:.4f} V={v_acc:.4f} Best={best_acc:.4f}")
        if ep >= MIN_EPOCHS and patience >= EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return float(best_acc), model, scaler
def objective(trial, X, y, groups, n_classes):
    t0 = time.time()
    ch_idx = trial.suggest_categorical("channel_idx", list(range(len(CHANNEL_OPTIONS))))
    selected_channels = CHANNEL_OPTIONS[int(ch_idx)]
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "fc_dim": trial.suggest_categorical("fc_dim", [64, 128, 256]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "channels": selected_channels,
    }
    optuna_epochs = int(OPTUNA_EPOCHS) if OPTUNA_EPOCHS else int(EPOCHS)
    if OPTUNA_VERBOSE:
        print(
            f"[Optuna] Trial {trial.number} start | epochs={optuna_epochs} | "
            f"lr={params['lr']:.2e} wd={params['weight_decay']:.1e} ls={params['label_smoothing']:.3f} "
            f"dropout={params['dropout']:.2f} fc_dim={params['fc_dim']} k={params['kernel_size']} ch={params['channels']}",
            flush=True
        )
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=OPTUNA_CV_SPLITS, shuffle=True, random_state=RNG)
        splits = list(cv.split(X, y, groups))
    except Exception:
        cv = GroupKFold(n_splits=OPTUNA_CV_SPLITS)
        splits = list(cv.split(X, y, groups))
    scores = []
    n_splits = len(splits)
    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        if OPTUNA_VERBOSE:
            print(
                f"[Optuna] Trial {trial.number} fold {fold_i}/{n_splits} | "
                f"train={len(train_idx)} val={len(val_idx)}",
                flush=True
            )
        Xt, Xv = X[train_idx], X[val_idx]
        yt, yv = y[train_idx], y[val_idx]
        try:
            acc, _, _ = run_training(
                params, Xt, yt, Xv, yv, n_classes,
                max_epochs=optuna_epochs,
                verbose=False,
                num_workers=OPTUNA_NUM_WORKERS
            )
        except Exception as e:
            print(f"[Optuna][ERROR] Trial {trial.number} fold {fold_i} failed: {e}", flush=True)
            traceback.print_exc()
            free_mem()
            raise
        scores.append(acc)
        trial.report(acc, step=fold_i)
        if trial.should_prune():
            if OPTUNA_VERBOSE:
                print(f"[Optuna] Trial {trial.number} pruned at fold {fold_i}", flush=True)
            raise optuna.exceptions.TrialPruned()
    value = float(np.mean(scores))
    if OPTUNA_VERBOSE:
        dt = time.time() - t0
        print(f"[Optuna] Trial {trial.number} done | value={value:.4f} | time={dt:.1f}s", flush=True)
    return value
def plot_confusion_matrix(cm, labels, title, outpath_no_ext, normalize=None, double_col=True):
    if (not _HAS_MPL) or (plt is None):
        return
    cm = np.asarray(cm)
    if normalize == "true":
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm.astype(int)
        fmt = "d"
        vmin, vmax = 0, int(cm_plot.max()) if cm_plot.size else 1
    width_mm = ELSEVIER_DOUBLE_COL_MM if double_col else ELSEVIER_SINGLE_COL_MM
    fig, ax = plt.subplots(figsize=journal_figsize(width_mm=width_mm, aspect=1))
    if _HAS_SNS:
        sns.heatmap(
            cm_plot, cmap="Blues", vmin=vmin, vmax=vmax,
            square=True, linewidths=0.25, linecolor="white",
            annot=True, fmt=fmt,
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"shrink": 0.85, "pad": 0.02},
            ax=ax
        )
    else:
        im = ax.imshow(cm_plot, cmap="Blues", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title, pad=10, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
    ax.set_ylabel("True", fontsize=10, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    _save_fig(fig, outpath_no_ext)
def plot_multiclass_roc_ovr(y_true, proba, class_names, outpath_no_ext, double_col=True, title_suffix="CNN probabilities"):
    if (not _HAS_MPL) or (plt is None):
        return
    y_true = np.asarray(y_true, dtype=int).ravel()
    proba = np.asarray(proba, dtype=float)
    n_classes = len(class_names)
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    if Y.shape[1] != n_classes:
        Y2 = np.zeros((len(y_true), n_classes), dtype=int)
        for i in range(n_classes):
            Y2[:, i] = (y_true == i).astype(int)
        Y = Y2
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        if Y[:, i].sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    width_mm = ELSEVIER_DOUBLE_COL_MM if double_col else ELSEVIER_SINGLE_COL_MM
    fig, ax = plt.subplots(figsize=journal_figsize(width_mm=width_mm, aspect=1))
    for i in range(n_classes):
        if i not in fpr:
            continue
        ax.plot(fpr[i], tpr[i], lw=1.0, alpha=0.85, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], lw=1.2, alpha=0.8, ls="--", color="gray", label="Random (AUC=0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    ax.set_title(f"ROC curves (OvR) - {title_suffix}", pad=10, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, ncol=1, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    _save_fig(fig, outpath_no_ext)
def plot_shap_mean_abs_stacked_bar(shap_values_list, feature_names, class_names, outpath_no_ext, double_col=True):
    if (not _HAS_MPL) or (plt is None):
        return
    sv_list = [np.asarray(v) for v in shap_values_list]
    C = len(sv_list)
    if C == 0:
        return
    M = np.stack([np.mean(np.abs(v), axis=0) for v in sv_list], axis=0)
    total = M.sum(axis=0)
    k = min(int(SHAP_MAX_DISPLAY), len(feature_names), M.shape[1])
    order = np.argsort(total)[::-1][:k]
    M = M[:, order]
    fn = [feature_names[i] for i in order]
    width_mm = ELSEVIER_DOUBLE_COL_MM if double_col else ELSEVIER_SINGLE_COL_MM
    fig, ax = plt.subplots(figsize=journal_figsize(width_mm=width_mm, aspect=0.75))
    y = np.arange(k)
    left = np.zeros(k, dtype=float)
    for c in range(C):
        ax.barh(y, M[c], left=left, height=0.8, label=str(class_names[c]))
        left += M[c]
    ax.set_yticks(y)
    ax.set_yticklabels(fn)
    ax.invert_yaxis()
    ax.set_title("Mean absolute SHAP values", pad=8, fontsize=11, fontweight="bold")
    ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, ncol=1, fontsize=9)
    _save_fig(fig, outpath_no_ext)
def plot_shap_summary_dot(shap_signed, X_plot, feature_names, outpath_no_ext):
    if (not _HAS_MPL) or (plt is None):
        return
    try:
        import shap
    except Exception:
        print("[WARN] shap 未安装，跳过 SHAP。")
        return
    shap.summary_plot(
        shap_signed,
        X_plot,
        feature_names=feature_names,
        plot_type="dot",
        max_display=min(int(SHAP_MAX_DISPLAY), len(feature_names)),
        show=False
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("SHAP summary (dot) - 1D-CNN peak area", pad=4, fontweight="bold")
    _save_fig(fig, outpath_no_ext)
def normalize_shap_values(shap_values_raw, n_classes: int):
    if isinstance(shap_values_raw, list):
        if len(shap_values_raw) == n_classes:
            out = []
            for v in shap_values_raw:
                v = np.asarray(v)
                if v.ndim == 3 and v.shape[1] == 1:
                    v = v[:, 0, :]
                elif v.ndim == 2:
                    pass
                else:
                    v = np.squeeze(v)
                    if v.ndim == 3 and v.shape[1] == 1:
                        v = v[:, 0, :]
                out.append(np.ascontiguousarray(v, dtype=np.float32))
            return out
        if len(shap_values_raw) == 1:
            shap_values_raw = shap_values_raw[0]
        else:
            tmp = [np.squeeze(np.asarray(v)) for v in shap_values_raw]
            return [np.ascontiguousarray(v, dtype=np.float32) for v in tmp]
    arr = np.asarray(shap_values_raw)
    if arr.ndim == 4 and arr.shape[-1] == n_classes:
        out = []
        for c in range(n_classes):
            v = arr[..., c]
            if v.ndim == 3 and v.shape[1] == 1:
                v = v[:, 0, :]
            out.append(np.ascontiguousarray(v, dtype=np.float32))
        return out
    if arr.ndim == 3 and arr.shape[-1] == n_classes:
        return [np.ascontiguousarray(arr[..., c], dtype=np.float32) for c in range(n_classes)]
    if arr.ndim == 4 and arr.shape[0] == n_classes:
        out = []
        for c in range(n_classes):
            v = arr[c]
            if v.ndim == 3 and v.shape[1] == 1:
                v = v[:, 0, :]
            out.append(np.ascontiguousarray(v, dtype=np.float32))
        return out
    if arr.ndim == 3 and arr.shape[0] == n_classes:
        return [np.ascontiguousarray(arr[c], dtype=np.float32) for c in range(n_classes)]
    raise ValueError(f"Unrecognized SHAP output shape: {arr.shape}")
def plot_learning_curve_groupcv(X, y, groups, best_params, n_classes, outpath):
    if (not _HAS_MPL) or (plt is None):
        return
    print("[INFO] Generating Learning Curve (this takes time)...")
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=LEARNING_CURVE_CV_SPLITS, shuffle=True, random_state=RNG)
        split_iter = list(cv.split(X, y, groups))
    except Exception:
        cv = GroupKFold(n_splits=LEARNING_CURVE_CV_SPLITS)
        split_iter = list(cv.split(X, y, groups))
    sizes = LEARNING_CURVE_SIZES
    n_sizes = len(sizes)
    n_splits = len(split_iter)
    train_scores = np.zeros((n_sizes, n_splits), dtype=float)
    val_scores = np.zeros((n_sizes, n_splits), dtype=float)
    counts = np.zeros((n_sizes, n_splits), dtype=float)
    for fold_i, (t_idx_all, v_idx) in enumerate(split_iter):
        Xt_all, yt_all = X[t_idx_all], y[t_idx_all]
        gt_all = groups[t_idx_all]
        Xv, yv = X[v_idx], y[v_idx]
        uniq_g = np.unique(gt_all)
        for si, frac in enumerate(sizes):
            n_pick = max(1, int(len(uniq_g) * frac))
            np.random.seed(RNG + si + 1000 * fold_i)
            picked_g = np.random.choice(uniq_g, size=n_pick, replace=False)
            mask = np.isin(gt_all, picked_g)
            Xt_sub, yt_sub = Xt_all[mask], yt_all[mask]
            counts[si, fold_i] = float(len(yt_sub))
            acc_v, model, scaler = run_training(
                best_params, Xt_sub, yt_sub, Xv, yv, n_classes,
                max_epochs=LEARNING_CURVE_EPOCHS, verbose=False
            )
            val_scores[si, fold_i] = float(acc_v)
            Xtr_eval = scaler.transform(Xt_sub) if scaler is not None else Xt_sub
            dl_tr = DataLoader(SpectraDataset(Xtr_eval, yt_sub), batch_size=BATCH_SIZE, shuffle=False)
            _, tr_acc, _, _, _ = eval_epoch(model, dl_tr, nn.CrossEntropyLoss())
            train_scores[si, fold_i] = float(tr_acc)
    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std = val_scores.std(axis=1)
    x_mean = counts.mean(axis=1)
    fig, ax = plt.subplots(figsize=journal_figsize(width_mm=ELSEVIER_DOUBLE_COL_MM, aspect=0.75))
    ax.plot(x_mean, tr_mean, marker="o", lw=1.2, label="Train")
    ax.plot(x_mean, va_mean, marker="o", lw=1.2, label="CV")
    ax.fill_between(x_mean, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15)
    ax.fill_between(x_mean, va_mean - va_std, va_mean + va_std, alpha=0.15)
    ax.set_xlabel("Training examples", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Score ({LEARNING_CURVE_SCORING})", fontsize=10, fontweight="bold")
    ax.set_title("Learning curve (GroupCV)", pad=10, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    _save_fig(fig, outpath)
def main():
    set_seed(RNG)
    print("\n[1] Loading Training Data...")
    train_root = Path(TRAINING_FOLDER_PATH)
    csv_files = sorted(train_root.glob("*.csv"))
    if not csv_files:
        sys.exit("No CSV files found in train folder.")
    class_names = sorted(list({p.stem.split("_")[0] for p in csv_files}))
    label_map = {name: i for i, name in enumerate(class_names)}
    id_map = {i: name for name, i in label_map.items()}
    n_classes = len(class_names)
    print(f"Classes: {class_names}")
    mz0 = 50.0
    mz_step = 0.25
    print(f"[FIX] Enforcing Grid: Start={mz0:.2f}, Step={mz_step:.2f}")
    idx_min = int(np.rint((MZ_MIN - mz0) / mz_step))
    idx_max = int(np.rint((MZ_MAX - mz0) / mz_step))
    feature_idx_all = list(range(idx_min, idx_max + 1))
    print(f"[INFO] Grid Length: {len(feature_idx_all)}")
    dfs = []
    for p in csv_files:
        prefix = p.stem.split("_")[0]
        part = load_and_transpose_csv_grid(p, label_map[prefix], mz0, mz_step)
        if part is not None:
            dfs.append(part)
    if not dfs:
        sys.exit("No valid training data parsed.")
    full_df = pd.concat(dfs, ignore_index=True)
    X_raw, meta = get_aligned_data(full_df, feature_idx_all)
    y_raw = meta["label"].values.astype(int)
    groups = meta["group_id"].values.astype(str)
    X_processed = preprocess_X(X_raw)
    print(f"Data Shape after binning: {X_processed.shape}")
    best_params = {
        "lr": 1e-3, "dropout": 0.3, "fc_dim": 128,
        "kernel_size": 7, "channels": (32, 64, 128),
    }
    if ENABLE_OPTUNA:
        print()
        print("[2] Starting Optuna...")
        optuna.logging.set_verbosity(optuna.logging.INFO)
        sampler = optuna.samplers.TPESampler(seed=RNG)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        def _optuna_cb(study, trial):
            try:
                v = trial.value
            except Exception:
                v = None
            print(f"[Optuna] Finished trial={trial.number} value={v} state={trial.state.name}", flush=True)
        study.optimize(
            lambda t: objective(t, X_processed, y_raw, groups, n_classes),
            n_trials=OPTUNA_N_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            callbacks=[_optuna_cb],
            gc_after_trial=True
        )
        best_params = dict(study.best_params)
        if "channel_idx" in best_params:
            idx = int(best_params.pop("channel_idx"))
            best_params["channels"] = CHANNEL_OPTIONS[idx]
        elif "channels" not in best_params:
            best_params["channels"] = (32, 64, 128)
        print("Best Params:", json.dumps(best_params, indent=2, ensure_ascii=False))
        with open(optuna_best_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
    print("\n[3] Training Final Model...")
    if ENABLE_LEARNING_CURVE:
        plot_learning_curve_groupcv(
            X_processed, y_raw, groups, best_params, n_classes,
            fig_dir / "fig_06_learning_curve_groupcv"
        )
    final_acc, final_model, final_scaler = run_training(
        best_params, X_processed, y_raw, None, None, n_classes,
        max_epochs=EPOCHS, verbose=True
    )
    proba_calibrator = None
    try:
        print("\n[CAL] Fitting temperature scaling on OOF logits (GroupCV)...", flush=True)
        calib_epochs = int(min(20, max(5, int(OPTUNA_EPOCHS) if OPTUNA_EPOCHS else 20)))
        oof_logits, oof_y = collect_oof_logits_for_temp(
            best_params, X_processed, y_raw, groups, n_classes,
            cv_splits=int(OPTUNA_CV_SPLITS),
            calib_epochs=calib_epochs
        )
        T, nll_raw, nll_cal, ok = fit_temperature_from_logits(oof_logits, oof_y, max_iter=120)
        proba_calibrator = {
            "method": "temperature",
            "T": float(T),
            "source": "oof_groupcv",
            "cv_splits": int(OPTUNA_CV_SPLITS),
            "calib_epochs": int(calib_epochs),
            "nll_raw": float(nll_raw),
            "nll_cal": float(nll_cal),
            "success": bool(ok),
        }
        print(f"[INFO] 概率校准完成：T={T:.4f}, NLL_raw={nll_raw:.6f}, NLL_cal={nll_cal:.6f}, success={ok}", flush=True)
    except Exception as e:
        proba_calibrator = None
        print(f"[WARN] Probability calibration skipped (can ignore): {e}", flush=True)
    feature_names_all = [mz_label(mz0 + i * mz_step, decimals=MZ_LABEL_DECIMALS) for i in range(len(feature_idx_all))]
    joblib.dump({
        "model_state": final_model.state_dict(),
        "scaler": final_scaler,
        "best_params": best_params,
        "label_map": label_map,
        "mz_info": {"mz0": mz0, "mz_step": mz_step, "len": len(feature_idx_all)},
        "feature_names": feature_names_all,
        "proba_calibrator": proba_calibrator,
        "unknown_threshold": float(UNKNOWN_THRESHOLD),
        "group_unknown_threshold": float(GROUP_UNKNOWN_THRESHOLD),
    }, model_save_path)
    print("[INFO] Model saved:", str(model_save_path))
    print("\n[4] Testing...")
    test_root = Path(TEST_FOLDER_PATH)
    test_dfs = []
    for p in sorted(test_root.glob("*.csv")):
        prefix = p.stem.split("_")[0]
        if prefix in label_map:
            part = load_and_transpose_csv_grid(p, label_map[prefix], mz0, mz_step)
            if part is not None:
                test_dfs.append(part)
    if not test_dfs:
        print("[WARN] No test data found.")
        return
    test_full = pd.concat(test_dfs, ignore_index=True)
    X_test_raw, meta_test = get_aligned_data(test_full, feature_idx_all)
    y_test = meta_test["label"].values.astype(int)
    g_test = meta_test["group_id"].values.astype(str)
    s_test = meta_test["sample_id"].values.astype(str)
    X_test = preprocess_X(X_test_raw)
    if final_scaler is not None:
        X_test = final_scaler.transform(X_test)
    test_ds = SpectraDataset(X_test, y_test)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    T_use = float(proba_calibrator.get('T', 1.0)) if isinstance(proba_calibrator, dict) else 1.0
    _, acc, probs, preds, trues = eval_epoch(final_model, test_loader, nn.CrossEntropyLoss(), amp_enabled=(DEVICE=='cuda'), temperature=T_use)
    print(f"[RESULT] Test Acc: {acc:.4f}")
    print(classification_report(trues, preds, target_names=class_names))
    cm = confusion_matrix(trues, preds)
    plot_confusion_matrix(cm, class_names, "Confusion Matrix", fig_dir / "fig_01_confusion_matrix", normalize=None)
    plot_confusion_matrix(cm, class_names, "Confusion Matrix (Norm)", fig_dir / "fig_02_confusion_matrix_norm",
                          normalize="true")
    if ENABLE_ROC and probs.size:
        plot_multiclass_roc_ovr(trues, probs, class_names, fig_dir / "fig_03_roc_ovr_proba")
    max_p = probs.max(axis=1) if probs.size else np.zeros((len(trues),), dtype=float)
    out_df = pd.DataFrame({
        "Sample": s_test,
        "Group": g_test,
        "True_Label": [id_map[int(t)] for t in trues],
        "Pred_Label": [id_map[int(p)] for p in preds],
        "Similarity": max_p,
        "Is_Unknown": (max_p < UNKNOWN_THRESHOLD),
    })
    out_df.to_csv(test_pred_save_path, index=False, encoding="utf-8-sig")
    grp_res = []
    for g in np.unique(g_test):
        idx = np.where(g_test == g)[0]
        mean_p = probs[idx].mean(axis=0) if probs.size else np.zeros((n_classes,), dtype=float)
        pred_idx = int(mean_p.argmax()) if mean_p.size else 0
        conf = float(mean_p.max()) if mean_p.size else 0.0
        pred_name = "UNKNOWN" if (GROUP_UNKNOWN_THRESHOLD > 0 and conf < GROUP_UNKNOWN_THRESHOLD) else id_map[pred_idx]
        grp_res.append({
            "Group": g,
            "True_Label": id_map[int(trues[idx[0]])],
            "Pred_Label": pred_name,
            "Confidence": conf,
        })
    pd.DataFrame(grp_res).to_csv(group_test_pred_path, index=False, encoding="utf-8-sig")
    if ENABLE_SHAP:
        print("\n[5] SHAP Analysis...")
        try:
            import shap
            free_mem()
            final_model.eval()
            X_train_for_model = final_scaler.transform(X_processed) if final_scaler is not None else X_processed
            bg_size = min(int(SHAP_BACKGROUND_SIZE), len(X_train_for_model))
            plot_size = min(int(SHAP_PLOT_SIZE), len(X_test))
            if bg_size <= 0 or plot_size <= 0:
                print("[WARN] SHAP skipped due to empty data.")
                return
            np.random.seed(RNG)
            bg_idx = np.random.choice(len(X_train_for_model), bg_size, replace=False)
            plot_idx = np.random.choice(len(X_test), plot_size, replace=False)
            X_bg_tensor = torch.from_numpy(np.ascontiguousarray(X_train_for_model[bg_idx], dtype=np.float32)) \
                .to(DEVICE).contiguous().unsqueeze(1)
            X_eval_tensor = torch.from_numpy(np.ascontiguousarray(X_test[plot_idx], dtype=np.float32)) \
                .to(DEVICE).contiguous().unsqueeze(1)
            explainer = shap.GradientExplainer(final_model, X_bg_tensor)
            shap_values_raw = explainer.shap_values(X_eval_tensor)
            sv_list = normalize_shap_values(shap_values_raw, n_classes)
            print("[INFO] SHAP per-class shapes:", [v.shape for v in sv_list])
            feature_names = feature_names_all
            plot_shap_mean_abs_stacked_bar(
                sv_list, feature_names, class_names, fig_dir / "fig_04_shap_stacked_bar"
            )
            y_pred_eval = np.asarray(preds[plot_idx]).astype(int).ravel()
            n_eval, n_feats = sv_list[0].shape
            shap_signed_selected = np.zeros((n_eval, n_feats), dtype=np.float32)
            for i in range(n_eval):
                c = int(y_pred_eval[i])
                shap_signed_selected[i, :] = sv_list[c][i, :]
            plot_shap_summary_dot(
                shap_signed_selected,
                X_test[plot_idx],
                feature_names,
                fig_dir / "fig_05_shap_summary_dot"
            )
            print("[INFO] SHAP plots saved.")
        except Exception as e:
            print(f"[ERROR] SHAP Failed: {e}")
            traceback.print_exc()
        finally:
            free_mem()
    print("\n=== Pipeline Complete ===")
def _smoke_test():
    print()
    print("[SMOKE] Running synthetic-data smoke test...", flush=True)
    set_seed(RNG)
    global ENABLE_SHAP, ENABLE_LEARNING_CURVE, ENABLE_ROC
    global ENABLE_OPTUNA, OPTUNA_N_TRIALS, OPTUNA_CV_SPLITS, OPTUNA_EPOCHS
    global BATCH_SIZE, NUM_WORKERS, OPTUNA_NUM_WORKERS
    ENABLE_SHAP = False
    ENABLE_LEARNING_CURVE = False
    ENABLE_ROC = False
    ENABLE_OPTUNA = True
    OPTUNA_N_TRIALS = 1
    OPTUNA_CV_SPLITS = 2
    OPTUNA_EPOCHS = 2
    BATCH_SIZE = 16
    NUM_WORKERS = 0
    OPTUNA_NUM_WORKERS = 0
    n_samples = 30
    n_features = 100
    n_classes = 3
    rng = np.random.default_rng(RNG)
    y = rng.integers(0, n_classes, size=n_samples, endpoint=False).astype(int)
    groups = np.array([f"g{gi:02d}" for gi in range(n_samples // 6) for _ in range(6)], dtype=object)
    if len(groups) < n_samples:
        groups = np.concatenate(
            [groups, np.array([f"g{len(groups)//6:02d}"] * (n_samples - len(groups)), dtype=object)]
        )
    X = rng.normal(0, 1, size=(n_samples, n_features)).astype(np.float32)
    for c in range(n_classes):
        X[y == c, c * 20:(c + 1) * 20] += 1.5
    Xp = preprocess_X(X)
    print(f"[SMOKE] Data shape: {Xp.shape} | classes={n_classes} | groups={len(np.unique(groups))}", flush=True)
    print()
    print("[SMOKE] Starting Optuna...", flush=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=RNG)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda t: objective(t, Xp, y, groups, n_classes), n_trials=OPTUNA_N_TRIALS, gc_after_trial=True)
    bp = dict(study.best_params)
    if "channel_idx" in bp:
        idx = int(bp.pop("channel_idx"))
        bp["channels"] = CHANNEL_OPTIONS[idx]
    print("[SMOKE] Best params:", bp, flush=True)
    print()
    print("[SMOKE] Final train (no val)...", flush=True)
    acc, _, _ = run_training(bp, Xp, y, None, None, n_classes, max_epochs=2, verbose=True, num_workers=0)
    print(f"[SMOKE] Final train acc proxy: {acc:.4f}", flush=True)
    print("[SMOKE] OK", flush=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick synthetic-data sanity check and exit")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        main()