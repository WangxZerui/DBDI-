import sys
import json
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    sns = None
    _HAS_SNS = False
TRAINING_FOLDER_PATH = r"J:\白酒数据集\train1"
TEST_FOLDER_PATH     = r"J:\白酒数据集\test1"
OUTPUT_FOLDER_PATH   = r"J:\白酒数据集\result_RF概率校正"
RNG = 42
MZ_MIN = 50.0
MZ_MAX = 500.0
MZ_EPS = 1e-3
MZ_LABEL_DECIMALS = 1
UNKNOWN_THRESHOLD = 0.60
GROUP_UNKNOWN_THRESHOLD = 0.0
GAUSSIAN_SIGMA_DA = 0.25
PEAK_MIN_WIDTH_DA = 0.20
PEAK_RELATIVE_HEIGHT = 0.01
PEAK_MIN_INTENSITY = 0.0
PEAK_WINDOWS_MODE = "per_class_union"
PEAK_MIN_SAMPLES_PER_CLASS = 20
PEAK_MAX_PEAKS_PER_CLASS = 0
UNION_MAX_WINDOWS = 0
MERGE_OVERLAPPED_WINDOWS = True
RE_NORMALIZE_PEAK_AREAS = True
USE_FEATURE_SELECTION = True
TOP_K_FEATURES = 600
ENABLE_OPTUNA = True
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = None
OPTUNA_SCORING = "accuracy"
OPTUNA_CV_SPLITS = 5
FIG_DPI = 300
SAVE_PDF = True
SAVE_PNG = True
ENABLE_ROC = True
ENABLE_SHAP = True
SHAP_MAX_SAMPLES = 1000
SHAP_MAX_DISPLAY = 20
READ_CHUNK_BASE = 200000
BATCH_SIZE_MEAN = 64
BATCH_SIZE_FEAT = 64
NON_FEATURE_COLS = ["label", "sample_id", "group_id"]
ENABLE_LEARNING_CURVE = True
LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.3, 0.5, 0.7, 1.0]
LEARNING_CURVE_SCORING = OPTUNA_SCORING
LEARNING_CURVE_CV_SPLITS = OPTUNA_CV_SPLITS
MM_TO_INCH = 1.0 / 25.4
ELSEVIER_SINGLE_COL_MM = 90
ELSEVIER_DOUBLE_COL_MM = 190
def mm_to_in(mm: float) -> float:
    return mm * MM_TO_INCH
def journal_figsize(width_mm=ELSEVIER_SINGLE_COL_MM, aspect=1.0):
    w = mm_to_in(width_mm)
    h = w * aspect
    return (w, h)
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
output_dir = Path(OUTPUT_FOLDER_PATH or ".")
output_dir.mkdir(parents=True, exist_ok=True)
fig_dir = output_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
model_save_path = output_dir / "rf_peakarea.pkl"
labels_save_path = output_dir / "label_mapping.csv"
features_save_path = output_dir / "peak_feature_names.json"
test_pred_save_path = output_dir / "test_predictions.csv"
group_test_pred_path = output_dir / "group_test_predictions.csv"
print(f"[INFO] 输出目录: {output_dir.resolve()}")
def mz_label(x, decimals=1) -> str:
    return f"{float(x):.{int(decimals)}f}"
def _save_fig(fig, outpath_no_ext: Path):
    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(str(outpath_no_ext) + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(str(outpath_no_ext) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
def infer_mz_step_and_mz0_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path, nrows=20000)
    if "m/z" not in df.columns:
        raise RuntimeError(f"{csv_path} 缺少 m/z 列")
    mz = pd.to_numeric(df["m/z"], errors="coerce").dropna().values.astype(float)
    mz = mz[(mz >= MZ_MIN - MZ_EPS) & (mz <= MZ_MAX + MZ_EPS)]
    mz = np.sort(np.unique(mz))
    if mz.size < 10:
        raise RuntimeError("m/z 点太少，无法推断步长")
    diffs = np.diff(mz)
    diffs = diffs[diffs > 1e-12]
    step = float(np.median(diffs))
    mz0 = float(mz[0])
    return mz0, step
def convert_peak_params_da_to_bins(mz_step_da: float):
    if mz_step_da is None or mz_step_da <= 0:
        raise RuntimeError("mz_step_da 无效")
    sigma_bins = max(0.5, float(GAUSSIAN_SIGMA_DA) / float(mz_step_da))
    min_width_bins = max(1, int(np.ceil(float(PEAK_MIN_WIDTH_DA) / float(mz_step_da))))
    return float(sigma_bins), int(min_width_bins)
def _dynamic_chunksize(n_samples):
    if n_samples <= 0:
        return 20000
    cs = int(max(2000, min(50000, READ_CHUNK_BASE // max(1, n_samples))))
    return cs
def _parse_sample_and_group_ids(int_cols):
    sample_ids = [str(c) for c in int_cols]
    group_ids = []
    for sid in sample_ids:
        g = sid.split("|", 1)[0].strip() if "|" in sid else sid.strip()
        group_ids.append(g)
    return np.array(sample_ids, dtype=object), np.array(group_ids, dtype=object)
def load_csv_to_tic_spectra(file_path: Path, mz0: float, mz_step: float, idx_min: int, n_bins: int):
    header = pd.read_csv(file_path, nrows=0)
    if "m/z" not in header.columns:
        return None
    int_cols = [c for c in header.columns if c != "m/z"]
    if len(int_cols) == 0:
        return None
    sample_ids, group_ids = _parse_sample_and_group_ids(int_cols)
    n_samples = len(int_cols)
    chunksize = _dynamic_chunksize(n_samples)
    S = np.zeros((n_samples, n_bins), dtype=np.float32)
    usecols = ["m/z"] + int_cols
    INT_COL_BLOCK = 64
    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize):
        mz = pd.to_numeric(chunk["m/z"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        mask = np.isfinite(mz)
        if not np.any(mask):
            continue
        mz = mz[mask]
        sub = chunk.loc[mask, int_cols]
        m2 = (mz >= MZ_MIN - MZ_EPS) & (mz <= MZ_MAX + MZ_EPS)
        if not np.any(m2):
            continue
        mz = mz[m2]
        sub = sub.loc[m2, :]
        idx = np.rint((mz - float(mz0)) / float(mz_step)).astype(np.int64)
        ib = idx - int(idx_min)
        m3 = (ib >= 0) & (ib < n_bins)
        if not np.any(m3):
            continue
        ib = ib[m3].astype(np.int64)
        sub_m3 = sub.loc[m3, :]
        u, inv = np.unique(ib, return_inverse=True)
        for j0 in range(0, n_samples, INT_COL_BLOCK):
            j1 = min(n_samples, j0 + INT_COL_BLOCK)
            cols_blk = int_cols[j0:j1]
            Xblk = sub_m3[cols_blk].to_numpy(dtype=np.float32, copy=False)
            sums_blk = np.zeros((len(u), j1 - j0), dtype=np.float32)
            np.add.at(sums_blk, inv, Xblk)
            S[j0:j1, u] += sums_blk.T
            del Xblk, sums_blk
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    S /= row_sums
    return S, sample_ids, group_ids
def ensure_proba_full_order(proba, model_classes, n_classes):
    proba = np.asarray(proba, dtype=float)
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    model_classes = np.asarray(model_classes, dtype=int)
    for j, c in enumerate(model_classes):
        if 0 <= int(c) < n_classes:
            out[:, int(c)] = proba[:, j]
    return out
def _temp_scale_probs(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    T = float(max(eps, T))
    logp = np.log(np.clip(p, eps, 1.0))
    z = logp / T
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    den = np.sum(ez, axis=1, keepdims=True)
    den[den == 0] = 1.0
    out = ez / den
    return out.astype(np.float64)
def _nll_from_probs(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.clip(p, eps, 1.0)
    idx = np.arange(p.shape[0], dtype=int)
    return float(-np.mean(np.log(p[idx, y])))
def fit_temperature_scaling_from_proba(
    probs_oof: np.ndarray,
    y_true: np.ndarray,
    Tmin: float = 0.25,
    Tmax: float = 6.0,
    n_grid: int = 60,
    refine_rounds: int = 3,
    eps: float = 1e-12
):
    p = np.asarray(probs_oof, dtype=np.float64)
    y = np.asarray(y_true, dtype=int).ravel()
    assert p.ndim == 2 and p.shape[0] == y.shape[0]
    Ts = np.exp(np.linspace(np.log(Tmin), np.log(Tmax), int(n_grid)))
    best_T, best_loss = float(Ts[0]), float("inf")
    for T in Ts:
        loss = _nll_from_probs(_temp_scale_probs(p, float(T), eps=eps), y, eps=eps)
        if loss < best_loss:
            best_loss, best_T = loss, float(T)
    lo, hi = best_T / 1.8, best_T * 1.8
    lo = max(Tmin, lo); hi = min(Tmax, hi)
    for _ in range(int(refine_rounds)):
        Ts2 = np.exp(np.linspace(np.log(lo), np.log(hi), 30))
        for T in Ts2:
            loss = _nll_from_probs(_temp_scale_probs(p, float(T), eps=eps), y, eps=eps)
            if loss < best_loss:
                best_loss, best_T = loss, float(T)
        lo, hi = best_T / 1.6, best_T * 1.6
        lo = max(Tmin, lo); hi = min(Tmax, hi)
    nll_before = _nll_from_probs(p, y, eps=eps)
    nll_after = _nll_from_probs(_temp_scale_probs(p, best_T, eps=eps), y, eps=eps)
    info = {"nll_before": float(nll_before), "nll_after": float(nll_after)}
    return float(best_T), info
def apply_calibrator(proba: np.ndarray, calibrator):
    if calibrator is None:
        return proba
    if not isinstance(calibrator, dict):
        return proba
    method = str(calibrator.get("method", "")).lower().strip()
    if method == "temperature":
        T = float(calibrator.get("T", 1.0))
        return _temp_scale_probs(proba, T).astype(np.float64)
    return proba
def _score_func(y_true, y_pred, scoring="accuracy"):
    if scoring == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    return accuracy_score(y_true, y_pred)
def class_group_stats(y, groups, id_map):
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=str)
    rows = []
    for c in np.unique(y):
        m = (y == c)
        rows.append({
            "class": id_map[int(c)] if id_map else int(c),
            "n_samples": int(m.sum()),
            "n_groups": int(len(np.unique(groups[m]))),
        })
    return pd.DataFrame(rows).sort_values("n_groups")
def make_group_cv(y, groups, target_splits=5, rng=42):
    y = np.asarray(y, dtype=int).ravel()
    groups = np.asarray(groups, dtype=str).ravel()
    min_g = np.inf
    for c in np.unique(y):
        m = (y == c)
        min_g = min(min_g, len(np.unique(groups[m])))
    min_g = int(min_g) if np.isfinite(min_g) else 0
    n_splits = max(2, min(int(target_splits), int(min_g))) if min_g > 0 else 2
    if min_g >= 2:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(rng))
            _ = next(cv.split(np.zeros_like(y), y, groups))
            print(f"[INFO] GroupCV=StratifiedGroupKFold splits={n_splits}, min_groups_per_class={min_g}")
            return ("StratifiedGroupKFold", cv, n_splits, min_g)
        except Exception:
            cv = GroupKFold(n_splits=n_splits)
            print(f"[INFO] GroupCV=GroupKFold splits={n_splits}, min_groups_per_class={min_g}")
            return ("GroupKFold", cv, n_splits, min_g)
    print(f"[WARN] min_groups_per_class={min_g} 太少，退化为样本级 StratifiedKFold（仅用于CV/调参）。")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(rng))
    return ("StratifiedKFold_sample_level", cv, 5, min_g)
def _peak_windows_from_mean_spectrum(mean_spec_1d: np.ndarray, mz_step_da: float, max_peaks: int = 0):
    sigma_bins, min_width_bins = convert_peak_params_da_to_bins(mz_step_da)
    mean_spec_1d = np.asarray(mean_spec_1d, dtype=np.float32).ravel()
    mean_smooth = gaussian_filter1d(mean_spec_1d, sigma=float(sigma_bins), mode="nearest")
    vmax = float(mean_smooth.max()) if mean_smooth.size else 0.0
    prom = max(float(PEAK_MIN_INTENSITY), float(PEAK_RELATIVE_HEIGHT) * vmax)
    peaks, props = find_peaks(mean_smooth, prominence=prom, width=int(min_width_bins))
    if len(peaks) == 0:
        return []
    prominences = props.get("prominences", np.ones(len(peaks), dtype=float))
    left_ips = props.get("left_ips", np.maximum(peaks - 1, 0))
    right_ips = props.get("right_ips", np.minimum(peaks + 1, mean_smooth.size - 1))
    left = np.floor(left_ips).astype(int)
    right = np.ceil(right_ips).astype(int)
    if max_peaks and max_peaks > 0:
        order = np.argsort(np.asarray(prominences))[::-1][: int(max_peaks)]
        peaks = np.asarray(peaks)[order]
        left = np.asarray(left)[order]
        right = np.asarray(right)[order]
        prominences = np.asarray(prominences)[order]
    order2 = np.argsort(peaks)
    peaks = np.asarray(peaks)[order2]
    left = np.asarray(left)[order2]
    right = np.asarray(right)[order2]
    prominences = np.asarray(prominences)[order2]
    windows = []
    for p, l, r, pr in zip(peaks, left, right, prominences):
        l = max(int(l), 0)
        r = min(int(r), mean_smooth.size - 1)
        windows.append({"l": int(l), "r": int(r), "apex": int(p), "prom": float(pr)})
    return windows
def _merge_windows(windows: list, merge_overlap: bool = True):
    if not windows:
        return []
    ws = sorted(windows, key=lambda w: (int(w["l"]), int(w["r"])))
    merged = [ws[0].copy()]
    def _better(a, b):
        pa = float(a.get("prom", 0.0))
        pb = float(b.get("prom", 0.0))
        return b if pb > pa else a
    for w in ws[1:]:
        l, r = int(w["l"]), int(w["r"])
        last = merged[-1]
        if merge_overlap and l <= int(last["r"]):
            last["r"] = max(int(last["r"]), r)
            best = _better(last, w)
            last["apex"] = int(best.get("apex", last["apex"]))
            last["prom"] = float(best.get("prom", last.get("prom", 0.0)))
            last["l"] = min(int(last["l"]), l)
        else:
            merged.append(w.copy())
    return merged
def find_peak_windows_union_by_class_streaming(X_mm, y, mz_step_da: float, id_map: dict, tr_idx: np.ndarray):
    y = np.asarray(y, dtype=int).ravel()
    classes = np.unique(y[tr_idx])
    n_bins = X_mm.shape[1]
    all_windows = []
    sigma_bins, min_width_bins = convert_peak_params_da_to_bins(mz_step_da)
    for c in classes:
        idx_c = tr_idx[y[tr_idx] == c]
        if idx_c.size < int(PEAK_MIN_SAMPLES_PER_CLASS):
            cname = id_map.get(int(c), str(c))
            print(f"[PEAK] class={cname} 样本数{idx_c.size} < {PEAK_MIN_SAMPLES_PER_CLASS}，跳过找峰")
            continue
        sum_spec = np.zeros(n_bins, dtype=np.float64)
        cnt = 0
        for s in range(0, len(idx_c), BATCH_SIZE_MEAN):
            b = idx_c[s:s + BATCH_SIZE_MEAN]
            Xb = np.asarray(X_mm[b], dtype=np.float32)
            sum_spec += Xb.sum(axis=0)
            cnt += len(b)
            del Xb
        mean_spec = (sum_spec / max(1, cnt)).astype(np.float32)
        ws = _peak_windows_from_mean_spectrum(mean_spec, mz_step_da=mz_step_da, max_peaks=int(PEAK_MAX_PEAKS_PER_CLASS))
        ws = _merge_windows(ws, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
        cname = id_map.get(int(c), str(c))
        print(f"[PEAK] class={cname}: windows={len(ws)} | sigma_bins={sigma_bins:.2f} | min_width_bins={min_width_bins}")
        all_windows.extend(ws)
    if not all_windows:
        raise RuntimeError("每类找峰后窗口为空：请降低 PEAK_RELATIVE_HEIGHT/PEAK_MIN_WIDTH_DA 或降低 PEAK_MIN_SAMPLES_PER_CLASS。")
    all_windows = _merge_windows(all_windows, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
    if UNION_MAX_WINDOWS and UNION_MAX_WINDOWS > 0 and len(all_windows) > int(UNION_MAX_WINDOWS):
        all_windows_sorted = sorted(all_windows, key=lambda w: float(w.get("prom", 0.0)), reverse=True)[: int(UNION_MAX_WINDOWS)]
        all_windows = _merge_windows(all_windows_sorted, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
    print(f"[PEAK] union windows(after merge)={len(all_windows)}")
    windows = [{"l": int(w["l"]), "r": int(w["r"]), "apex": int(w["apex"])} for w in all_windows]
    return windows
def find_global_peak_windows_from_train_mean_streaming(X_mm, mz_step_da: float, tr_idx: np.ndarray):
    n_bins = X_mm.shape[1]
    sum_spec = np.zeros(n_bins, dtype=np.float64)
    cnt = 0
    for s in range(0, len(tr_idx), BATCH_SIZE_MEAN):
        b = tr_idx[s:s + BATCH_SIZE_MEAN]
        Xb = np.asarray(X_mm[b], dtype=np.float32)
        sum_spec += Xb.sum(axis=0)
        cnt += len(b)
        del Xb
    mean_spec = (sum_spec / max(1, cnt)).astype(np.float32)
    ws = _peak_windows_from_mean_spectrum(mean_spec, mz_step_da=mz_step_da, max_peaks=0)
    ws = _merge_windows(ws, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
    windows = [{"l": int(w["l"]), "r": int(w["r"]), "apex": int(w["apex"])} for w in ws]
    print(f"[PEAK] global-mean windows={len(windows)}")
    return windows
def transform_to_peak_areas_batch(X_source, indices, windows: list, mz_step_da: float, batch_size=64):
    sigma_bins, _ = convert_peak_params_da_to_bins(mz_step_da)
    n = len(indices)
    p = len(windows)
    Xp = np.zeros((n, p), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        bidx = indices[start:end]
        Xb = np.asarray(X_source[bidx], dtype=np.float32)
        Xs = gaussian_filter1d(Xb, sigma=float(sigma_bins), axis=1, mode="nearest")
        for j, w in enumerate(windows):
            l, r = int(w["l"]), int(w["r"])
            l = max(0, min(l, Xs.shape[1] - 1))
            r = max(l, min(r, Xs.shape[1] - 1))
            Xp[start:end, j] = Xs[:, l:r + 1].sum(axis=1)
        del Xb, Xs
        gc.collect()
    if RE_NORMALIZE_PEAK_AREAS:
        Xp = Xp / (Xp.sum(axis=1, keepdims=True) + 1e-10)
    return Xp
def windows_to_feature_names(windows, mz0, mz_step):
    names = []
    for w in windows:
        apex = int(w["apex"])
        mz_c = float(mz0 + apex * mz_step)
        names.append(mz_label(mz_c, MZ_LABEL_DECIMALS))
    return names
def streaming_feature_variance(X_source, indices, windows, mz_step_da: float, batch_size=64):
    sigma_bins, _ = convert_peak_params_da_to_bins(mz_step_da)
    p = len(windows)
    n_total = 0
    mean = np.zeros(p, dtype=np.float64)
    M2 = np.zeros(p, dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        end = min(len(indices), start + batch_size)
        bidx = indices[start:end]
        Xb = np.asarray(X_source[bidx], dtype=np.float32)
        Xs = gaussian_filter1d(Xb, sigma=float(sigma_bins), axis=1, mode="nearest")
        B = Xs.shape[0]
        Xp = np.zeros((B, p), dtype=np.float32)
        for j, w in enumerate(windows):
            l, r = int(w["l"]), int(w["r"])
            l = max(0, min(l, Xs.shape[1] - 1))
            r = max(l, min(r, Xs.shape[1] - 1))
            Xp[:, j] = Xs[:, l:r + 1].sum(axis=1)
        if RE_NORMALIZE_PEAK_AREAS:
            Xp = Xp / (Xp.sum(axis=1, keepdims=True) + 1e-10)
        Xp64 = Xp.astype(np.float64, copy=False)
        m = Xp64.shape[0]
        if m > 0:
            batch_mean = Xp64.mean(axis=0)
            batch_M2 = ((Xp64 - batch_mean) ** 2).sum(axis=0)
            if n_total == 0:
                mean = batch_mean
                M2 = batch_M2
                n_total = m
            else:
                delta = batch_mean - mean
                n_new = n_total + m
                mean = mean + delta * (m / n_new)
                M2 = M2 + batch_M2 + (delta ** 2) * (n_total * m / n_new)
                n_total = n_new
        del Xb, Xs, Xp, Xp64
        gc.collect()
    if n_total <= 1:
        return np.zeros(p, dtype=np.float64)
    var = M2 / (n_total - 1)
    return var.astype(np.float64)
def plot_confusion_matrix(cm, labels, title, outpath_no_ext, normalize=None, double_col=True):
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
def plot_multiclass_roc_ovr(y_true, proba, class_names, outpath_no_ext, double_col=True, title_suffix="proba"):
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
def plot_learning_curve_groupcv(estimator, X, y, groups, scoring, cv_splits, outpath_no_ext, rng=42, double_col=True):
    cv_name, cv, real_splits, min_g = make_group_cv(y, groups, target_splits=cv_splits, rng=rng)
    train_sizes = np.array(LEARNING_CURVE_TRAIN_SIZES, dtype=float)
    train_sizes = np.clip(train_sizes, 0.05, 1.0)
    cv_for_lc = cv if cv_name != "StratifiedKFold_sample_level" else StratifiedKFold(n_splits=5, shuffle=True, random_state=int(rng))
    sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups if cv_name != "StratifiedKFold_sample_level" else None,
        cv=cv_for_lc,
        scoring=str(scoring),
        train_sizes=train_sizes,
        n_jobs=-1,
        shuffle=True,
        random_state=int(rng)
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    width_mm = ELSEVIER_DOUBLE_COL_MM if double_col else ELSEVIER_SINGLE_COL_MM
    fig, ax = plt.subplots(figsize=journal_figsize(width_mm=width_mm, aspect=0.75))
    ax.plot(sizes, train_mean, marker="o", lw=1.2, label="Train")
    ax.plot(sizes, val_mean, marker="o", lw=1.2, label="CV")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_xlabel("Training examples", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Score ({scoring})", fontsize=10, fontweight="bold")
    ax.set_title("Learning curve (GroupCV)", pad=10, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)
    _save_fig(fig, outpath_no_ext)
def plot_shap_summary_dot(shap_values_2d, X_plot, feature_names, outpath_no_ext):
    try:
        import shap
    except Exception:
        print("[WARN] shap 未安装，跳过 SHAP。")
        return
    shap_values_2d = np.asarray(shap_values_2d)
    shap.summary_plot(
        shap_values_2d,
        X_plot,
        feature_names=feature_names,
        plot_type="dot",
        max_display=min(int(SHAP_MAX_DISPLAY), len(feature_names)),
        show=False
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("SHAP summary (dot) - RF peak area", pad=4, fontweight="bold")
    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(str(outpath_no_ext) + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(str(outpath_no_ext) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
def plot_shap_mean_abs_stacked_bar(shap_values_list, feature_names, class_names, outpath_no_ext, double_col=True):
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
    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(str(outpath_no_ext) + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(str(outpath_no_ext) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
def prepare_cv_splits_only(y_full, g_full, n_splits=5, rng=42):
    cv_name, cv, real_splits, min_g = make_group_cv(y_full, g_full, target_splits=n_splits, rng=rng)
    if cv_name == "StratifiedKFold_sample_level":
        splits = list(cv.split(np.zeros(len(y_full)), y_full))
    else:
        splits = list(cv.split(np.zeros(len(y_full)), y_full, g_full))
    out = []
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        out.append({
            "fold": fold_i,
            "tr_idx": np.asarray(tr_idx, dtype=np.int64),
            "va_idx": np.asarray(va_idx, dtype=np.int64),
        })
    print(f"[INFO] CV splits prepared: {len(out)} folds")
    return out
def precompute_fold_windows_k(X_mm, y_full, mz_step, id_map, splits):
    fold_meta = []
    for fd in splits:
        tr_idx = fd["tr_idx"]
        if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
            windows = find_peak_windows_union_by_class_streaming(
                X_mm, y_full, mz_step_da=mz_step, id_map=id_map, tr_idx=tr_idx
            )
        else:
            windows = find_global_peak_windows_from_train_mean_streaming(
                X_mm, mz_step_da=mz_step, tr_idx=tr_idx
            )
        windows_k = windows
        top_idx = None
        if USE_FEATURE_SELECTION:
            vars_ = streaming_feature_variance(
                X_source=X_mm,
                indices=tr_idx,
                windows=windows,
                mz_step_da=mz_step,
                batch_size=BATCH_SIZE_FEAT
            )
            k = min(int(TOP_K_FEATURES), len(windows))
            top_idx = np.argsort(vars_)[::-1][:k]
            top_idx.sort()
            windows_k = [windows[i] for i in top_idx]
        fold_meta.append({
            "fold": fd["fold"],
            "tr_idx": fd["tr_idx"],
            "va_idx": fd["va_idx"],
            "windows_k": windows_k,
        })
        print(f"[INFO] Fold {fd['fold']+1}: windows_all={len(windows)} windows_k={len(windows_k)}")
    return fold_meta
def optuna_tune_rf_streaming(X_mm, y_full, mz_step, fold_meta, scoring="accuracy", n_trials=50, timeout=None, rng=42):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
        scores = []
        for fd in fold_meta:
            tr_idx, va_idx = fd["tr_idx"], fd["va_idx"]
            windows_k = fd["windows_k"]
            Xtr = transform_to_peak_areas_batch(X_mm, tr_idx, windows_k, mz_step_da=mz_step, batch_size=BATCH_SIZE_FEAT)
            Xva = transform_to_peak_areas_batch(X_mm, va_idx, windows_k, mz_step_da=mz_step, batch_size=BATCH_SIZE_FEAT)
            ytr = np.asarray(y_full[tr_idx], dtype=int)
            yva = np.asarray(y_full[va_idx], dtype=int)
            clf = RandomForestClassifier(**params, random_state=int(rng), n_jobs=-1)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xva)
            scores.append(float(_score_func(yva, pred, scoring=scoring)))
            del Xtr, Xva, ytr, yva, clf
            gc.collect()
        return float(np.mean(scores)) if scores else 0.0
    sampler = optuna.samplers.TPESampler(seed=int(rng))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), timeout=timeout)
    best = study.best_params
    best["best_score"] = float(study.best_value)
    return best, study
def main():
    t0 = time.time()
    train_dir = Path(TRAINING_FOLDER_PATH)
    if not train_dir.is_dir():
        sys.exit(f"[ERROR] 训练目录不存在: {train_dir}")
    csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        sys.exit(f"[ERROR] 训练目录为空: {train_dir}")
    class_names = sorted(list({p.stem.split("_")[0] for p in csv_files}))
    label_map = {name: i for i, name in enumerate(class_names)}
    id_map = {i: name for name, i in label_map.items()}
    n_classes = len(class_names)
    print(f"[INFO] 类别({n_classes}): {class_names}")
    mz0, mz_step = infer_mz_step_and_mz0_from_csv(csv_files[0])
    print(f"[INFO] m/z 网格: mz0={mz0:.6f}, mz_step={mz_step:.6f}")
    idx_min = int(np.rint((MZ_MIN - mz0) / mz_step))
    idx_max = int(np.rint((MZ_MAX - mz0) / mz_step))
    n_bins = int(idx_max - idx_min + 1)
    print(f"[INFO] master bins: {n_bins} ({MZ_MIN}~{MZ_MAX})")
    total_samples = 0
    file_infos = []
    for p in csv_files:
        h = pd.read_csv(p, nrows=0)
        int_cols = [c for c in h.columns if c != "m/z"]
        n_samp = len(int_cols)
        if n_samp == 0:
            continue
        prefix = p.stem.split("_")[0]
        if prefix not in label_map:
            continue
        file_infos.append((p, prefix, n_samp))
        total_samples += n_samp
    if total_samples == 0:
        sys.exit("[ERROR] 训练集中未找到有效强度列")
    print(f"[INFO] Total samples (train): {total_samples}")
    mm_path = output_dir / "X_train_tic_float32.memmap"
    X_mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(total_samples, n_bins))
    y_train = np.zeros(total_samples, dtype=np.int32)
    g_train = np.empty(total_samples, dtype=object)
    s_train = np.empty(total_samples, dtype=object)
    print("[INFO] Loading training data into memmap (TIC-normalized)...")
    cursor = 0
    for p, prefix, n_samp in file_infos:
        label = int(label_map[prefix])
        out = load_csv_to_tic_spectra(p, mz0, mz_step, idx_min, n_bins)
        if out is None:
            print(f"[WARN] Skip invalid file: {p.name}")
            continue
        S, sample_ids, group_ids = out
        X_mm[cursor:cursor + S.shape[0], :] = S
        y_train[cursor:cursor + S.shape[0]] = label
        g_train[cursor:cursor + S.shape[0]] = group_ids
        s_train[cursor:cursor + S.shape[0]] = sample_ids
        cursor += S.shape[0]
        del S
        gc.collect()
        print(f"  - {p.name}: samples={n_samp}, cursor={cursor}/{total_samples}")
    if cursor != total_samples:
        total_samples = cursor
        X_mm = np.memmap(mm_path, dtype=np.float32, mode="r+", shape=(cursor, n_bins))
        y_train = y_train[:cursor]
        g_train = g_train[:cursor]
        s_train = s_train[:cursor]
        print(f"[WARN] Some files skipped. Effective samples: {total_samples}")
    print("[INFO] Train 每类样本/组统计（n_groups 最小的最危险）:")
    print(class_group_stats(y_train, g_train, id_map).to_string(index=False))
    if ENABLE_OPTUNA:
        splits = prepare_cv_splits_only(
            y_full=y_train,
            g_full=g_train,
            n_splits=int(OPTUNA_CV_SPLITS),
            rng=RNG
        )
        fold_meta = precompute_fold_windows_k(
            X_mm=X_mm,
            y_full=y_train,
            mz_step=mz_step,
            id_map=id_map,
            splits=splits
        )
        best_params, study = optuna_tune_rf_streaming(
            X_mm=X_mm,
            y_full=y_train,
            mz_step=mz_step,
            fold_meta=fold_meta,
            scoring=str(OPTUNA_SCORING),
            n_trials=int(OPTUNA_N_TRIALS),
            timeout=OPTUNA_TIMEOUT,
            rng=RNG
        )
        print(f"[INFO] Optuna best: {best_params}")
        try:
            study.trials_dataframe().to_csv(output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass
        with open(output_dir / "optuna_best.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        del fold_meta, splits
        gc.collect()
    else:
        best_params = {
            "n_estimators": 800,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": "balanced",
            "criterion": "gini",
            "best_score": None
        }
        print("[WARN] 未启用 Optuna，使用默认:", best_params)
    print("[INFO] 用全量 Train 拟合峰窗口并生成最终特征（两遍法 TopK）...")
    all_idx = np.arange(total_samples, dtype=np.int64)
    if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
        windows_all = find_peak_windows_union_by_class_streaming(
            X_mm, y_train, mz_step_da=mz_step, id_map=id_map, tr_idx=all_idx
        )
    else:
        windows_all = find_global_peak_windows_from_train_mean_streaming(
            X_mm, mz_step_da=mz_step, tr_idx=all_idx
        )
    peak_feature_names_all = windows_to_feature_names(windows_all, mz0=mz0, mz_step=mz_step)
    with open(features_save_path, "w", encoding="utf-8") as f:
        json.dump(peak_feature_names_all, f, ensure_ascii=False, indent=2)
    top_idx = None
    windows_k = windows_all
    final_names = peak_feature_names_all
    if USE_FEATURE_SELECTION:
        print("[INFO] Streaming variance for TopK on full train (no full feature matrix)...")
        vars_ = streaming_feature_variance(
            X_source=X_mm,
            indices=all_idx,
            windows=windows_all,
            mz_step_da=mz_step,
            batch_size=BATCH_SIZE_FEAT
        )
        k = min(int(TOP_K_FEATURES), len(windows_all))
        top_idx = np.argsort(vars_)[::-1][:k]
        top_idx.sort()
        windows_k = [windows_all[i] for i in top_idx]
        final_names = [peak_feature_names_all[i] for i in top_idx]
    X_train_peak = transform_to_peak_areas_batch(
        X_source=X_mm,
        indices=all_idx,
        windows=windows_k,
        mz_step_da=mz_step,
        batch_size=BATCH_SIZE_FEAT
    )
    calibrator = None
    try:
        print("[CAL] Fitting temperature scaling on OOF probabilities (GroupCV)...")
        cv_name, cv, real_splits, min_g = make_group_cv(y_train, g_train, target_splits=int(OPTUNA_CV_SPLITS), rng=RNG)
        if cv_name == "StratifiedKFold_sample_level":
            splits_cal = list(cv.split(np.zeros(len(y_train)), y_train))
        else:
            splits_cal = list(cv.split(np.zeros(len(y_train)), y_train, g_train))
        oof = np.zeros((len(y_train), n_classes), dtype=np.float64)
        filled = np.zeros(len(y_train), dtype=bool)
        for fold_i, (tr_idx, va_idx) in enumerate(splits_cal):
            clf_cal = RandomForestClassifier(
                n_estimators=int(best_params["n_estimators"]),
                max_depth=int(best_params["max_depth"]),
                min_samples_split=int(best_params["min_samples_split"]),
                min_samples_leaf=int(best_params["min_samples_leaf"]),
                max_features=best_params["max_features"],
                bootstrap=bool(best_params["bootstrap"]),
                class_weight=best_params["class_weight"],
                criterion=best_params["criterion"],
                random_state=int(RNG) + int(fold_i),
                n_jobs=-1
            )
            clf_cal.fit(X_train_peak[tr_idx], y_train[tr_idx])
            proba_raw = clf_cal.predict_proba(X_train_peak[va_idx])
            proba_full = ensure_proba_full_order(proba_raw, clf_cal.classes_, n_classes) if hasattr(clf_cal, "classes_") else proba_raw
            oof[va_idx] = proba_full
            filled[va_idx] = True
            del clf_cal, proba_raw, proba_full
            gc.collect()
        if not np.all(filled):
            print(f"[WARN] OOF not fully filled: {int(filled.sum())}/{len(filled)} (will fit on filled subset)")
        T, info = fit_temperature_scaling_from_proba(oof[filled], y_train[filled])
        calibrator = {
            "method": "temperature",
            "T": float(T),
            "fitted_on": "oof_groupcv",
            "n_samples": int(filled.sum()),
            "nll_before": float(info.get("nll_before", np.nan)),
            "nll_after": float(info.get("nll_after", np.nan)),
        }
        with open(output_dir / "proba_calibrator.json", "w", encoding="utf-8") as f:
            json.dump(calibrator, f, ensure_ascii=False, indent=2)
        print(f"[CAL] Temperature scaling fitted: T={float(T):.4f} | NLL {info.get('nll_before'):.4f} -> {info.get('nll_after'):.4f}")
    except Exception as e:
        print(f"[WARN] Probability calibration skipped (can ignore): {e}")
        calibrator = None
    rf = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        min_samples_split=int(best_params["min_samples_split"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        max_features=best_params["max_features"],
        bootstrap=bool(best_params["bootstrap"]),
        class_weight=best_params["class_weight"],
        criterion=best_params["criterion"],
        random_state=int(RNG),
        n_jobs=-1
    )
    rf.fit(X_train_peak, y_train)
    if ENABLE_LEARNING_CURVE:
        try:
            print("[INFO] 绘制学习曲线（GroupCV）...")
            rf_for_lc = RandomForestClassifier(
                n_estimators=int(best_params["n_estimators"]),
                max_depth=int(best_params["max_depth"]),
                min_samples_split=int(best_params["min_samples_split"]),
                min_samples_leaf=int(best_params["min_samples_leaf"]),
                max_features=best_params["max_features"],
                bootstrap=bool(best_params["bootstrap"]),
                class_weight=best_params["class_weight"],
                criterion=best_params["criterion"],
                random_state=int(RNG),
                n_jobs=-1
            )
            plot_learning_curve_groupcv(
                estimator=rf_for_lc,
                X=X_train_peak,
                y=y_train,
                groups=g_train,
                scoring=str(LEARNING_CURVE_SCORING),
                cv_splits=int(LEARNING_CURVE_CV_SPLITS),
                outpath_no_ext=fig_dir / "fig_06_learning_curve_groupcv",
                rng=RNG
            )
            print("[INFO] 学习曲线已输出到 figures/。")
        except Exception as e:
            print(f"[WARN] 学习曲线失败(可忽略): {e}")
    test_dir = Path(TEST_FOLDER_PATH)
    if not test_dir.is_dir():
        sys.exit(f"[ERROR] 测试目录不存在: {test_dir}")
    y_true_all = []
    pred_all = []
    proba_all = []
    g_test_all = []
    s_test_all = []
    shap_X_buf = []
    shap_take = int(SHAP_MAX_SAMPLES)
    for p in sorted(test_dir.glob("*.csv")):
        prefix = p.stem.split("_")[0]
        if prefix not in label_map:
            print(f"[WARN] 跳过: {p.name} (前缀 '{prefix}' 不在训练类别中)")
            continue
        out = load_csv_to_tic_spectra(p, mz0, mz_step, idx_min, n_bins)
        if out is None:
            continue
        S, sample_ids, group_ids = out
        idx_local = np.arange(S.shape[0], dtype=np.int64)
        Xp = transform_to_peak_areas_batch(
            X_source=S,
            indices=idx_local,
            windows=windows_k,
            mz_step_da=mz_step,
            batch_size=BATCH_SIZE_FEAT
        )
        y_true = np.full(S.shape[0], int(label_map[prefix]), dtype=np.int32)
        pred = rf.predict(Xp)
        proba_raw = rf.predict_proba(Xp)
        proba = ensure_proba_full_order(proba_raw, rf.classes_, n_classes) if hasattr(rf, "classes_") else proba_raw
        proba = apply_calibrator(proba, calibrator)
        y_true_all.append(y_true)
        pred_all.append(pred.astype(np.int32, copy=False))
        proba_all.append(proba.astype(np.float32, copy=False))
        g_test_all.append(np.asarray(group_ids, dtype=object))
        s_test_all.append(np.asarray(sample_ids, dtype=object))
        if ENABLE_SHAP and shap_take > 0:
            take = min(shap_take, Xp.shape[0])
            if take > 0:
                shap_X_buf.append(Xp[:take].copy())
                shap_take -= take
        del S, Xp, proba_raw, proba, pred, y_true
        gc.collect()
    if len(y_true_all) == 0:
        sys.exit("[ERROR] 未读取到任何测试集数据")
    y_test = np.concatenate(y_true_all).astype(np.int32, copy=False)
    preds = np.concatenate(pred_all).astype(np.int32, copy=False)
    probs = np.vstack(proba_all).astype(np.float32, copy=False)
    g_test = np.concatenate(g_test_all).astype(object, copy=False)
    s_test = np.concatenate(s_test_all).astype(object, copy=False)
    del y_true_all, pred_all, proba_all, g_test_all, s_test_all
    gc.collect()
    print(f"[INFO] 最终维度: {X_train_peak.shape[1]} | Train={len(X_train_peak)} | Test={len(y_test)}")
    names = [id_map[i] for i in range(n_classes)]
    labels_all = list(range(n_classes))
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] Test Acc: {acc:.4f}")
    print(classification_report(y_test, preds, labels=labels_all, target_names=names, zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=labels_all)
    plot_confusion_matrix(cm, names, "Confusion matrix (count)", fig_dir / "fig_01_confusion_matrix", normalize=None)
    plot_confusion_matrix(cm, names, "Confusion matrix (normalized)", fig_dir / "fig_02_confusion_matrix_norm", normalize="true")
    max_p = probs.max(axis=1)
    if ENABLE_ROC:
        try:
            plot_multiclass_roc_ovr(
                y_true=y_test,
                proba=probs,
                class_names=names,
                outpath_no_ext=fig_dir / "fig_03_roc_ovr_proba",
                title_suffix="RF probabilities"
            )
            print("[INFO] ROC 已输出到 figures/。")
        except Exception as e:
            print(f"[WARN] ROC 绘制失败(可忽略): {e}")
    if ENABLE_SHAP:
        try:
            import shap
            print("[INFO] 计算 SHAP（TreeExplainer）...")
            if len(shap_X_buf) == 0:
                raise RuntimeError("SHAP 缓存样本为空（测试集可能为空或读取失败）。")
            X_plot = np.vstack(shap_X_buf).astype(np.float32, copy=False)
            del shap_X_buf
            gc.collect()
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(X_plot)
            sv_list = None
            if isinstance(shap_vals, list):
                sv_list = [np.asarray(v) for v in shap_vals]
            else:
                sv = np.asarray(shap_vals)
                if sv.ndim == 3:
                    sv_list = [sv[:, :, i] for i in range(sv.shape[2])]
                elif sv.ndim == 2:
                    sv_list = [sv]
            if sv_list is not None and len(sv_list) > 1:
                plot_shap_mean_abs_stacked_bar(
                    shap_values_list=sv_list,
                    feature_names=final_names,
                    class_names=names,
                    outpath_no_ext=fig_dir / "fig_04_shap_stacked_bar"
                )
            else:
                print("[WARN] SHAP 返回非多分类结果，堆叠条形图可忽略。")
            if sv_list is None or len(sv_list) == 0:
                raise RuntimeError("SHAP 返回空结果，跳过。")
            if len(sv_list) == 1:
                shap_dot = np.asarray(sv_list[0], dtype=np.float32)
            else:
                yhat_plot = rf.predict(X_plot).astype(int)
                cls_order = np.asarray(getattr(rf, "classes_", np.arange(len(sv_list))), dtype=int)
                cls2pos = {int(c): i for i, c in enumerate(cls_order)}
                shap_dot = np.zeros((X_plot.shape[0], X_plot.shape[1]), dtype=np.float32)
                for i in range(X_plot.shape[0]):
                    c = int(yhat_plot[i])
                    pos = cls2pos.get(c, 0)
                    if pos < 0 or pos >= len(sv_list):
                        pos = 0
                    if i < sv_list[pos].shape[0]:
                        shap_dot[i, :] = np.asarray(sv_list[pos][i, :], dtype=np.float32)
            plot_shap_summary_dot(shap_dot, X_plot, final_names, fig_dir / "fig_05_shap_summary_dot")
            print("[INFO] SHAP 图已输出到 figures/。")
            del X_plot, shap_vals, sv_list, shap_dot
            gc.collect()
        except Exception as e:
            print(f"[WARN] SHAP 计算/绘图失败(可忽略): {e}")
    out = pd.DataFrame({
        "Sample": np.asarray(s_test, dtype=str),
        "Group": np.asarray(g_test, dtype=str),
        "True_Label": [id_map[int(i)] for i in y_test],
        "Pred_Label": [id_map[int(i)] for i in preds],
        "Similarity": max_p,
        "Is_Unknown": (max_p < float(UNKNOWN_THRESHOLD))
    })
    out.to_csv(test_pred_save_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 样本级预测已保存: {test_pred_save_path}")
    grp_res = []
    for g in np.unique(g_test):
        idx = np.where(g_test == g)[0]
        true_name = id_map[int(y_test[idx[0]])]
        mean_p = probs[idx].mean(axis=0)
        pred_idx = int(mean_p.argmax())
        conf = float(mean_p.max())
        pred_name = "UNKNOWN" if (GROUP_UNKNOWN_THRESHOLD > 0 and conf < GROUP_UNKNOWN_THRESHOLD) else id_map[pred_idx]
        grp_res.append({"Group": g, "True_Label": true_name, "Pred_Label": pred_name, "Confidence": conf})
    pd.DataFrame(grp_res).to_csv(group_test_pred_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] group级预测已保存: {group_test_pred_path}")
    pd.DataFrame({"id": list(id_map.keys()), "name": list(id_map.values())}).to_csv(
        labels_save_path, index=False, encoding="utf-8-sig"
    )
    joblib.dump(
        {
            "rf": rf,
            "id_map": id_map,
            "best_params": best_params,
            "mz0": mz0,
            "mz_step": mz_step,
            "feature_type": "peak_area_per_class_union" if str(PEAK_WINDOWS_MODE).lower() == "per_class_union" else "peak_area_global_mean",
            "windows_all": windows_all,
            "windows_used": windows_k,
            "peak_feature_names_all": peak_feature_names_all,
            "feature_names_used": final_names,
            "top_idx_in_windows_all": top_idx,
            "memmap_path": str(mm_path),
            "n_bins": int(n_bins),
            "idx_min": int(idx_min),
            "proba_calibrator": calibrator,
            "unknown_threshold": float(UNKNOWN_THRESHOLD),
            "group_unknown_threshold": float(GROUP_UNKNOWN_THRESHOLD),
        },
        model_save_path
    )
    print(f"[INFO] 模型已保存: {model_save_path}")
    print(f"[INFO] 全部完成，用时 {time.time() - t0:.1f}s")
if __name__ == "__main__":
    main()