import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    sns = None
    _HAS_SNS = False
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import minimize
try:
    from scipy.special import logsumexp
except Exception:
    def logsumexp(a, axis=None, keepdims=False):
        a = np.asarray(a)
        amax = np.max(a, axis=axis, keepdims=True)
        out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError("缺少 xgboost，请先 pip install xgboost") from e
import joblib
TRAINING_FOLDER_PATH = r"J:\白酒数据集\train1"
TEST_FOLDER_PATH     = r"J:\白酒数据集\test1"
OUTPUT_FOLDER_PATH   = r"J:\白酒数据集\result_XGB概率校正"
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
ENABLE_PROBA_CALIBRATION = True
CALIBRATION_METHOD = "temperature"
CALIBRATION_CV_SPLITS = OPTUNA_CV_SPLITS
CALIBRATION_EPS = 1e-12
FIG_DPI = 300
SAVE_PDF = True
SAVE_PNG = True
ENABLE_ROC = True
ENABLE_SHAP = True
SHAP_MAX_SAMPLES = 1000
SHAP_MAX_DISPLAY = 20
NON_FEATURE_COLS = ["label", "sample_id", "group_id"]
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
model_save_path = output_dir / "xgb_peakarea.pkl"
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
def tic_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (X.sum(axis=1, keepdims=True) + 1e-10)
def infer_mz_step_and_mz0_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
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
def mz_to_grid_idx(mz: np.ndarray, mz0: float, mz_step: float) -> np.ndarray:
    return np.rint((mz - float(mz0)) / float(mz_step)).astype(int)
def convert_peak_params_da_to_bins(mz_step_da: float):
    if mz_step_da is None or mz_step_da <= 0:
        raise RuntimeError("mz_step_da 无效")
    sigma_bins = max(0.5, float(GAUSSIAN_SIGMA_DA) / float(mz_step_da))
    min_width_bins = max(1, int(np.ceil(float(PEAK_MIN_WIDTH_DA) / float(mz_step_da))))
    return float(sigma_bins), int(min_width_bins)
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
    meta = df[NON_FEATURE_COLS].copy()
    data_df = df.drop(columns=NON_FEATURE_COLS, errors="ignore").copy()
    cols, keep = [], []
    for c in data_df.columns:
        try:
            cols.append(int(c))
            keep.append(True)
        except Exception:
            keep.append(False)
    data_df = data_df.loc[:, keep]
    data_df.columns = cols
    target_cols = list(map(int, feature_idx_all))
    df_aligned = data_df.reindex(columns=target_cols, fill_value=0.0)
    return df_aligned.values.astype(np.float32), meta
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
def _score_func(y_true, y_pred, scoring="accuracy"):
    if scoring == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    return accuracy_score(y_true, y_pred)
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
def _temp_scale_probs(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, eps, 1.0)
    logp = np.log(p)
    z = logp / float(T)
    z = z - logsumexp(z, axis=1, keepdims=True)
    out = np.exp(z)
    out = out / (out.sum(axis=1, keepdims=True) + eps)
    return out
def fit_temperature_scaler(oof_probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12):
    P = np.asarray(oof_probs, dtype=float)
    y = np.asarray(y_true, dtype=int).ravel()
    n, c = P.shape
    if n != len(y):
        raise ValueError("oof_probs 与 y_true 长度不一致")
    P = np.clip(P, eps, 1.0)
    P = P / (P.sum(axis=1, keepdims=True) + eps)
    def nll(logT_arr):
        T = float(np.exp(logT_arr[0]))
        PT = _temp_scale_probs(P, T=T, eps=eps)
        return float(-np.mean(np.log(PT[np.arange(n), y] + eps)))
    res = minimize(
        nll,
        x0=np.array([0.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0)],
    )
    T = float(np.exp(res.x[0]))
    info = {
        "method": "temperature",
        "T": T,
        "eps": float(eps),
        "nll": float(res.fun),
        "success": bool(getattr(res, "success", True)),
        "message": str(getattr(res, "message", "")),
    }
    return info
def apply_calibrator(probs: np.ndarray, calibrator) -> np.ndarray:
    if calibrator is None:
        return np.asarray(probs, dtype=float)
    method = str(calibrator.get("method", "temperature")).lower()
    eps = float(calibrator.get("eps", 1e-12))
    if method == "temperature":
        T = float(calibrator.get("T", 1.0))
        T = max(T, 1e-6)
        return _temp_scale_probs(probs, T=T, eps=eps)
    return np.asarray(probs, dtype=float)
def compute_oof_probs_xgb_for_calibration(
    X_raw_full: np.ndarray,
    y_full: np.ndarray,
    g_full: np.ndarray,
    mz_step: float,
    id_map: dict,
    xgb_params: dict,
    n_classes: int,
    n_splits: int = 5,
    rng: int = 42,
):
    y_full = np.asarray(y_full, dtype=int).ravel()
    g_full = np.asarray(g_full, dtype=str).ravel()
    n = len(y_full)
    cv_name, cv, real_splits, min_g = make_group_cv(y_full, g_full, target_splits=int(n_splits), rng=rng)
    if cv_name == "StratifiedKFold_sample_level":
        splits = list(cv.split(np.zeros_like(y_full), y_full))
    else:
        splits = list(cv.split(np.zeros_like(y_full), y_full, g_full))
    oof = np.zeros((n, int(n_classes)), dtype=float)
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        Xtr_raw, ytr = X_raw_full[tr_idx], y_full[tr_idx]
        Xva_raw, yva = X_raw_full[va_idx], y_full[va_idx]
        Xtr_tic = tic_normalize(Xtr_raw)
        Xva_tic = tic_normalize(Xva_raw)
        if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
            windows = find_peak_windows_union_by_class(Xtr_tic, ytr, mz_step_da=mz_step, id_map=id_map)
        else:
            windows = find_global_peak_windows_from_train_mean(Xtr_tic, mz_step_da=mz_step)
        Xtr = transform_to_peak_areas(Xtr_tic, windows, mz_step_da=mz_step)
        Xva = transform_to_peak_areas(Xva_tic, windows, mz_step_da=mz_step)
        if USE_FEATURE_SELECTION:
            k = min(int(TOP_K_FEATURES), Xtr.shape[1])
            if k < Xtr.shape[1]:
                vars_ = np.var(Xtr, axis=0)
                top_idx = np.argsort(vars_)[::-1][:k]
                top_idx.sort()
                Xtr = Xtr[:, top_idx]
                Xva = Xva[:, top_idx]
        try:
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xva)
            p_full = np.zeros((p.shape[0], int(n_classes)), dtype=float)
            for j, cls in enumerate(getattr(clf, "classes_", np.arange(p.shape[1]))):
                p_full[:, int(cls)] = p[:, j]
            p_full = p_full / (p_full.sum(axis=1, keepdims=True) + CALIBRATION_EPS)
            oof[va_idx] = p_full
            print(f"[INFO] Calib OOF fold {fold_i+1}/{len(splits)} ok | va={len(va_idx)}")
        except Exception as e:
            oof[va_idx] = 1.0 / float(n_classes)
            print(f"[WARN] Calib OOF fold {fold_i+1}/{len(splits)} failed: {e}")
    bad = np.where(oof.sum(axis=1) <= 0)[0]
    if bad.size:
        oof[bad] = 1.0 / float(n_classes)
    oof = oof / (oof.sum(axis=1, keepdims=True) + CALIBRATION_EPS)
    return oof
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
def find_peak_windows_union_by_class(X_train_tic: np.ndarray, y_train: np.ndarray, mz_step_da: float, id_map: dict):
    X_train_tic = np.asarray(X_train_tic, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int).ravel()
    all_windows = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        if idx.size < int(PEAK_MIN_SAMPLES_PER_CLASS):
            continue
        mean_spec = X_train_tic[idx].mean(axis=0)
        ws = _peak_windows_from_mean_spectrum(
            mean_spec, mz_step_da=mz_step_da, max_peaks=int(PEAK_MAX_PEAKS_PER_CLASS)
        )
        ws = _merge_windows(ws, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
        all_windows.extend(ws)
    if not all_windows:
        raise RuntimeError("每类找峰后窗口为空：请降低 PEAK_RELATIVE_HEIGHT/PEAK_MIN_WIDTH_DA 或降低 PEAK_MIN_SAMPLES_PER_CLASS。")
    all_windows = _merge_windows(all_windows, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
    if UNION_MAX_WINDOWS and UNION_MAX_WINDOWS > 0 and len(all_windows) > int(UNION_MAX_WINDOWS):
        all_windows_sorted = sorted(all_windows, key=lambda w: float(w.get("prom", 0.0)), reverse=True)[: int(UNION_MAX_WINDOWS)]
        all_windows = _merge_windows(all_windows_sorted, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
    windows = [{"l": int(w["l"]), "r": int(w["r"]), "apex": int(w["apex"])} for w in all_windows]
    return windows
def find_global_peak_windows_from_train_mean(X_train_tic: np.ndarray, mz_step_da: float):
    mean_spec = X_train_tic.mean(axis=0)
    ws = _peak_windows_from_mean_spectrum(mean_spec, mz_step_da=mz_step_da, max_peaks=0)
    ws = _merge_windows(ws, merge_overlap=MERGE_OVERLAPPED_WINDOWS)
    windows = [{"l": int(w["l"]), "r": int(w["r"]), "apex": int(w["apex"])} for w in ws]
    return windows
def transform_to_peak_areas(X_tic: np.ndarray, windows: list, mz_step_da: float):
    sigma_bins, _ = convert_peak_params_da_to_bins(mz_step_da)
    X_smooth = gaussian_filter1d(X_tic, sigma=float(sigma_bins), axis=1, mode="nearest")
    n = X_smooth.shape[0]
    p = len(windows)
    Xp = np.zeros((n, p), dtype=np.float32)
    for j, w in enumerate(windows):
        l, r = int(w["l"]), int(w["r"])
        Xp[:, j] = X_smooth[:, l:r+1].sum(axis=1)
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
def plot_shap_summary_bar(shap_abs_2d, feature_names, outpath_no_ext):
    try:
        import shap
    except Exception:
        print("[WARN] shap 未安装，跳过 SHAP。")
        return
    shap.summary_plot(
        shap_abs_2d,
        features=None,
        feature_names=feature_names,
        plot_type="bar",
        max_display=min(int(SHAP_MAX_DISPLAY), len(feature_names)),
        show=False
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("SHAP summary (bar) - XGB peak area", pad=4, fontweight="bold")
    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(str(outpath_no_ext) + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(str(outpath_no_ext) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
def _shap_to_class_list(shap_vals, n_classes=None):
    if hasattr(shap_vals, "values"):
        sv = shap_vals.values
    else:
        sv = shap_vals
    if isinstance(sv, list):
        return [np.asarray(v) for v in sv]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        if n_classes is not None:
            if sv.shape[2] == int(n_classes):
                return [sv[:, :, i] for i in range(int(n_classes))]
            if sv.shape[1] == int(n_classes):
                return [sv[:, i, :] for i in range(int(n_classes))]
        return [sv[:, :, i] for i in range(sv.shape[2])]
    if sv.ndim == 2:
        return [sv]
    return None
def _shap_for_dot_from_class_list(sv_list, y_pred):
    sv_list = [np.asarray(v) for v in (sv_list or [])]
    if not sv_list:
        return None
    if len(sv_list) == 1:
        return sv_list[0]
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    n = sv_list[0].shape[0]
    f = sv_list[0].shape[1]
    out = np.zeros((n, f), dtype=float)
    for c, sv in enumerate(sv_list):
        idx = np.where(y_pred == c)[0]
        if idx.size > 0:
            out[idx] = sv[idx]
    return out
def plot_shap_mean_abs_stacked_bar(shap_values_list, feature_names, class_names, outpath_no_ext, double_col=True):
    sv_list = [np.asarray(v) for v in shap_values_list]
    C = len(sv_list)
    if C <= 1:
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
def plot_shap_summary_dot(shap_2d, X_plot, feature_names, outpath_no_ext):
    try:
        import shap
    except Exception:
        print("[WARN] shap 未安装，跳过 SHAP。")
        return
    shap.summary_plot(
        shap_2d,
        X_plot,
        feature_names=feature_names,
        plot_type="dot",
        max_display=min(int(SHAP_MAX_DISPLAY), len(feature_names)),
        show=False
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("SHAP summary (dot) - XGB peak area", pad=4, fontweight="bold")
    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(str(outpath_no_ext) + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(str(outpath_no_ext) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
def prepare_cv_folds_peak_features(X_raw_full, y_full, g_full, mz_step, id_map, n_splits=5, rng=42):
    cv_name, cv, real_splits, min_g = make_group_cv(y_full, g_full, target_splits=n_splits, rng=rng)
    if cv_name == "StratifiedKFold_sample_level":
        splits = list(cv.split(X_raw_full, y_full))
    else:
        splits = list(cv.split(X_raw_full, y_full, g_full))
    folds = []
    for fold_i, (tr_idx, va_idx) in enumerate(splits):
        Xtr_raw, ytr = X_raw_full[tr_idx], y_full[tr_idx]
        Xva_raw, yva = X_raw_full[va_idx], y_full[va_idx]
        Xtr_tic = tic_normalize(Xtr_raw)
        Xva_tic = tic_normalize(Xva_raw)
        if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
            windows = find_peak_windows_union_by_class(Xtr_tic, ytr, mz_step_da=mz_step, id_map=id_map)
        else:
            windows = find_global_peak_windows_from_train_mean(Xtr_tic, mz_step_da=mz_step)
        Xtr = transform_to_peak_areas(Xtr_tic, windows, mz_step_da=mz_step)
        Xva = transform_to_peak_areas(Xva_tic, windows, mz_step_da=mz_step)
        if USE_FEATURE_SELECTION:
            k = min(int(TOP_K_FEATURES), Xtr.shape[1])
            if k < Xtr.shape[1]:
                vars_ = np.var(Xtr, axis=0)
                top_idx = np.argsort(vars_)[::-1][:k]
                top_idx.sort()
                Xtr = Xtr[:, top_idx]
                Xva = Xva[:, top_idx]
        folds.append({"fold": fold_i, "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva})
    print(f"[INFO] CV folds prepared: {len(folds)} folds (peak windows+TopK fitted inside fold)")
    return folds
def optuna_tune_xgb_on_folds(folds, n_classes, scoring="accuracy", n_trials=50, timeout=None, rng=42):
    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": int(n_classes),
            "tree_method": "hist",
            "random_state": int(rng),
            "n_jobs": -1,
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e2, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e2, log=True),
        }
        scores = []
        for fd in folds:
            Xtr, ytr = fd["Xtr"], fd["ytr"]
            Xva, yva = fd["Xva"], fd["yva"]
            clf = xgb.XGBClassifier(**params)
            try:
                clf.fit(Xtr, ytr)
                pred = clf.predict(Xva)
                scores.append(float(_score_func(yva, pred, scoring=scoring)))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0
    sampler = optuna.samplers.TPESampler(seed=int(rng))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials), timeout=timeout)
    best = dict(study.best_params)
    best["best_score"] = float(study.best_value)
    return best, study
def main():
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
    feature_idx_all = list(range(idx_min, idx_max + 1))
    print(f"[INFO] master 特征数: {len(feature_idx_all)}")
    train_dfs = []
    for p in csv_files:
        prefix = p.stem.split("_")[0]
        part = load_and_transpose_csv_grid(p, label_map[prefix], mz0=mz0, mz_step=mz_step)
        if part is not None:
            train_dfs.append(part)
    if not train_dfs:
        sys.exit("[ERROR] 无有效训练数据")
    df_train_raw = pd.concat(train_dfs, ignore_index=True)
    X_train_raw_full, meta_train = get_aligned_data(df_train_raw, feature_idx_all)
    y_train_full = meta_train["label"].values.astype(int)
    g_train_full = meta_train["group_id"].astype(str).values
    print("[INFO] Train 每类样本/组统计（n_groups 最小的最危险）:")
    print(class_group_stats(y_train_full, g_train_full, id_map).to_string(index=False))
    if ENABLE_OPTUNA:
        folds = prepare_cv_folds_peak_features(
            X_raw_full=X_train_raw_full,
            y_full=y_train_full,
            g_full=g_train_full,
            mz_step=mz_step,
            id_map=id_map,
            n_splits=int(OPTUNA_CV_SPLITS),
            rng=RNG
        )
        best_params, study = optuna_tune_xgb_on_folds(
            folds=folds,
            n_classes=n_classes,
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
    else:
        best_params = {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 800,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "best_score": None
        }
        print("[WARN] 未启用 Optuna，使用默认:", best_params)
    calibrator = None
    if ENABLE_PROBA_CALIBRATION and str(CALIBRATION_METHOD).lower() == "temperature":
        try:
            xgb_params_calib = dict(best_params)
            xgb_params_calib.update({
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": int(n_classes),
                "tree_method": "hist",
                "random_state": int(RNG),
                "n_jobs": -1,
            })
            xgb_params_calib.pop("best_score", None)
            print("[INFO] 开始概率校准：计算 OOF probabilities ...")
            oof_probs = compute_oof_probs_xgb_for_calibration(
                X_raw_full=X_train_raw_full,
                y_full=y_train_full,
                g_full=g_train_full,
                mz_step=mz_step,
                id_map=id_map,
                xgb_params=xgb_params_calib,
                n_classes=n_classes,
                n_splits=int(CALIBRATION_CV_SPLITS),
                rng=RNG,
            )
            calibrator = fit_temperature_scaler(oof_probs, y_train_full, eps=float(CALIBRATION_EPS))
            print(f"[INFO] 概率校准完成：T={calibrator['T']:.4f}, NLL={calibrator['nll']:.6f}, success={calibrator['success']}")
        except Exception as e:
            calibrator = None
            print(f"[WARN] 概率校准失败(可忽略，将使用原始概率): {e}")
    print("[INFO] 用全量 Train 拟合峰窗口并生成最终特征...")
    X_train_tic_full = tic_normalize(X_train_raw_full)
    if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
        windows = find_peak_windows_union_by_class(X_train_tic_full, y_train_full, mz_step_da=mz_step, id_map=id_map)
    else:
        windows = find_global_peak_windows_from_train_mean(X_train_tic_full, mz_step_da=mz_step)
    peak_feature_names = windows_to_feature_names(windows, mz0=mz0, mz_step=mz_step)
    X_train_peak = transform_to_peak_areas(X_train_tic_full, windows, mz_step_da=mz_step)
    top_idx = None
    final_names = peak_feature_names
    if USE_FEATURE_SELECTION:
        k = min(int(TOP_K_FEATURES), X_train_peak.shape[1])
        vars_ = np.var(X_train_peak, axis=0)
        top_idx = np.argsort(vars_)[::-1][:k]
        top_idx.sort()
        X_train_peak = X_train_peak[:, top_idx]
        final_names = [peak_feature_names[i] for i in top_idx]
    with open(features_save_path, "w", encoding="utf-8") as f:
        json.dump(peak_feature_names, f, ensure_ascii=False, indent=2)
    xgb_params_final = dict(best_params)
    xgb_params_final.update({
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": int(n_classes),
        "tree_method": "hist",
        "random_state": int(RNG),
        "n_jobs": -1,
    })
    xgb_params_final.pop("best_score", None)
    print(f"[INFO] 训练最终 XGBoost（params={xgb_params_final}）...")
    model = xgb.XGBClassifier(**xgb_params_final)
    model.fit(X_train_peak, y_train_full)
    test_dir = Path(TEST_FOLDER_PATH)
    if not test_dir.is_dir():
        sys.exit(f"[ERROR] 测试目录不存在: {test_dir}")
    test_dfs = []
    for p in sorted(test_dir.glob("*.csv")):
        prefix = p.stem.split("_")[0]
        if prefix in label_map:
            part = load_and_transpose_csv_grid(p, label_map[prefix], mz0=mz0, mz_step=mz_step)
            if part is not None:
                test_dfs.append(part)
        else:
            print(f"[WARN] 跳过: {p.name} (前缀 '{prefix}' 不在训练类别中)")
    if not test_dfs:
        sys.exit("[ERROR] 未读取到任何测试集数据")
    df_test_raw = pd.concat(test_dfs, ignore_index=True)
    X_test_raw, meta_test = get_aligned_data(df_test_raw, feature_idx_all)
    y_test = meta_test["label"].values.astype(int)
    g_test = meta_test["group_id"].astype(str).values
    s_test = meta_test["sample_id"].astype(str).values
    X_test_tic = tic_normalize(X_test_raw)
    X_test_peak = transform_to_peak_areas(X_test_tic, windows, mz_step_da=mz_step)
    if top_idx is not None:
        X_test_peak = X_test_peak[:, top_idx]
    print(f"[INFO] 最终维度: {X_train_peak.shape[1]} | Train={len(X_train_peak)} | Test={len(X_test_peak)}")
    names = [id_map[i] for i in range(n_classes)]
    labels_all = list(range(n_classes))
    preds = model.predict(X_test_peak)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] Test Acc: {acc:.4f}")
    print(classification_report(y_test, preds, labels=labels_all, target_names=names, zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=labels_all)
    plot_confusion_matrix(cm, names, "Confusion matrix (count)", fig_dir / "fig_01_confusion_matrix", normalize=None)
    plot_confusion_matrix(cm, names, "Confusion matrix (normalized)", fig_dir / "fig_02_confusion_matrix_norm", normalize="true")
    probs_raw = model.predict_proba(X_test_peak)
    probs = apply_calibrator(probs_raw, calibrator)
    max_p_raw = probs_raw.max(axis=1)
    max_p = probs.max(axis=1)
    if ENABLE_ROC:
        try:
            plot_multiclass_roc_ovr(
                y_true=y_test,
                proba=probs,
                class_names=names,
                outpath_no_ext=fig_dir / "fig_03_roc_ovr_proba",
                title_suffix="XGBoost probabilities"
            )
            print("[INFO] ROC 已输出到 figures/。")
        except Exception as e:
            print(f"[WARN] ROC 绘制失败(可忽略): {e}")
    if ENABLE_SHAP:
        try:
            import shap
            print("[INFO] 计算 SHAP（TreeExplainer）...")
            plot_n = min(int(SHAP_MAX_SAMPLES), X_test_peak.shape[0])
            X_plot = X_test_peak[:plot_n]
            y_pred_plot = model.predict(X_plot)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_plot)
            sv_list = _shap_to_class_list(shap_vals, n_classes=n_classes)
            if sv_list is not None and len(sv_list) > 1:
                plot_shap_mean_abs_stacked_bar(
                    shap_values_list=sv_list,
                    feature_names=final_names,
                    class_names=names,
                    outpath_no_ext=fig_dir / "fig_04_shap_stacked_bar"
                )
            else:
                print("[WARN] 当前 SHAP 输出不是多分类可堆叠形式，跳过堆叠 bar。")
            if sv_list is not None:
                shap_abs = np.mean([np.abs(v) for v in sv_list], axis=0)
            else:
                sv = np.asarray(shap_vals)
                if sv.ndim == 2:
                    shap_abs = np.abs(sv)
                elif sv.ndim == 3:
                    shap_abs = np.mean(np.abs(sv), axis=2)
                else:
                    raise RuntimeError("SHAP 输出维度异常，无法生成 bar。")
            plot_shap_summary_bar(shap_abs, final_names, fig_dir / "fig_05_shap_summary_bar")
            if sv_list is None:
                sv = np.asarray(shap_vals)
                if sv.ndim == 2:
                    shap_dot = sv
                elif sv.ndim == 3:
                    if sv.shape[-1] == n_classes:
                        shap_dot = np.zeros((sv.shape[0], sv.shape[1]), dtype=float)
                        for c in range(n_classes):
                            idx = np.where(y_pred_plot == c)[0]
                            if idx.size:
                                shap_dot[idx] = sv[idx, :, c]
                    elif sv.shape[1] == n_classes:
                        shap_dot = np.zeros((sv.shape[0], sv.shape[2]), dtype=float)
                        for c in range(n_classes):
                            idx = np.where(y_pred_plot == c)[0]
                            if idx.size:
                                shap_dot[idx] = sv[idx, c, :]
                    else:
                        raise RuntimeError("SHAP 3D 输出无法识别 class 轴。")
                else:
                    raise RuntimeError("SHAP 输出维度异常，无法生成 dot。")
            else:
                shap_dot = _shap_for_dot_from_class_list(sv_list, y_pred_plot)
            plot_shap_summary_dot(shap_dot, X_plot, final_names, fig_dir / "fig_06_shap_summary_dot")
            print("[INFO] SHAP 图已输出到 figures/。")
        except Exception as e:
            print(f"[WARN] SHAP 计算/绘图失败(可忽略): {e}")
    out = pd.DataFrame({
        "Sample": np.asarray(s_test, dtype=str),
        "Group": np.asarray(g_test, dtype=str),
        "True_Label": [id_map[int(i)] for i in y_test],
        "Pred_Label": [id_map[int(i)] for i in preds],
        "Similarity_raw": max_p_raw,
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
            "xgb_model": model,
            "id_map": id_map,
            "best_params": best_params,
            "mz0": mz0,
            "mz_step": mz_step,
            "feature_type": "peak_area_per_class_union" if str(PEAK_WINDOWS_MODE).lower() == "per_class_union" else "peak_area_global_mean",
            "windows": windows,
            "peak_feature_names": peak_feature_names,
            "top_idx": top_idx,
            "calibrator": calibrator,
            "unknown_threshold": float(UNKNOWN_THRESHOLD),
            "group_unknown_threshold": float(GROUP_UNKNOWN_THRESHOLD),
        },
        model_save_path
    )
    print(f"[INFO] 模型已保存: {model_save_path}")
    print("[INFO] 全部完成。")
if __name__ == "__main__":
    main()