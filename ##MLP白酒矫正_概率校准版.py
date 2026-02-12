import sys
import json
import warnings
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
TRAINING_FOLDER_PATH = r"J:\白酒数据集\train1"
TEST_FOLDER_PATH     = r"J:\白酒数据集\test1"
OUTPUT_FOLDER_PATH   = r"J:\白酒数据集\result_mpl_new"
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
ENABLE_LEARNING_CURVE = True
LEARNING_CURVE_SIZES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_CURVE_SCORING = "accuracy"
LEARNING_CURVE_RANDOM_REPEATS = 1
FIG_DPI = 300
SAVE_PDF = True
SAVE_PNG = True
ENABLE_ROC = True
ENABLE_SHAP = True
SHAP_MAX_SAMPLES = 300
SHAP_BACKGROUND = 50
SHAP_MAX_EVALS = 2000
SHAP_MAX_DISPLAY = 20
NON_FEATURE_COLS = ["label", "sample_id", "group_id"]
MM_TO_INCH = 1.0 / 25.4
ELSEVIER_SINGLE_COL_MM = 90
ELSEVIER_DOUBLE_COL_MM = 190
def mm_to_in(mm: float) -> float:
    return float(mm) * MM_TO_INCH
def journal_figsize(width_mm: float, aspect: float = 0.75):
    w = mm_to_in(width_mm)
    return (w, w * float(aspect))
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
model_save_path = output_dir / "mlp_peakarea.pkl"
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
    df2 = df2.astype(np.float32, copy=False)
    df_T = df2.T.copy()
    df_T["sample_id"] = df_T.index.astype(str)
    df_T["group_id"] = df_T["sample_id"].str.split("|", n=1).str[0].str.strip()
    df_T["label"] = int(label)
    df_T.reset_index(drop=True, inplace=True)
    return df_T
def get_aligned_data(df: pd.DataFrame, feature_idx_all):
    meta = df[NON_FEATURE_COLS].copy()
    feat_pos = {int(k): i for i, k in enumerate(feature_idx_all)}
    n_samples = int(df.shape[0])
    n_feats = int(len(feature_idx_all))
    X = np.zeros((n_samples, n_feats), dtype=np.float32)
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        try:
            c_int = int(c)
        except Exception:
            continue
        j = feat_pos.get(c_int)
        if j is None:
            continue
        col = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32).values
        X[:, j] = col
    return X, meta
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
def _softmax_2d(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)
def _temp_scale_probs(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    logits = np.log(probs)
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        return probs.astype(np.float32)
    scaled = logits / T
    return _softmax_2d(scaled).astype(np.float32)
def _nll_from_probs(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=int).ravel()
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    p = probs[np.arange(y_true.size), y_true]
    return float(-np.mean(np.log(p + eps)))
def fit_temperature_scaling_from_oof(oof_probs: np.ndarray, y_true: np.ndarray):
    y_true = np.asarray(y_true, dtype=int).ravel()
    oof_probs = np.asarray(oof_probs, dtype=np.float64)
    if oof_probs.ndim != 2 or oof_probs.shape[0] != y_true.size:
        raise ValueError(f"Bad oof_probs shape: {oof_probs.shape}, y_true={y_true.shape}")
    nll_raw = _nll_from_probs(y_true, oof_probs)
    def obj(x):
        logT = float(x[0])
        T = float(np.exp(logT))
        p = _temp_scale_probs(oof_probs, T)
        return _nll_from_probs(y_true, p)
    best = None
    success = False
    try:
        from scipy.optimize import minimize
        res = minimize(
            obj,
            x0=np.array([0.0], dtype=float),
            bounds=[(-4.0, 4.0)],
            method="L-BFGS-B",
            options={"maxiter": 200}
        )
        logT = float(res.x[0])
        T = float(np.exp(logT))
        success = bool(getattr(res, "success", False))
        best = T
    except Exception:
        Ts = np.exp(np.linspace(-2.0, 2.0, 81))
        vals = []
        for T in Ts:
            vals.append(obj([float(np.log(float(T)))]))
        best = float(Ts[int(np.argmin(vals))])
        success = True
    oof_cal = _temp_scale_probs(oof_probs, best)
    nll_cal = _nll_from_probs(y_true, oof_cal)
    return {
        "method": "temperature",
        "T": float(best),
        "nll_raw": float(nll_raw),
        "nll_cal": float(nll_cal),
        "n_samples": int(y_true.size),
        "success": bool(success),
    }
def apply_calibrator(probs: np.ndarray, calibrator):
    if calibrator is None:
        return np.asarray(probs, dtype=np.float32)
    if isinstance(calibrator, dict) and str(calibrator.get("method", "")).lower() == "temperature":
        T = float(calibrator.get("T", 1.0))
        return _temp_scale_probs(probs, T)
    return np.asarray(probs, dtype=np.float32)
def compute_oof_proba_groupcv_mlp(
    X_raw_full: np.ndarray,
    y_full: np.ndarray,
    g_full: np.ndarray,
    mz_step: float,
    best_params: dict,
    n_splits: int = 5,
    rng: int = 42,
):
    y_full = np.asarray(y_full, dtype=int).ravel()
    g_full = np.asarray(g_full, dtype=str).ravel()
    X_raw_full = np.asarray(X_raw_full, dtype=np.float32)
    n_classes = int(len(np.unique(y_full)))
    oof = np.zeros((y_full.size, n_classes), dtype=np.float32)
    filled = np.zeros(y_full.size, dtype=bool)
    cv_name, cv, _, _ = make_group_cv(y_full, g_full, target_splits=int(n_splits), rng=int(rng))
    if cv_name == "StratifiedKFold_sample_level":
        split_iter = list(cv.split(X_raw_full, y_full))
    else:
        split_iter = list(cv.split(X_raw_full, y_full, g_full))
    mlp_params = build_mlp_params_from_best(best_params)
    for fold_i, (tr_idx, va_idx) in enumerate(split_iter):
        Xtr_raw = X_raw_full[tr_idx]
        ytr = y_full[tr_idx]
        Xva_raw = X_raw_full[va_idx]
        Xtr_tic = tic_normalize(Xtr_raw)
        Xva_tic = tic_normalize(Xva_raw)
        if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
            windows = find_peak_windows_union_by_class(Xtr_tic, ytr, mz_step_da=mz_step)
        else:
            windows = find_global_peak_windows_from_train_mean(Xtr_tic, mz_step_da=mz_step)
        Xtr_feat = transform_to_peak_areas(Xtr_tic, windows, mz_step_da=mz_step)
        Xva_feat = transform_to_peak_areas(Xva_tic, windows, mz_step_da=mz_step)
        if USE_FEATURE_SELECTION and Xtr_feat.shape[1] > int(TOP_K_FEATURES):
            k = min(int(TOP_K_FEATURES), Xtr_feat.shape[1])
            vars_ = np.var(Xtr_feat, axis=0)
            top_idx = np.argsort(vars_)[::-1][:k]
            top_idx.sort()
            Xtr_feat = Xtr_feat[:, top_idx]
            Xva_feat = Xva_feat[:, top_idx]
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr_feat)
        Xva_s = scaler.transform(Xva_feat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = MLPClassifier(**mlp_params)
            clf.fit(Xtr_s, ytr)
        pr = clf.predict_proba(Xva_s).astype(np.float32, copy=False)
        oof[va_idx] = pr
        filled[va_idx] = True
    if not bool(filled.all()):
        miss = np.where(~filled)[0]
        if miss.size > 0:
            oof[miss] = (1.0 / float(n_classes))
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
    if max_peaks and max_peaks > 0 and len(peaks) > int(max_peaks):
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
def find_peak_windows_union_by_class(X_train_tic: np.ndarray, y_train: np.ndarray, mz_step_da: float):
    X_train_tic = np.asarray(X_train_tic, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int).ravel()
    all_windows = []
    for c in np.unique(y_train):
        idx = np.where(y_train == c)[0]
        if idx.size < int(PEAK_MIN_SAMPLES_PER_CLASS):
            continue
        mean_spec = X_train_tic[idx].mean(axis=0)
        ws = _peak_windows_from_mean_spectrum(mean_spec, mz_step_da=mz_step_da, max_peaks=int(PEAK_MAX_PEAKS_PER_CLASS))
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
        Xp[:, j] = X_smooth[:, l:r + 1].sum(axis=1)
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
def plot_peak_visualization(X_tic, y, windows, mz0, mz_step, outpath):
    classes = np.unique(y)
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 3 * len(classes)), sharex=True)
    if len(classes) == 1:
        axes = [axes]
    mz_axis = mz0 + np.arange(X_tic.shape[1]) * mz_step
    for ax, c in zip(axes, classes):
        mean_spec = X_tic[y == c].mean(axis=0)
        ax.plot(mz_axis, mean_spec, color="black", linewidth=0.8, label=f"Class {c} Mean")
        for w in windows:
            l_mz = mz0 + int(w["l"]) * mz_step
            r_mz = mz0 + int(w["r"]) * mz_step
            ax.axvspan(l_mz, r_mz, color="red", alpha=0.1)
        ax.set_ylabel("Intensity")
        ax.legend(loc="upper right")
        ax.set_title(f"Class {c} Peak Windows", fontsize=10)
    axes[-1].set_xlabel("m/z")
    _save_fig(fig, outpath)
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
def _shap_to_class_list(shap_vals, n_classes_hint=None, n_features_hint=None):
    if hasattr(shap_vals, "values"):
        shap_vals = shap_vals.values
    if isinstance(shap_vals, list):
        out = [np.asarray(v) for v in shap_vals]
        return out if len(out) > 1 else None
    sv = np.asarray(shap_vals)
    if sv.ndim != 3:
        return None
    n, a, b = sv.shape
    if (n_features_hint is not None and a == int(n_features_hint)) or (n_classes_hint is not None and b == int(n_classes_hint)):
        return [sv[:, :, i] for i in range(b)]
    if (n_classes_hint is not None and a == int(n_classes_hint)) or (n_features_hint is not None and b == int(n_features_hint)):
        return [sv[:, i, :] for i in range(a)]
    return [sv[:, :, i] for i in range(b)]
def _pick_shap_signed_for_pred_class(sv_list, y_pred):
    if sv_list is None or len(sv_list) == 0:
        return None
    if len(sv_list) == 1:
        return np.asarray(sv_list[0], dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    n = y_pred.shape[0]
    p = int(np.asarray(sv_list[0]).shape[1])
    out = np.zeros((n, p), dtype=np.float32)
    C = len(sv_list)
    for i in range(n):
        c = int(y_pred[i])
        c = 0 if c < 0 else (C - 1 if c >= C else c)
        out[i, :] = np.asarray(sv_list[c][i, :], dtype=np.float32)
    return out
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
    _save_fig(fig, outpath_no_ext)
def plot_shap_summary_dot(shap_signed_2d, X_plot, feature_names, outpath_no_ext):
    try:
        import shap
    except Exception:
        print("[WARN] shap 未安装，跳过 SHAP。")
        return
    shap.summary_plot(
        shap_signed_2d,
        X_plot,
        feature_names=feature_names,
        plot_type="dot",
        max_display=min(int(SHAP_MAX_DISPLAY), len(feature_names)),
        show=False
    )
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("SHAP summary (dot) - MLP peak area", pad=4, fontweight="bold")
    _save_fig(fig, outpath_no_ext)
def prepare_cv_folds_peak_features(X_raw_full, y_full, g_full, mz_step, n_splits=5, rng=42):
    cv_name, cv, real_splits, _ = make_group_cv(y_full, g_full, target_splits=n_splits, rng=rng)
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
            windows = find_peak_windows_union_by_class(Xtr_tic, ytr, mz_step_da=mz_step)
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
def optuna_tune_mlp_on_folds(folds, scoring="accuracy", n_trials=50, timeout=None, rng=42):
    def build_hidden_layers(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        units = []
        for i in range(n_layers):
            u = trial.suggest_int(f"units_l{i+1}", 32, 512, log=True)
            units.append(int(u))
        return tuple(units)
    def objective(trial):
        hidden_layer_sizes = build_hidden_layers(trial)
        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
        early_stopping = trial.suggest_categorical("early_stopping", [True, False])
        max_iter = trial.suggest_int("max_iter", 200, 1200)
        n_iter_no_change = trial.suggest_int("n_iter_no_change", 10, 40)
        clf_params = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            alpha=float(alpha),
            batch_size=int(batch_size),
            learning_rate="adaptive",
            learning_rate_init=float(learning_rate_init),
            early_stopping=bool(early_stopping),
            validation_fraction=0.1,
            n_iter_no_change=int(n_iter_no_change),
            max_iter=int(max_iter),
            shuffle=True,
            random_state=int(rng),
            verbose=False,
        )
        scores = []
        for fd in folds:
            Xtr, ytr = fd["Xtr"], fd["ytr"]
            Xva, yva = fd["Xva"], fd["yva"]
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr_s = scaler.fit_transform(Xtr)
            Xva_s = scaler.transform(Xva)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = MLPClassifier(**clf_params)
                try:
                    clf.fit(Xtr_s, ytr)
                    pred = clf.predict(Xva_s)
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
def _sample_groups_stratified(y_tr, g_tr, frac, rng=42):
    y_tr = np.asarray(y_tr)
    g_tr = np.asarray(g_tr)
    uniq_groups = np.unique(g_tr)
    n_total_g = len(uniq_groups)
    n_pick_g = int(np.clip(np.round(frac * n_total_g), 1, n_total_g))
    group_to_cls = {}
    for g in uniq_groups:
        cls = np.bincount(y_tr[g_tr == g]).argmax()
        group_to_cls[g] = int(cls)
    cls_to_groups = {}
    for g, c in group_to_cls.items():
        cls_to_groups.setdefault(c, []).append(g)
    rs = np.random.RandomState(int(rng))
    for c in cls_to_groups:
        rs.shuffle(cls_to_groups[c])
    classes = sorted(cls_to_groups.keys())
    picked = []
    for c in classes:
        if len(picked) >= n_pick_g:
            break
        if len(cls_to_groups[c]) > 0:
            picked.append(cls_to_groups[c].pop())
    if len(picked) < n_pick_g:
        remaining = []
        weights = []
        for c in classes:
            gs = cls_to_groups.get(c, [])
            for g in gs:
                remaining.append(g)
                weights.append(len(gs))
        remaining = np.array(remaining, dtype=object)
        if len(remaining) > 0:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else None
            need = n_pick_g - len(picked)
            if need >= len(remaining):
                picked.extend(list(remaining))
            else:
                chosen = rs.choice(remaining, size=need, replace=False, p=weights)
                picked.extend(list(chosen))
    picked = np.array(picked, dtype=object)
    sel_idx = np.where(np.isin(g_tr, picked))[0]
    return sel_idx
def _build_hidden_from_best(bp):
    n_layers = int(bp.get("n_layers", 1))
    units = []
    for i in range(n_layers):
        units.append(int(bp.get(f"units_l{i+1}", 128)))
    return tuple(units)
def build_mlp_params_from_best(best_params):
    return dict(
        hidden_layer_sizes=_build_hidden_from_best(best_params),
        activation=str(best_params.get("activation", "relu")),
        solver="adam",
        alpha=float(best_params.get("alpha", 1e-4)),
        batch_size=int(best_params.get("batch_size", 128)),
        learning_rate="adaptive",
        learning_rate_init=float(best_params.get("learning_rate_init", 1e-3)),
        early_stopping=bool(best_params.get("early_stopping", True)),
        validation_fraction=0.1,
        n_iter_no_change=int(best_params.get("n_iter_no_change", 20)),
        max_iter=int(best_params.get("max_iter", 600)),
        shuffle=True,
        random_state=int(RNG),
        verbose=False,
    )
def plot_learning_curve_groupcv(X_raw, y, groups, mz_step, best_params, outpath):
    X_tic = tic_normalize(X_raw)
    cv_name, cv, n_splits, _ = make_group_cv(y, groups, target_splits=OPTUNA_CV_SPLITS, rng=RNG)
    print(f"[INFO] LearningCurve CV: {cv_name}, Splits: {n_splits}")
    sizes = list(LEARNING_CURVE_SIZES)
    train_scores = np.zeros((len(sizes), n_splits * LEARNING_CURVE_RANDOM_REPEATS), dtype=float)
    val_scores = np.zeros_like(train_scores)
    train_counts = np.zeros_like(train_scores)
    if cv_name == "StratifiedKFold_sample_level":
        split_iter = list(cv.split(X_tic, y))
    else:
        split_iter = list(cv.split(X_tic, y, groups))
    col = 0
    mlp_params = build_mlp_params_from_best(best_params)
    for fold_i, (tr_idx_all, va_idx) in enumerate(split_iter):
        Xtr_all = X_tic[tr_idx_all]
        ytr_all = np.asarray(y)[tr_idx_all]
        gtr_all = np.asarray(groups)[tr_idx_all]
        Xva_all = X_tic[va_idx]
        yva = np.asarray(y)[va_idx]
        for rep in range(int(LEARNING_CURVE_RANDOM_REPEATS)):
            rep_seed = RNG + 1000 * fold_i + 37 * rep
            for si, frac in enumerate(sizes):
                sub_rel_idx = _sample_groups_stratified(ytr_all, gtr_all, frac, rng=rep_seed)
                train_counts[si, col] = len(sub_rel_idx)
                Xtr = Xtr_all[sub_rel_idx]
                ytr = ytr_all[sub_rel_idx]
                if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
                    windows = find_peak_windows_union_by_class(Xtr, ytr, mz_step_da=mz_step)
                else:
                    windows = find_global_peak_windows_from_train_mean(Xtr, mz_step_da=mz_step)
                Xtr_feat = transform_to_peak_areas(Xtr, windows, mz_step_da=mz_step)
                Xva_feat = transform_to_peak_areas(Xva_all, windows, mz_step_da=mz_step)
                if USE_FEATURE_SELECTION and Xtr_feat.shape[1] > TOP_K_FEATURES:
                    vars_ = np.var(Xtr_feat, axis=0)
                    top_idx = np.argsort(vars_)[::-1][:int(TOP_K_FEATURES)]
                    top_idx.sort()
                    Xtr_feat = Xtr_feat[:, top_idx]
                    Xva_feat = Xva_feat[:, top_idx]
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xtr_s = scaler.fit_transform(Xtr_feat)
                Xva_s = scaler.transform(Xva_feat)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = MLPClassifier(**mlp_params)
                    clf.fit(Xtr_s, ytr)
                pred_tr = clf.predict(Xtr_s)
                pred_va = clf.predict(Xva_s)
                if LEARNING_CURVE_SCORING == "balanced_accuracy":
                    s_tr = balanced_accuracy_score(ytr, pred_tr)
                    s_va = balanced_accuracy_score(yva, pred_va)
                else:
                    s_tr = accuracy_score(ytr, pred_tr)
                    s_va = accuracy_score(yva, pred_va)
                train_scores[si, col] = s_tr
                val_scores[si, col] = s_va
            col += 1
    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1)
    va_std = val_scores.std(axis=1)
    x_mean = train_counts.mean(axis=1)
    width_mm = ELSEVIER_DOUBLE_COL_MM
    fig, ax = plt.subplots(figsize=journal_figsize(width_mm=width_mm, aspect=0.75))
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
    rng = np.random.default_rng(int(RNG))
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
            n_splits=int(OPTUNA_CV_SPLITS),
            rng=RNG
        )
        best_params, study = optuna_tune_mlp_on_folds(
            folds=folds,
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
            "n_layers": 2, "units_l1": 128, "units_l2": 64,
            "activation": "relu",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "batch_size": 128,
            "early_stopping": True,
            "max_iter": 600,
            "n_iter_no_change": 20,
            "best_score": None
        }
        print("[WARN] 未启用 Optuna，使用默认:", best_params)
    proba_calibrator = None
    try:
        print("[CAL] Fitting temperature scaling on OOF probabilities (GroupCV)...")
        oof_probs = compute_oof_proba_groupcv_mlp(
            X_raw_full=X_train_raw_full,
            y_full=y_train_full,
            g_full=g_train_full,
            mz_step=mz_step,
            best_params=best_params,
            n_splits=int(OPTUNA_CV_SPLITS),
            rng=int(RNG),
        )
        proba_calibrator = fit_temperature_scaling_from_oof(oof_probs, y_train_full)
        if isinstance(proba_calibrator, dict):
            print(
                f"[INFO] 概率校准完成：T={float(proba_calibrator.get('T', 1.0)):.4f}, "
                f"NLL_raw={float(proba_calibrator.get('nll_raw', 0.0)):.6f}, "
                f"NLL_cal={float(proba_calibrator.get('nll_cal', 0.0)):.6f}, "
                f"success={bool(proba_calibrator.get('success', False))}"
            )
            try:
                with open(output_dir / "proba_calibrator.json", "w", encoding="utf-8") as f:
                    json.dump(proba_calibrator, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Probability calibration skipped (can ignore): {e}")
        proba_calibrator = None
    print("[INFO] 用全量 Train 拟合峰窗口并生成最终特征...")
    X_train_tic_full = tic_normalize(X_train_raw_full)
    if str(PEAK_WINDOWS_MODE).lower() == "per_class_union":
        windows = find_peak_windows_union_by_class(X_train_tic_full, y_train_full, mz_step_da=mz_step)
    else:
        windows = find_global_peak_windows_from_train_mean(X_train_tic_full, mz_step_da=mz_step)
    plot_peak_visualization(X_train_tic_full, y_train_full, windows, mz0, mz_step, fig_dir / "fig_00_peak_detection_check")
    if ENABLE_LEARNING_CURVE:
        try:
            print("[INFO] Plotting learning curve...")
            plot_learning_curve_groupcv(
                X_train_raw_full, y_train_full, g_train_full,
                mz_step=mz_step,
                best_params=best_params,
                outpath=fig_dir / "fig_06_learning_curve_groupcv"
            )
            print("[INFO] Learning curve 已输出到 figures/。")
        except Exception as e:
            print(f"[WARN] Learning curve failed: {e}")
    peak_feature_names = windows_to_feature_names(windows, mz0=mz0, mz_step=mz_step)
    X_train_peak = transform_to_peak_areas(X_train_tic_full, windows, mz_step_da=mz_step)
    top_idx = None
    final_names = peak_feature_names
    if USE_FEATURE_SELECTION:
        k = min(int(TOP_K_FEATURES), X_train_peak.shape[1])
        if k < X_train_peak.shape[1]:
            vars_ = np.var(X_train_peak, axis=0)
            top_idx = np.argsort(vars_)[::-1][:k]
            top_idx.sort()
            X_train_peak = X_train_peak[:, top_idx]
            final_names = [peak_feature_names[i] for i in top_idx]
    with open(features_save_path, "w", encoding="utf-8") as f:
        json.dump(final_names, f, ensure_ascii=False, indent=2)
    mlp_params_final = build_mlp_params_from_best(best_params)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train_peak)
    print(f"[INFO] 训练最终 MLP（params={mlp_params_final}）...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPClassifier(**mlp_params_final)
        model.fit(X_train_s, y_train_full)
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
    X_test_s = scaler.transform(X_test_peak)
    print(f"[INFO] 最终维度: {X_train_peak.shape[1]} | Train={len(X_train_peak)} | Test={len(X_test_peak)}")
    names = [id_map[i] for i in range(n_classes)]
    labels_all = list(range(n_classes))
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] Test Acc: {acc:.4f}")
    print(classification_report(y_test, preds, labels=labels_all, target_names=names, zero_division=0))
    cm = confusion_matrix(y_test, preds, labels=labels_all)
    plot_confusion_matrix(cm, names, "Confusion matrix (count)", fig_dir / "fig_01_confusion_matrix", normalize=None)
    plot_confusion_matrix(cm, names, "Confusion matrix (normalized)", fig_dir / "fig_02_confusion_matrix_norm", normalize="true")
    probs_raw = model.predict_proba(X_test_s)
    probs = apply_calibrator(probs_raw, proba_calibrator)
    max_p = probs.max(axis=1)
    if ENABLE_ROC:
        try:
            plot_multiclass_roc_ovr(
                y_true=y_test,
                proba=probs,
                class_names=names,
                outpath_no_ext=fig_dir / "fig_03_roc_ovr_proba",
                title_suffix="MLP probabilities"
            )
            print("[INFO] ROC 已输出到 figures/。")
        except Exception as e:
            print(f"[WARN] ROC 绘制失败(可忽略): {e}")
    if ENABLE_SHAP:
        try:
            import shap
            print("[INFO] 计算 SHAP（MLP：PermutationExplainer，已采样控时）...")
            bg_n = min(int(SHAP_BACKGROUND), X_train_s.shape[0])
            bg_idx = rng.choice(X_train_s.shape[0], size=bg_n, replace=False)
            X_bg = X_train_s[bg_idx]
            plot_n = min(int(SHAP_MAX_SAMPLES), X_test_s.shape[0])
            if plot_n < X_test_s.shape[0]:
                idx = rng.choice(X_test_s.shape[0], size=plot_n, replace=False)
                X_plot = X_test_s[idx]
            else:
                X_plot = X_test_s
            try:
                explainer = shap.Explainer(model.predict_proba, X_bg, algorithm="permutation")
                exp = explainer(X_plot, max_evals=int(SHAP_MAX_EVALS))
                shap_vals = exp
            except Exception as e_perm:
                print(f"[WARN] permutation explainer 失败，尝试 KernelExplainer：{e_perm}")
                kexpl = shap.KernelExplainer(model.predict_proba, X_bg)
                shap_vals = kexpl.shap_values(X_plot, nsamples=min(200, int(SHAP_MAX_EVALS)))
            sv_list = _shap_to_class_list(shap_vals, n_classes_hint=n_classes, n_features_hint=len(final_names))
            if sv_list is not None:
                plot_shap_mean_abs_stacked_bar(
                    shap_values_list=sv_list,
                    feature_names=final_names,
                    class_names=names,
                    outpath_no_ext=fig_dir / "fig_04_shap_stacked_bar"
                )
                y_pred_plot = model.predict(X_plot)
                shap_dot_signed = _pick_shap_signed_for_pred_class(sv_list, y_pred_plot)
            else:
                sv = np.asarray(getattr(shap_vals, "values", shap_vals))
                if sv.ndim == 2:
                    shap_dot_signed = sv.astype(np.float32, copy=False)
                elif sv.ndim == 3:
                    y_pred_plot = model.predict(X_plot).astype(int)
                    if sv.shape[1] == len(final_names):
                        C = sv.shape[2]
                        y_pred_plot = np.clip(y_pred_plot, 0, C - 1)
                        shap_dot_signed = np.zeros((sv.shape[0], sv.shape[1]), dtype=np.float32)
                        for i in range(sv.shape[0]):
                            shap_dot_signed[i, :] = sv[i, :, int(y_pred_plot[i])]
                    elif sv.shape[2] == len(final_names):
                        C = sv.shape[1]
                        y_pred_plot = np.clip(y_pred_plot, 0, C - 1)
                        shap_dot_signed = np.zeros((sv.shape[0], sv.shape[2]), dtype=np.float32)
                        for i in range(sv.shape[0]):
                            shap_dot_signed[i, :] = sv[i, int(y_pred_plot[i]), :]
                    else:
                        raise RuntimeError(f"Unexpected SHAP values shape: {sv.shape}")
                else:
                    raise RuntimeError(f"Unexpected SHAP values shape: {sv.shape}")
                print("[WARN] SHAP 输出无法转为多分类列表，跳过堆叠 bar，仅画 summary。")
            plot_shap_summary_dot(shap_dot_signed, X_plot, final_names, fig_dir / "fig_05_shap_summary_dot")
            print("[INFO] SHAP 图已输出到 figures/。")
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
            "mlp_model": model,
            "scaler": scaler,
            "id_map": id_map,
            "best_params": best_params,
            "mz0": mz0,
            "mz_step": mz_step,
            "feature_type": "peak_area_per_class_union" if str(PEAK_WINDOWS_MODE).lower() == "per_class_union" else "peak_area_global_mean",
            "windows": windows,
            "peak_feature_names": peak_feature_names,
            "final_feature_names": final_names,
            "top_idx": top_idx,
"proba_calibrator": proba_calibrator,
"unknown_threshold": float(UNKNOWN_THRESHOLD),
"group_unknown_threshold": float(GROUP_UNKNOWN_THRESHOLD),
},
        model_save_path
    )
    print(f"[INFO] 模型已保存: {model_save_path}")
    print("[INFO] 全部完成。")
if __name__ == "__main__":
    main()