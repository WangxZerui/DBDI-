import sys
import json
from pathlib import Path
import time
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
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
except Exception as e:
    raise RuntimeError("缺少 scipy，请先 pip install scipy") from e
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
TRAINING_FOLDER_PATH = r"J:\白酒数据集\train1"
TEST_FOLDER_PATH     = r"J:\白酒数据集\test1"
OUTPUT_FOLDER_PATH   = r"J:\白酒数据集\result_SVM概率校正"
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
PEAK_MIN_SAMPLES_PER_CLASS = 5
PEAK_MAX_PEAKS_PER_CLASS = 200
UNION_MAX_WINDOWS = 1000
MERGE_OVERLAPPED_WINDOWS = True
RE_NORMALIZE_PEAK_AREAS = True
USE_FEATURE_SELECTION = True
TOP_K_FEATURES = 600
ENABLE_OPTUNA = True
OPTUNA_N_TRIALS = 50
OPTUNA_SCORING = "accuracy"
OPTUNA_CV_SPLITS = 5
OPTUNA_TIMEOUT = 3600
C_MIN, C_MAX = 1e-3, 1e3
TOL_MIN, TOL_MAX = 1e-5, 1e-3
CLASS_WEIGHT_CHOICES = [None, "balanced"]
LINEAR_MAX_ITER = 10000
CALIBRATE_PROBA = True
CALIBRATION_METHOD = "sigmoid"
CALIB_MAX_SPLITS = 5
FIG_DPI = 300
SAVE_PDF = True
SAVE_PNG = True
ENABLE_ROC = True
ENABLE_SHAP = True
SHAP_BACKGROUND_SIZE = 100
SHAP_MAX_DISPLAY = 20
NON_FEATURE_COLS = ["label", "sample_id", "group_id"]
MM_TO_INCH = 1.0 / 25.4
ELSEVIER_SINGLE_COL_MM = 90
ELSEVIER_DOUBLE_COL_MM = 190
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
ENABLE_LEARNING_CURVE = True
LEARNING_CURVE_SIZES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_CURVE_SCORING = "accuracy"
LEARNING_CURVE_RANDOM_REPEATS = 1
def mz_label(x, decimals=1) -> str:
    return f"{float(x):.{int(decimals)}f}"
def _save_fig(fig, outpath_no_ext: Path):
    fig.tight_layout()
    if SAVE_PDF:
        fig.savefig(str(outpath_no_ext) + ".pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(str(outpath_no_ext) + ".png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
def mm_to_in(mm: float) -> float:
    return float(mm) * MM_TO_INCH
def journal_figsize(width_mm: float, aspect: float = 0.75):
    w = mm_to_in(width_mm)
    return (w, w * float(aspect))
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
    ax.set_title("SHAP summary (dot) - SVM peak area", pad=4, fontweight="bold")
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
def tic_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return X / sums
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
    if mz_step_da <= 0:
        return 1.0, 1
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
        return None
    df["m/z"] = pd.to_numeric(df["m/z"], errors="coerce")
    df = df.dropna(subset=["m/z"])
    df = df[(df["m/z"] >= MZ_MIN - MZ_EPS) & (df["m/z"] <= MZ_MAX + MZ_EPS)]
    if df.empty:
        return None
    mz = df["m/z"].values.astype(float)
    idx = mz_to_grid_idx(mz, mz0=mz0, mz_step=mz_step)
    int_cols = [c for c in df.columns if c != "m/z"]
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
    data_cols = []
    valid_cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        try:
            col_int = int(c)
            data_cols.append(col_int)
            valid_cols.append(c)
        except Exception:
            pass
    data_df = df[valid_cols].copy()
    data_df.columns = data_cols
    df_aligned = data_df.reindex(columns=feature_idx_all, fill_value=0.0)
    return df_aligned.values.astype(np.float32), meta
def make_group_cv(y, groups, target_splits=5, rng=42):
    y = np.array(y)
    groups = np.array(groups)
    min_g = np.inf
    for c in np.unique(y):
        g_in_c = np.unique(groups[y == c])
        min_g = min(min_g, len(g_in_c))
    n_splits = max(2, min(int(target_splits), int(min_g))) if min_g > 0 else 2
    if min_g >= 2:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(rng))
            next(cv.split(np.zeros(len(y)), y, groups))
            return cv, n_splits, "StratifiedGroupKFold"
        except Exception:
            cv = GroupKFold(n_splits=n_splits)
            return cv, n_splits, "GroupKFold"
    print(f"[WARN] 组数过少 (min_groups={min_g})，退化为样本级 StratifiedKFold")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(rng))
    return cv, 5, "StratifiedKFold"
def _peak_windows_from_spec(spec: np.ndarray, mz_step_da: float, max_peaks=0):
    sigma_bins, min_width_bins = convert_peak_params_da_to_bins(mz_step_da)
    spec_smooth = gaussian_filter1d(spec, sigma=float(sigma_bins), mode="nearest")
    prom = max(float(PEAK_MIN_INTENSITY), float(PEAK_RELATIVE_HEIGHT) * float(spec_smooth.max()))
    peaks, props = find_peaks(spec_smooth, prominence=prom, width=int(min_width_bins))
    if len(peaks) == 0:
        return []
    left = np.floor(props["left_ips"]).astype(int)
    right = np.ceil(props["right_ips"]).astype(int)
    prominences = props["prominences"]
    if max_peaks > 0 and len(peaks) > max_peaks:
        top_idx = np.argsort(prominences)[::-1][:int(max_peaks)]
        peaks = peaks[top_idx]
        left = left[top_idx]
        right = right[top_idx]
        prominences = prominences[top_idx]
    windows = []
    for p, l, r, pr in zip(peaks, left, right, prominences):
        windows.append({
            "l": max(0, int(l)),
            "r": min(len(spec) - 1, int(r)),
            "apex": int(p),
            "prom": float(pr)
        })
    return windows
def _merge_windows(windows):
    if not windows:
        return []
    ws = sorted(windows, key=lambda x: x["l"])
    merged = [ws[0].copy()]
    for current in ws[1:]:
        last = merged[-1]
        if MERGE_OVERLAPPED_WINDOWS and current["l"] <= last["r"]:
            last["r"] = max(last["r"], current["r"])
            if current["prom"] > last["prom"]:
                last["apex"] = current["apex"]
                last["prom"] = current["prom"]
            last["l"] = min(last["l"], current["l"])
        else:
            merged.append(current.copy())
    return merged
def find_peak_windows(X_tic, y, mz_step_da, mode="per_class_union"):
    all_windows = []
    if mode == "per_class_union":
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) < int(PEAK_MIN_SAMPLES_PER_CLASS):
                continue
            mean_spec = X_tic[idx].mean(axis=0)
            ws = _peak_windows_from_spec(mean_spec, mz_step_da, int(PEAK_MAX_PEAKS_PER_CLASS))
            all_windows.extend(ws)
    else:
        mean_spec = X_tic.mean(axis=0)
        all_windows = _peak_windows_from_spec(mean_spec, mz_step_da, 0)
    merged = _merge_windows(all_windows)
    if UNION_MAX_WINDOWS > 0 and len(merged) > int(UNION_MAX_WINDOWS):
        merged.sort(key=lambda x: x["prom"], reverse=True)
        merged = merged[:int(UNION_MAX_WINDOWS)]
        merged.sort(key=lambda x: x["l"])
    return merged
def transform_to_peak_areas(X_tic, windows, mz_step_da):
    sigma_bins, _ = convert_peak_params_da_to_bins(mz_step_da)
    X_smooth = gaussian_filter1d(X_tic, sigma=sigma_bins, axis=1, mode="nearest")
    n_samples = X_tic.shape[0]
    n_feats = len(windows)
    X_feats = np.zeros((n_samples, n_feats), dtype=np.float32)
    for i, w in enumerate(windows):
        l, r = int(w["l"]), int(w["r"])
        X_feats[:, i] = X_smooth[:, l:r + 1].sum(axis=1)
    if RE_NORMALIZE_PEAK_AREAS:
        sums = X_feats.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        X_feats /= sums
    return X_feats
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
def prepare_cv_folds(X_raw, y, groups, mz_step, mz0, cv_splits):
    X_tic = tic_normalize(X_raw)
    cv, n_splits, name = make_group_cv(y, groups, target_splits=cv_splits, rng=RNG)
    print(f"[INFO] CV Strategy: {name}, Splits: {n_splits}")
    folds = []
    if "Group" in name:
        splitter = cv.split(X_tic, y, groups)
    else:
        splitter = cv.split(X_tic, y)
    for fold_i, (tr_idx, va_idx) in enumerate(splitter):
        Xtr, ytr = X_tic[tr_idx], y[tr_idx]
        Xva, yva = X_tic[va_idx], y[va_idx]
        windows = find_peak_windows(Xtr, ytr, mz_step, PEAK_WINDOWS_MODE)
        if not windows:
            windows = find_peak_windows(Xtr, ytr, mz_step, "global_mean")
        Xtr_feat = transform_to_peak_areas(Xtr, windows, mz_step)
        Xva_feat = transform_to_peak_areas(Xva, windows, mz_step)
        if USE_FEATURE_SELECTION and Xtr_feat.shape[1] > TOP_K_FEATURES:
            vars_ = np.var(Xtr_feat, axis=0)
            top_idx = np.argsort(vars_)[::-1][:int(TOP_K_FEATURES)]
            top_idx.sort()
            Xtr_feat = Xtr_feat[:, top_idx]
            Xva_feat = Xva_feat[:, top_idx]
        folds.append({
            "Xtr": Xtr_feat, "ytr": ytr,
            "Xva": Xva_feat, "yva": yva,
            "dual": (Xtr_feat.shape[0] <= Xtr_feat.shape[1])
        })
        print(f"  Fold {fold_i + 1}: Features={Xtr_feat.shape[1]}, Train={len(ytr)}, Val={len(yva)}")
    return folds
def run_optuna(folds, n_trials=50):
    def objective(trial):
        C = trial.suggest_float("C", C_MIN, C_MAX, log=True)
        tol = trial.suggest_float("tol", TOL_MIN, TOL_MAX, log=True)
        cw = trial.suggest_categorical("class_weight", CLASS_WEIGHT_CHOICES)
        scores = []
        for f in folds:
            clf = LinearSVC(C=C, tol=tol, class_weight=cw, max_iter=LINEAR_MAX_ITER,
                            dual=f["dual"], random_state=RNG)
            try:
                clf.fit(f["Xtr"], f["ytr"])
                pred = clf.predict(f["Xva"])
                if OPTUNA_SCORING == "balanced_accuracy":
                    s = balanced_accuracy_score(f["yva"], pred)
                else:
                    s = accuracy_score(f["yva"], pred)
                scores.append(s)
            except Exception:
                return 0.0
        return float(np.mean(scores)) if scores else 0.0
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RNG))
    study.optimize(objective, n_trials=int(n_trials), timeout=OPTUNA_TIMEOUT)
    return study.best_params
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
def plot_learning_curve_groupcv(X_raw, y, groups, mz_step, mz0, best_params, outpath):
    X_tic = tic_normalize(X_raw)
    cv, n_splits, name = make_group_cv(y, groups, target_splits=OPTUNA_CV_SPLITS, rng=RNG)
    print(f"[INFO] LearningCurve CV: {name}, Splits: {n_splits}")
    sizes = list(LEARNING_CURVE_SIZES)
    train_scores = np.zeros((len(sizes), n_splits * LEARNING_CURVE_RANDOM_REPEATS), dtype=float)
    val_scores = np.zeros_like(train_scores)
    train_counts = np.zeros_like(train_scores)
    if "Group" in name:
        split_iter = list(cv.split(X_tic, y, groups))
    else:
        split_iter = list(cv.split(X_tic, y))
    col = 0
    for fold_i, (tr_idx_all, va_idx) in enumerate(split_iter):
        Xtr_all, ytr_all, gtr_all = X_tic[tr_idx_all], np.asarray(y)[tr_idx_all], np.asarray(groups)[tr_idx_all]
        Xva, yva = X_tic[va_idx], np.asarray(y)[va_idx]
        for rep in range(int(LEARNING_CURVE_RANDOM_REPEATS)):
            rep_seed = RNG + 1000 * fold_i + 37 * rep
            for si, frac in enumerate(sizes):
                sub_rel_idx = _sample_groups_stratified(ytr_all, gtr_all, frac, rng=rep_seed)
                train_counts[si, col] = len(sub_rel_idx)
                Xtr = Xtr_all[sub_rel_idx]
                ytr = ytr_all[sub_rel_idx]
                windows = find_peak_windows(Xtr, ytr, mz_step, PEAK_WINDOWS_MODE)
                if not windows:
                    windows = find_peak_windows(Xtr, ytr, mz_step, "global_mean")
                Xtr_feat = transform_to_peak_areas(Xtr, windows, mz_step)
                Xva_feat = transform_to_peak_areas(Xva, windows, mz_step)
                if USE_FEATURE_SELECTION and Xtr_feat.shape[1] > TOP_K_FEATURES:
                    vars_ = np.var(Xtr_feat, axis=0)
                    top_idx = np.argsort(vars_)[::-1][:int(TOP_K_FEATURES)]
                    top_idx.sort()
                    Xtr_feat = Xtr_feat[:, top_idx]
                    Xva_feat = Xva_feat[:, top_idx]
                dual_mode = (Xtr_feat.shape[0] <= Xtr_feat.shape[1])
                clf = LinearSVC(
                    C=best_params["C"], tol=best_params["tol"],
                    class_weight=best_params["class_weight"],
                    max_iter=LINEAR_MAX_ITER, dual=dual_mode, random_state=RNG
                )
                clf.fit(Xtr_feat, ytr)
                pred_tr = clf.predict(Xtr_feat)
                pred_va = clf.predict(Xva_feat)
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
    print("[INFO] LearningCurve:")
    for frac, a, b in zip(sizes, tr_mean, va_mean):
        print(f"  frac={frac:.2f}  train={a:.3f}  val={b:.3f}")
def main():
    start_time = time.time()
    train_dir = Path(TRAINING_FOLDER_PATH)
    csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        sys.exit("训练集为空")
    class_names = sorted(list({p.stem.split("_")[0] for p in csv_files}))
    label_map = {name: i for i, name in enumerate(class_names)}
    id_map = {i: name for name, i in label_map.items()}
    print(f"[INFO] Classes: {class_names}")
    mz0, mz_step = infer_mz_step_and_mz0_from_csv(csv_files[0])
    print(f"[INFO] Grid: mz0={mz0:.4f}, step={mz_step:.6f}")
    idx_min = int(np.rint((MZ_MIN - mz0) / mz_step))
    idx_max = int(np.rint((MZ_MAX - mz0) / mz_step))
    feature_idx_all = list(range(idx_min, idx_max + 1))
    print("[INFO] Loading training data...")
    train_dfs = []
    for p in csv_files:
        prefix = p.stem.split("_")[0]
        part = load_and_transpose_csv_grid(p, label_map[prefix], mz0, mz_step)
        if part is not None:
            train_dfs.append(part)
    df_train = pd.concat(train_dfs, ignore_index=True)
    X_train_raw, meta_train = get_aligned_data(df_train, feature_idx_all)
    y_train = meta_train["label"].values.astype(int)
    g_train = meta_train["group_id"].values.astype(str)
    if ENABLE_OPTUNA:
        print("[INFO] Preparing CV folds for Optuna...")
        folds = prepare_cv_folds(X_train_raw, y_train, g_train, mz_step, mz0, OPTUNA_CV_SPLITS)
        print("[INFO] Tuning hyperparameters...")
        best_params = run_optuna(folds, OPTUNA_N_TRIALS)
        print(f"[INFO] Best Params: {best_params}")
    else:
        best_params = {"C": 1.0, "tol": 1e-4, "class_weight": "balanced"}
    if ENABLE_LEARNING_CURVE:
        try:
            print("[INFO] Plotting learning curve...")
            plot_learning_curve_groupcv(
                X_train_raw, y_train, g_train,
                mz_step=mz_step, mz0=mz0,
                best_params=best_params,
                outpath=fig_dir / "fig_06_learning_curve_groupcv"
            )
        except Exception as e:
            print(f"[WARN] Learning curve failed: {e}")
    print("[INFO] Training final model...")
    X_train_tic = tic_normalize(X_train_raw)
    windows = find_peak_windows(X_train_tic, y_train, mz_step, PEAK_WINDOWS_MODE)
    plot_peak_visualization(X_train_tic, y_train, windows, mz0, mz_step, fig_dir / "fig_00_peak_detection_check")
    X_train_feat = transform_to_peak_areas(X_train_tic, windows, mz_step)
    top_idx = None
    final_windows = windows
    if USE_FEATURE_SELECTION and X_train_feat.shape[1] > TOP_K_FEATURES:
        vars_ = np.var(X_train_feat, axis=0)
        top_idx = np.argsort(vars_)[::-1][:int(TOP_K_FEATURES)]
        top_idx.sort()
        X_train_feat = X_train_feat[:, top_idx]
        final_windows = [windows[i] for i in top_idx]
    feat_names = [mz_label(mz0 + int(w["apex"]) * mz_step) for w in final_windows]
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feat_names, f, ensure_ascii=False, indent=2)
    dual_mode = (X_train_feat.shape[0] <= X_train_feat.shape[1])
    base_clf = LinearSVC(
        C=best_params["C"], tol=best_params["tol"],
        class_weight=best_params["class_weight"],
        max_iter=LINEAR_MAX_ITER, dual=dual_mode, random_state=RNG
    )
    base_clf.fit(X_train_feat, y_train)
    calibrated_clf = None
    if CALIBRATE_PROBA:
        print("[INFO] Calibrating probabilities...")
        calib_cv, _, _ = make_group_cv(y_train, g_train, CALIB_MAX_SPLITS, RNG)
        calib_est = LinearSVC(
            C=best_params["C"], tol=best_params["tol"],
            class_weight=best_params["class_weight"],
            max_iter=LINEAR_MAX_ITER, dual=dual_mode, random_state=RNG
        )
        if hasattr(calib_cv, "split"):
            cv_iter = list(calib_cv.split(X_train_feat, y_train, g_train))
        else:
            cv_iter = calib_cv
        calibrated_clf = CalibratedClassifierCV(calib_est, method=CALIBRATION_METHOD, cv=cv_iter)
        calibrated_clf.fit(X_train_feat, y_train)
    print("[INFO] Evaluating on Test set...")
    test_dir = Path(TEST_FOLDER_PATH)
    test_dfs = []
    for p in sorted(test_dir.glob("*.csv")):
        prefix = p.stem.split("_")[0]
        if prefix in label_map:
            part = load_and_transpose_csv_grid(p, label_map[prefix], mz0, mz_step)
            if part is not None:
                test_dfs.append(part)
    if not test_dfs:
        print("[WARN] No test data found.")
        return
    df_test = pd.concat(test_dfs, ignore_index=True)
    X_test_raw, meta_test = get_aligned_data(df_test, feature_idx_all)
    y_test = meta_test["label"].values.astype(int)
    g_test = meta_test["group_id"].values.astype(str)
    s_test = meta_test["sample_id"].values.astype(str)
    X_test_tic = tic_normalize(X_test_raw)
    X_test_feat = transform_to_peak_areas(X_test_tic, windows, mz_step)
    if top_idx is not None:
        X_test_feat = X_test_feat[:, top_idx]
    y_pred = base_clf.predict(X_test_feat)
    probs = None
    max_probs = np.ones(len(y_test), dtype=float)
    if calibrated_clf is not None:
        probs = calibrated_clf.predict_proba(X_test_feat)
        max_probs = probs.max(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    names = [id_map[i] for i in range(len(id_map))]
    labels_all = list(range(len(names)))
    cm = confusion_matrix(y_test, y_pred, labels=labels_all)
    plot_confusion_matrix(cm, names, "Confusion matrix (count)", fig_dir / "fig_01_confusion_matrix", normalize=None)
    plot_confusion_matrix(cm, names, "Confusion matrix (normalized)", fig_dir / "fig_02_confusion_matrix_norm", normalize="true")
    if ENABLE_ROC:
        try:
            if probs is not None:
                plot_multiclass_roc_ovr(
                    y_true=y_test,
                    proba=probs,
                    class_names=names,
                    outpath_no_ext=fig_dir / "fig_03_roc_ovr_proba",
                    title_suffix="SVM probabilities"
                )
            else:
                scores = base_clf.decision_function(X_test_feat)
                if len(names) == 2 and np.ndim(scores) == 1:
                    scores = np.column_stack([-scores, scores])
                plot_multiclass_roc_ovr(
                    y_true=y_test,
                    proba=scores,
                    class_names=names,
                    outpath_no_ext=fig_dir / "fig_03_roc_ovr_score",
                    title_suffix="SVM decision function"
                )
            print("[INFO] ROC 已输出到 figures/。")
        except Exception as e:
            print(f"[WARN] ROC 绘制失败(可忽略): {e}")
    if ENABLE_SHAP:
        try:
            import shap
            import gc
            print("[INFO] 计算 SHAP（LinearExplainer）。")
            class_names_ordered = [id_map[i] for i in range(len(id_map))]
            rs = np.random.RandomState(int(RNG))
            bg_idx = rs.choice(
                X_train_feat.shape[0],
                min(int(SHAP_BACKGROUND_SIZE), X_train_feat.shape[0]),
                replace=False
            )
            explainer = shap.LinearExplainer(base_clf, X_train_feat[bg_idx])
            plot_n = min(500, X_test_feat.shape[0])
            plot_idx = rs.choice(X_test_feat.shape[0], plot_n, replace=False)
            X_plot = X_test_feat[plot_idx]
            y_pred_plot = base_clf.predict(X_plot)
            shap_vals = explainer.shap_values(X_plot)
            sv_list = _shap_to_class_list(shap_vals, n_classes=len(class_names_ordered))
            if sv_list is not None and len(sv_list) > 1:
                plot_shap_mean_abs_stacked_bar(
                    shap_values_list=sv_list,
                    feature_names=feat_names,
                    class_names=class_names_ordered,
                    outpath_no_ext=fig_dir / "fig_04_shap_stacked_bar"
                )
            shap_dot = _shap_for_dot_from_class_list(sv_list, y_pred_plot)
            if shap_dot is None:
                sv = np.asarray(shap_vals)
                if sv.ndim == 2:
                    shap_dot = sv
                else:
                    raise RuntimeError("SHAP 输出形状无法用于 dot 图。")
            plot_shap_summary_dot(
                shap_2d=shap_dot,
                X_plot=X_plot,
                feature_names=feat_names,
                outpath_no_ext=fig_dir / "fig_05_shap_summary_dot"
            )
            print("[INFO] SHAP 图已输出到 figures/。")
            del shap_vals, sv_list, shap_dot, X_plot
            gc.collect()
        except Exception as e:
            print(f"[WARN] SHAP failed: {e}")
    res_df = pd.DataFrame({
        "Sample": s_test,
        "Group": g_test,
        "True": [id_map[int(i)] for i in y_test],
        "Pred": [id_map[int(i)] for i in y_pred],
        "Prob": max_probs,
        "Is_Unknown": max_probs < float(UNKNOWN_THRESHOLD)
    })
    res_df.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump({
        "model": base_clf,
        "calibrator": calibrated_clf,
        "windows": windows,
        "top_idx": top_idx,
        "mz0": float(mz0),
        "mz_step": float(mz_step),
        "idx_min": int(idx_min),
        "n_bins": int(len(feature_idx_all)),
        "mz_params": (float(mz0), float(mz_step)),
        "unknown_threshold": float(UNKNOWN_THRESHOLD),
        "group_unknown_threshold": float(GROUP_UNKNOWN_THRESHOLD),
        "id_map": id_map
    }, output_dir / "model.pkl")
    print(f"[INFO] All Done. Time: {time.time() - start_time:.1f}s")
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
if __name__ == "__main__":
    main()