import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
try:
    import pyopenms
except ImportError:
    print("请先安装 pyopenms 库：pip install pyopenms")
    raise SystemExit(1)
def read_scans_from_mzml(mzml_path: Path,
                         ms_levels: Optional[set] = None,
                         start_rt_min: Optional[float] = None,
                         end_rt_min: Optional[float] = None,
                         min_intensity: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
    exp = pyopenms.MSExperiment()
    original_cwd = os.getcwd()
    try:
        os.chdir(mzml_path.parent)
        pyopenms.MzMLFile().load(str(mzml_path.name), exp)
    except Exception as e:
        print(f"[读取错误] 无法加载文件: {mzml_path.name}")
        print(f"  错误详情: {e}")
        os.chdir(original_cwd)
        return []
    finally:
        os.chdir(original_cwd)
    scans = []
    sample = mzml_path.stem
    rt_lo = None if start_rt_min is None else float(start_rt_min) * 60.0
    rt_hi = None if end_rt_min is None else float(end_rt_min) * 60.0
    dropped_count = 0
    for idx, spec in enumerate(exp.getSpectra()):
        ms_level = spec.getMSLevel()
        if ms_levels is not None and ms_level not in ms_levels:
            continue
        rt_sec = float(spec.getRT())
        if (rt_lo is not None and rt_sec < rt_lo) or (rt_hi is not None and rt_sec > rt_hi):
            continue
        mz_array, it_array = spec.get_peaks()
        if mz_array is None or it_array is None or it_array.size == 0:
            continue
        max_val = np.max(it_array)
        if max_val < min_intensity:
            dropped_count += 1
            continue
        if mz_array.size > 1:
            order = np.argsort(mz_array, kind="mergesort")
            mz_array = mz_array[order]
            it_array = it_array[order]
        meta = {
            "sample": sample,
            "native_id": spec.getNativeID() or f"scan{idx + 1}",
            "rt_min": rt_sec / 60.0,
            "ms_level": ms_level,
            "file": str(mzml_path.name),
            "index": idx
        }
        scans.append((mz_array.astype(float), it_array.astype(float), meta))
    return scans
def build_reference_grid_from_scans(scans: List[Tuple[np.ndarray, np.ndarray, Dict]],
                                    mode: str = "first",
                                    ppm_tol: float = 10.0) -> np.ndarray:
    if not scans:
        return np.array([], dtype=float)
    if mode == "first":
        return scans[0][0].copy()
    if mode != "union":
        raise ValueError("ref-mode 只能为 {'first','union'}")
    all_mz = np.concatenate([mz for mz, _, _ in scans])
    if all_mz.size == 0:
        return np.array([], dtype=float)
    mz_sorted = np.sort(all_mz, kind="mergesort")
    merged = [mz_sorted[0]]
    for x in mz_sorted[1:]:
        last = merged[-1]
        ppm = abs(x - last) / max(last, 1e-12) * 1e6
        if ppm <= ppm_tol:
            merged[-1] = (last + x) / 2.0
        else:
            merged.append(x)
    return np.asarray(merged, dtype=float)
def map_scan_to_ref(mz_ref: np.ndarray, mz: np.ndarray, it: np.ndarray, ppm_tol: float) -> np.ndarray:
    y = np.zeros(mz_ref.shape, dtype=float)
    if mz.size == 0 or mz_ref.size == 0:
        return y
    idx = np.searchsorted(mz_ref, mz)
    left_idx = np.clip(idx - 1, 0, mz_ref.size - 1)
    right_idx = np.clip(idx, 0, mz_ref.size - 1)
    dl = np.abs(mz - mz_ref[left_idx])
    dr = np.abs(mz - mz_ref[right_idx])
    choose_left = dl <= dr
    nearest = np.where(choose_left, left_idx, right_idx)
    ref_vals = mz_ref[nearest]
    in_tol = (np.abs(mz - ref_vals) / np.maximum(ref_vals, 1e-12) * 1e6) <= ppm_tol
    if np.any(in_tol):
        sel_idx = nearest[in_tol]
        sel_it = it[in_tol].astype(float)
        np.add.at(y, sel_idx, sel_it)
    return y
def main():
    ap = argparse.ArgumentParser(description="Build wide matrix (Fix Chinese Path)")
    ap.add_argument("--input", required=True, help="输入目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--tolerance-ppm", type=float, default=10.0, help="ppm 容差")
    ap.add_argument("--min-intensity", type=float, default=100.0, help="整帧丢弃阈值")
    ap.add_argument("--ref-mode", type=str, default="first", choices=["first", "union"])
    ap.add_argument("--ms-levels", type=str, default="1")
    ap.add_argument("--start-rt", type=float, default=None)
    ap.add_argument("--end-rt", type=float, default=None)
    ap.add_argument("--max-scans", type=int, default=None)
    args = ap.parse_args()
    in_root = Path(args.input)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    folder_name = in_root.name
    sample_prefix = folder_name.split(' ')[0]
    if not sample_prefix:
        sample_prefix = folder_name
    wide_filename = f"{sample_prefix}.csv"
    meta_filename = f"{sample_prefix}_meta.csv"
    ms_levels = set()
    if args.ms_levels:
        for tok in args.ms_levels.split(","):
            if tok.strip(): ms_levels.add(int(tok.strip()))
    if not ms_levels: ms_levels = {1}
    mzml_files = sorted([p for p in in_root.iterdir() if p.is_file() and p.suffix.lower() == ".mzml"])
    if not mzml_files:
        print("[错误] 未找到 mzML 文件。")
        return
    scans = []
    metas = []
    print(f"[配置] 输入目录: {folder_name}")
    print(f"[配置] 输出文件名: {wide_filename}")
    print(f"[配置] 帧丢弃阈值: {args.min_intensity}")
    for i, mzml in enumerate(mzml_files):
        print(f"  读取 ({i + 1}/{len(mzml_files)}): {mzml.name}")
        file_scans = read_scans_from_mzml(
            mzml,
            ms_levels=ms_levels,
            start_rt_min=args.start_rt,
            end_rt_min=args.end_rt,
            min_intensity=args.min_intensity
        )
        for mz, it, meta in file_scans:
            scans.append((mz, it, meta))
            metas.append({
                "sample": meta["sample"],
                "native_id": meta["native_id"],
                "rt_min": meta["rt_min"],
                "ms_level": meta["ms_level"],
                "column_name": f"{meta['sample']}|{meta['native_id']}|rt={meta['rt_min']:.4f}ms{meta['ms_level']}"
            })
            if args.max_scans is not None and len(scans) >= args.max_scans:
                break
        if args.max_scans is not None and len(scans) >= args.max_scans:
            break
    if not scans:
        print(f"[错误] 没有符合条件的扫描。")
        return
    print(f"[信息] 最终保留有效扫描数: {len(scans)}")
    print("[信息] 构建参考网格...")
    mz_ref = build_reference_grid_from_scans(scans, mode=args.ref_mode, ppm_tol=args.tolerance_ppm)
    print("[信息] 生成矩阵...")
    columns = ["m/z"] + [m["column_name"] for m in metas]
    matrix = np.zeros((mz_ref.size, len(columns)), dtype=float)
    matrix[:, 0] = mz_ref
    for j, (mz, it, meta) in enumerate(scans, start=1):
        if j % 100 == 0: print(f"  Scan {j}/{len(scans)}", end="\r")
        matrix[:, j] = map_scan_to_ref(mz_ref, mz, it, ppm_tol=args.tolerance_ppm)
    print("")
    wide_path = out_root / wide_filename
    wide_df = pd.DataFrame(matrix, columns=columns)
    wide_df.to_csv(wide_path, index=False, encoding="utf-8")
    meta_path = out_root / meta_filename
    pd.DataFrame(metas).to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"[完成] 矩阵文件: {wide_path}")
    print(f"[完成] 元数据文件: {meta_path}")
if __name__ == "__main__":
    main()