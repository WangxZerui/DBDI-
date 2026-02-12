import pandas as pd
import os
def batch_filter_weak_spectra(input_folder, output_folder, threshold=100000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"--> 已新建输出文件夹: {output_folder}")
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    if not files:
        print(f"警告: 在 '{input_folder}' 中没有找到 .csv 文件。")
        return
    print(f"共找到 {len(files)} 个文件，开始批量处理...\n" + "-" * 30)
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(file_path)
            scan_intensities = df.iloc[:, 1:].sum()
            strong_cols = scan_intensities[scan_intensities >= threshold].index.tolist()
            cols_to_keep = [df.columns[0]] + strong_cols
            df_filtered = df[cols_to_keep]
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}_filtered.csv"
            output_path = os.path.join(output_folder, new_filename)
            df_filtered.to_csv(output_path, index=False)
            removed_count = (df.shape[1] - 1) - len(strong_cols)
            print(f"[{filename}] 处理完毕")
            print(f"   - 保留: {len(strong_cols)} 组 | 删除: {removed_count} 组 (弱信号)")
        except Exception as e:
            print(f"xx 文件 {filename} 处理出错: {e}")
    print("-" * 30 + "\n所有文件处理完成！")
my_input_folder = r'J:\白酒数据集\test'
my_output_folder = r'J:\白酒数据集\test1'
batch_filter_weak_spectra(my_input_folder, my_output_folder, threshold=10000)