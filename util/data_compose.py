
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置 matplotlib 参数
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


def visualize_train_test_distribution(train_data, test_data, save_path="results/stastic/compose"):
    """
    可視化訓練集和測試集的分佈對比
    
    參數:
        train_data: DataFrame，訓練集數據
        test_data: DataFrame，測試集數據
        save_path: str，圖片保存路徑
    """
    print("=" * 60)
    print("Starting to generate train/test distribution visualizations...")
    print("=" * 60)
    
    os.makedirs(save_path, exist_ok=True)
    
    # 排除 student_id
    columns_to_plot = [col for col in train_data.columns if col != 'student_id']
    
    for i, col in enumerate(columns_to_plot, 1):
        print(f"\n[{i}/{len(columns_to_plot)}] Processing: {col}")
        try:
            _plot_feature_comparison(train_data[col], test_data[col], col, save_path)
        except Exception as e:
            print(f"❌ Error processing {col}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ All train/test comparisons completed! Saved to: {save_path}")
    print("=" * 60)


def _plot_feature_comparison(train_col, test_col, col_name, save_path):
    """
    繪製單個特徵在訓練集和測試集的分佈對比
    
    參數:
        train_col: Series，訓練集中的欄位數據
        test_col: Series，測試集中的欄位數據
        col_name: str，欄位名稱
        save_path: str，保存路徑
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 移除空值
    train_data_clean = train_col.dropna()
    test_data_clean = test_col.dropna()
    
    if len(train_data_clean) == 0 or len(test_data_clean) == 0:
        print(f"⚠️  Column '{col_name}' has no data. Skipping...")
        plt.close()
        return
    
    # 判斷數值型還是類別型
    if pd.api.types.is_numeric_dtype(train_data_clean) and train_data_clean.nunique() > 10:
        # 數值型：繪製重疊直方圖
        ax.hist(train_data_clean, bins=30, alpha=0.6, color='#3498db', 
                label=f'Train (n={len(train_data_clean)})', edgecolor='black', density=True)
        ax.hist(test_data_clean, bins=30, alpha=0.6, color='#e74c3c', 
                label=f'Test (n={len(test_data_clean)})', edgecolor='black', density=True)
        
        ax.set_ylabel('Density', fontsize=16, fontweight='bold')
    else:
        # 類別型：繪製並排條形圖
        train_counts = train_data_clean.value_counts().sort_index()
        test_counts = test_data_clean.value_counts().sort_index()
        
        # 對齊索引
        all_categories = sorted(set(train_counts.index) | set(test_counts.index))
        train_counts = train_counts.reindex(all_categories, fill_value=0)
        test_counts = test_counts.reindex(all_categories, fill_value=0)
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        ax.bar(x - width/2, train_counts.values, width, 
               label=f'Train (n={len(train_data_clean)})', 
               color='#3498db', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, test_counts.values, width, 
               label=f'Test (n={len(test_data_clean)})', 
               color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(cat) for cat in all_categories], rotation=45, ha='right')
        ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    
    # 設置標籤和標題
    ax.set_xlabel(col_name.replace('_', ' ').title(), fontsize=16, fontweight='bold')
    ax.set_title(f'Train vs Test Distribution: {col_name.replace("_", " ").title()}', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # 添加圖例
    ax.legend(fontsize=14, loc='upper right')
    
    # 添加網格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖片
    output_path = os.path.join(save_path, f"{col_name}_train_test.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison saved: {output_path}")


def record_feature_statistics(train_data, test_data, save_path="result/stastic/compose"):
    """
    紀錄每個特徵的類別名稱、對應數值（train/test 計數）、中位數和平均數

    參數:
        train_data: DataFrame，訓練集數據
        test_data: DataFrame，測試集數據
        save_path: str，JSON 保存路徑

    返回:
        dict，特徵統計資訊
    """
    columns = [col for col in train_data.columns if col != 'student_id']
    result = {}

    for col in columns:
        train_col = train_data[col].dropna()
        test_col = test_data[col].dropna()

        feature_info = {}

        # 每個唯一值的 train/test 計數
        train_counts = train_col.value_counts()
        test_counts = test_col.value_counts()
        all_values = sorted(set(train_counts.index) | set(test_counts.index))

        for val in all_values:
            key = str(val)
            feature_info[key] = [
                int(train_counts.get(val, 0)),
                int(test_counts.get(val, 0))
            ]

        # 平均數和中位數
        if pd.api.types.is_numeric_dtype(train_col):
            feature_info["mean"] = [float(train_col.mean()), float(test_col.mean())]
            feature_info["median"] = [float(train_col.median()), float(test_col.median())]
        else:
            feature_info["mean"] = None
            feature_info["median"] = None

        result[col] = feature_info

    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, "feature_statistics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✅ Feature statistics saved to: {output_path}")
    return result
