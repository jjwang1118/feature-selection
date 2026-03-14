## 將每個屬性計算平均和中位數
#並將數據視覺化，內容顯示數據、平均和中位數的分布情況、也要顯示數值
#檔案名為該屬性名稱

import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# 设置 matplotlib 参数
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


def need_caculate_feature():
    """從 download_process.py 複製的函數，用於判斷哪些特徵需要標示 mean/median"""
    academic_performance = [
        'last_exam_score',
        'assignment_scores_avg',
        'final_score',
        'concept_understanding_score',
        'improvement_rate'
    ]
    
    ai_usage = [
        'ai_usage_time_minutes',
        'ai_dependency_score',
        'ai_generated_content_percentage',
        'ai_prompts_per_week'
    ]
    
    study_behavior = [
        'study_hours_per_day',
        'attendance_percentage',
        'study_consistency_index',
        'class_participation_score'
    ]
    
    lifestyle = [
        'sleep_hours',
        'social_media_hours',
        'tutoring_hours'
    ]
    
    secondary = [
        'age',
        'ai_ethics_score',
        'uses_ai',
        'passed'
    ]
    
    all_numeric_columns = (
        academic_performance + 
        ai_usage + 
        study_behavior + 
        lifestyle + 
        secondary
    )
    
    return all_numeric_columns



def download_stastic_report(mean_median_report_path, missing_value_report_path):

    if os.path.exists(mean_median_report_path):
        with open(mean_median_report_path, "r", encoding="utf-8") as f:
            mean_median_report = json.load(f)
    else:
        mean_median_report = {}
    
    if os.path.exists(missing_value_report_path):
        with open(missing_value_report_path, "r", encoding="utf-8") as f:
            missing_value_report = json.load(f)
    else:
        missing_value_report = {}
    
    return mean_median_report, missing_value_report



def feature_graph():
    """
    決定每個 feature 用哪一種圖形呈現
    
    圖表類型：
    - histogram: 直方圖（數值型連續變量）
    - bar: 條形圖（類別變量、二元變量）
    """
    
    show_graph = {
        # 基本信息
        "age": "histogram",
        "gender": "bar",
        "grade_level": "bar",
        
        # 學習行為
        "study_hours_per_day": "histogram",
        "attendance_percentage": "histogram",
        "study_consistency_index": "histogram",
        "class_participation_score": "histogram",
        
        # AI 使用相關
        "uses_ai": "bar",
        "ai_usage_time_minutes": "histogram",
        "ai_tools_used": "bar",
        "ai_usage_purpose": "bar",
        "ai_dependency_score": "histogram",
        "ai_generated_content_percentage": "histogram",
        "ai_prompts_per_week": "histogram",
        "ai_ethics_score": "histogram",
        
        # 學業表現
        "last_exam_score": "histogram",
        "assignment_scores_avg": "histogram",
        "concept_understanding_score": "histogram",
        "improvement_rate": "histogram",
        "final_score": "histogram",
        "passed": "bar",
        "performance_category": "bar",
        
        # 生活習慣
        "sleep_hours": "histogram",
        "social_media_hours": "histogram",
        "tutoring_hours": "histogram",
    }
    
    return show_graph


def draw_histogram(data, col, mean_val=None, median_val=None, store_path="./"):
    """
    繪製直方圖（用於數值型變量）
    
    參數:
        data: Series，數據
        col: str，欄位名稱
        mean_val: float，平均數（如果需要標示）
        median_val: float，中位數（如果需要標示）
        store_path: str，保存路徑
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 繪製直方圖
    n, bins, patches = ax.hist(data, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    
    # 標示 mean 和 median（如果提供）
    if mean_val is not None:
        ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2.5, 
                   label=f'Mean = {mean_val:.2f}')
    
    if median_val is not None:
        ax.axvline(median_val, color='#f39c12', linestyle='--', linewidth=2.5, 
                   label=f'Median = {median_val:.2f}')
    
    # 設置標籤和標題
    ax.set_xlabel(col.replace('_', ' ').title(), fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.set_title(f'Distribution of {col.replace("_", " ").title()}', fontsize=18, fontweight='bold', pad=20)
    
    # 添加圖例
    if mean_val is not None or median_val is not None:
        ax.legend(fontsize=14, loc='upper right')
    
    # 添加網格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(store_path, exist_ok=True)
    output_path = os.path.join(store_path, f"{col}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Histogram saved: {output_path}")


def draw_bar_chart(data, col, store_path="./"):
    """
    繪製條形圖（用於類別變量和二元變量）
    
    參數:
        data: Series，數據
        col: str，欄位名稱
        store_path: str，保存路徑
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 統計每個類別的數量
    value_counts = data.value_counts().sort_index()
    
    # 使用不同顏色
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    bar_colors = [colors[i % len(colors)] for i in range(len(value_counts))]
    
    # 繪製條形圖
    bars = ax.bar(value_counts.index.astype(str), value_counts.values, 
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 在每個條形上添加數值標籤
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # 設置標籤和標題
    ax.set_xlabel(col.replace('_', ' ').title(), fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title(f'Distribution of {col.replace("_", " ").title()}', fontsize=18, fontweight='bold', pad=20)
    
    # 添加網格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(store_path, exist_ok=True)
    output_path = os.path.join(store_path, f"{col}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Bar chart saved: {output_path}")


def draw_picture(data, col, med_mean_path, missing_value_path, store_path="results/statistic/image"):
    """
    根據欄位類型繪製對應的圖表
    
    參數:
        data: DataFrame，完整數據
        col: str，要繪製的欄位名稱
        med_mean_path: str，mean/median JSON 路徑
        missing_value_path: str，missing value JSON 路徑
        store_path: str，圖片保存路徑
    """
    # 檢查欄位是否存在
    if col not in data.columns:
        print(f"⚠️  Column '{col}' not found in data. Skipping...")
        return
    
    # 獲取圖表配置
    graph_dict = feature_graph()
    med_mean_report, missing_value_report = download_stastic_report(med_mean_path, missing_value_path)
    
    # 獲取該欄位的數據
    col_data = data[col].dropna()
    
    if len(col_data) == 0:
        print(f"⚠️  Column '{col}' has no data. Skipping...")
        return
    
    # 判斷是否需要標示 mean/median
    calc_features = need_caculate_feature()
    should_show_stats = col in calc_features
    
    # 獲取 mean 和 median（如果需要且存在）
    mean_val = None
    median_val = None
    
    if should_show_stats and col in med_mean_report:
        mean_val = med_mean_report[col].get('mean')
        median_val = med_mean_report[col].get('median')
    
    # 判斷圖表類型
    # 1. 檢查是否是 One-Hot 編碼後的列（_usage 或 _task 後綴）
    if col.endswith('_usage') or col.endswith('_task'):
        draw_bar_chart(col_data, col, store_path)
    
    # 2. 檢查是否在預定義的圖表字典中
    elif col in graph_dict:
        chart_type = graph_dict[col]
        
        if chart_type == "histogram":
            draw_histogram(col_data, col, mean_val, median_val, store_path)
        elif chart_type == "bar":
            draw_bar_chart(col_data, col, store_path)
    
    # 3. 自動判斷：如果數值類型較少，使用 bar，否則 histogram
    else:
        unique_count = col_data.nunique()
        
        if unique_count <= 10:
            draw_bar_chart(col_data, col, store_path)
        else:
            # 只有數值型才能用 histogram
            if pd.api.types.is_numeric_dtype(col_data):
                draw_histogram(col_data, col, mean_val, median_val, store_path)
            else:
                draw_bar_chart(col_data, col, store_path)


def draw_all_pictures(data, med_mean_path, missing_value_path, store_path="results/stastic/image"):
    """
    參數:
        data: DataFrame，完整數據
        med_mean_path: str，mean/median JSON 路徑
        missing_value_path: str，missing value JSON 路徑
        store_path: str，圖片保存路徑
    """
    print("=" * 60)
    print("Starting to generate visualizations...")
    print("=" * 60)
    
    # 排除 student_id
    columns_to_plot = [col for col in data.columns if col != 'student_id']
    
    for i, col in enumerate(columns_to_plot, 1):
        print(f"\n[{i}/{len(columns_to_plot)}] Processing: {col}")
        try:
            draw_picture(data, col, med_mean_path, missing_value_path, store_path)
        except Exception as e:
            print(f"❌ Error processing {col}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ All visualizations completed! Saved to: {store_path}")
    print("=" * 60)


