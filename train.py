
# if __name__ == "__main__":
#     print("This is the main entry point for the training script.")

#     if exp==0:
#         # baseline
#         pass

#     elif exp==1:
#         pass
    
#     elif exp==2:
#         pass

#     elif exp==3:
#         pass
    

import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from util.feature_selectors import run_filter, run_wrapper, run_embedded
from util.evaluation import evaluate_model, calculate_overlap

def load_and_prep_data(data_path, seed=42, test_size=0.2):
    """讀取處理好的數據，將類別轉為數值，並切分訓練集/測試集"""
    df = pd.read_csv(f"{data_path}/ai_impact_student_performance_dataset-convert.csv")
    
    # 將所有文字類型的欄位轉換為數字 (Decision Tree 需要)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
            
    # 定義特徵 X 和目標 y
    X = df.drop(columns=['passed', 'student_id', 'performance_category', 'final_score'], errors='ignore')
    y = df['passed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Starting Training Pipeline...")
    
    # 1. 讀取現有的 Config
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        
    exp_idx = config["exp_idx"]
    k_value = config["k"]
    data_path = config["data_path"]
    
    # 固定項目 (從 README 來的)
    SEED = 42
    TEST_SIZE = 0.2
    
    # 2. 準備資料
    X_train, X_test, y_train, y_test = load_and_prep_data(data_path, SEED, TEST_SIZE)
    print(f"Data shape: X_train {X_train.shape}, X_test {X_test.shape}")

    # 建立實驗結果資料夾 (如果沒有在 config 裡，我們自己建一個預設的)
    results_dir = f'results/exp{exp_idx}'
    os.makedirs(results_dir, exist_ok=True)

    if exp_idx == 0:
        print("--- Running Baseline (All Features) ---")
        model = DecisionTreeClassifier(random_state=SEED)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        print(f"Baseline Metrics: {metrics}")
        
        # 儲存結果
        with open(f"{results_dir}/baseline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    elif exp_idx == 1:
        print("--- Running Experiment 1 ---")
        
        # Step A: 讓 Wrapper 自動找最佳特徵
        wrapper_features = run_wrapper(X_train, y_train)
        n_wrapper = len(wrapper_features)
        print(f"Wrapper selected {n_wrapper} features.")
        
        # Step B: 讓 Embedded 自動找最佳特徵
        embedded_features = run_embedded(X_train, y_train)
        n_embedded = len(embedded_features)
        print(f"Embedded selected {n_embedded} features.")
        
        # Step C: Filter 根據前兩者的數量去抓取
        filter_features_for_wrapper = run_filter(X_train, y_train, k=n_wrapper)
        filter_features_for_embedded = run_filter(X_train, y_train, k=n_embedded)
        
        # === 訓練與評估: Matchup 1 (Wrapper vs Filter) ===
        print(f"\n[Matchup 1: Wrapper vs Filter (N={n_wrapper})]")
        
        dt_wrap = DecisionTreeClassifier(random_state=SEED)
        dt_wrap.fit(X_train[wrapper_features], y_train)
        metrics_wrap = evaluate_model(dt_wrap, X_test[wrapper_features], y_test)
        
        dt_filt_1 = DecisionTreeClassifier(random_state=SEED)
        dt_filt_1.fit(X_train[filter_features_for_wrapper], y_train)
        metrics_filt_1 = evaluate_model(dt_filt_1, X_test[filter_features_for_wrapper], y_test)
        
        overlap_1 = calculate_overlap(wrapper_features, filter_features_for_wrapper)
        print(f"Wrapper F1: {metrics_wrap['f1_score']:.4f} | Filter F1: {metrics_filt_1['f1_score']:.4f} | Overlap: {overlap_1:.2f}")

        # === 訓練與評估: Matchup 2 (Embedded vs Filter) ===
        print(f"\n[Matchup 2: Embedded vs Filter (N={n_embedded})]")
        
        dt_embed = DecisionTreeClassifier(random_state=SEED)
        dt_embed.fit(X_train[embedded_features], y_train)
        metrics_embed = evaluate_model(dt_embed, X_test[embedded_features], y_test)
        
        dt_filt_2 = DecisionTreeClassifier(random_state=SEED)
        dt_filt_2.fit(X_train[filter_features_for_embedded], y_train)
        metrics_filt_2 = evaluate_model(dt_filt_2, X_test[filter_features_for_embedded], y_test)
        
        overlap_2 = calculate_overlap(embedded_features, filter_features_for_embedded)
        print(f"Embedded F1: {metrics_embed['f1_score']:.4f} | Filter F1: {metrics_filt_2['f1_score']:.4f} | Overlap: {overlap_2:.2f}")

        # === 將所有結果打包成一個大字典並存成 JSON ===
        exp1_results = {
            "matchup_1_wrapper_vs_filter": {
                "n_features": n_wrapper,
                "feature_overlap_ratio": overlap_1,
                "wrapper": {
                    "selected_features": wrapper_features,
                    "metrics": metrics_wrap
                },
                "filter": {
                    "selected_features": filter_features_for_wrapper,
                    "metrics": metrics_filt_1
                }
            },
            "matchup_2_embedded_vs_filter": {
                "n_features": n_embedded,
                "feature_overlap_ratio": overlap_2,
                "embedded": {
                    "selected_features": embedded_features,
                    "metrics": metrics_embed
                },
                "filter": {
                    "selected_features": filter_features_for_embedded,
                    "metrics": metrics_filt_2
                }
            }
        }
        
        # 儲存結果到 exp1 資料夾
        output_file = f"{results_dir}/exp1_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(exp1_results, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Experiment 1 complete! Results neatly logged to {output_file}")

    elif exp_idx == 2:
        pass

    elif exp_idx == 3:
        pass