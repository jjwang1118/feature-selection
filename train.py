import json
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier

# from util.feature_selectors import run_filter, run_wrapper, run_embedded
from util.evaluation import evaluate_model, calculate_overlap
from util.feature_selectors import run_filter, run_wrapper, run_embedded, run_embedded_k, run_wrapper_k
import joblib
import util.tree_visualize as tv


def load_split_data(data_path):
    """
    Loads the pre-split train/test data from Wang's pipeline, 
    separates features/targets, and handles categorical encoding.
    """
    split_path = f"{data_path}"
    
    train_df = pd.read_csv(f"{split_path}/train.csv")
    test_df = pd.read_csv(f"{split_path}/test.csv")
    
    # Drop target and data-leakage columns (like final_score)
    drop_cols = ['passed', 'student_id', 'performance_category', 'final_score']
    
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['passed']
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['passed']
    
    # Safely convert text categories into numerical format (better than LabelEncoder)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Align columns to ensure train and test sets have identical features
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Starting Training Pipeline...")
    
    # 1. Read the existing Config
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        
    exp_idx = config.get("exp_idx", 0)
    k_value = config.get("k", 7)
    data_path = config.get("data_path", "dataset")
    # Fixed settings
    SEED=config.get("seed", 42)
    
    # 2. Prepare Data (load pre-split data)
    split_path = f"{data_path}/split"
    X_train, X_test, y_train, y_test = load_split_data(split_path)
    print(f"Data shape: X_train {X_train.shape}, X_test {X_test.shape}")

    # Create results directory
    results_dir = f'results/exp{exp_idx}'
    os.makedirs(results_dir, exist_ok=True)
    model_file=f"model/exp_{exp_idx if exp_idx != 0 else 'baseline'}"
    model_name=f"exp_{exp_idx if exp_idx != 0 else 'baseline'}.pkl"

    if exp_idx == 0:
        print("\n--- Running Baseline (All Features) ---")
        if os.path.exists(f"{model_file}/{model_name}"):
            print(f"✅ exp_{exp_idx if exp_idx !=0 else 'baseline'} model already exists. Skipping training.")
            clf = joblib.load(f"{model_file}/{model_name}")  # 載入已存在的模型
        else:
            clf = DecisionTreeClassifier(
                criterion='entropy',
                random_state=SEED,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)
            os.makedirs(model_file, exist_ok=True) # Creates the folder if it doesn't exist
            joblib.dump(clf, f"{model_file}/{model_name}")
            print(f"✅ Baseline model saved to {model_file}/{model_name}")

        tv.visualize_decision_tree(clf,save_path=f"{results_dir}/baseline_tree.png", feature_names=X_train.columns.tolist(), class_names=['Not Passed', 'Passed'])
        tv.visualize_decision_tree_matplotlib(clf, save_path=f"{results_dir}/baseline_tree_matplotlib.png", feature_names=X_train.columns.tolist(), class_names=['Not Passed', 'Passed'], max_depth=10)
        tv.export_tree_text(clf, save_path=f"{results_dir}/baseline_tree.txt", feature_names=X_train.columns.tolist())
        matric=evaluate_model(clf, X_test, y_test)

        # matrix , depth and nodes
        matric['depth_nodes']={}
        matric['depth_nodes']['max_depth']=int(clf.get_depth())
        matric['depth_nodes']['n_leaves']=int(clf.get_n_leaves())
        with open(f"{results_dir}/baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(matric, f, indent=4, ensure_ascii=False)

        # importance of features
        feature_importance={}
        importances = clf.feature_importances_.tolist()
        feature_names= X_train.columns.tolist()
        for feature,value in  zip(feature_names, importances):
            feature_importance[feature]=value
        
        feature_importance_sorted = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        with open(f"{results_dir}/feature_importance.json", "w", encoding="utf-8") as f:
            json.dump(feature_importance_sorted, f, indent=4, ensure_ascii=False)




    elif exp_idx == 1:
        print("\n--- Running Experiment 1 (Feature Selection Comparisons) ---")
        
        model_dir = f"model/exp_{exp_idx}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Step A: Wrapper
        wrapper_features = run_wrapper(X_train, y_train)
        n_wrapper = len(wrapper_features)
        print(f"Wrapper selected {n_wrapper} features.")
        
        # Step B: Embedded
        embedded_features = run_embedded(X_train, y_train)
        n_embedded = len(embedded_features)
        print(f"Embedded selected {n_embedded} features.")
        
        # Step C: Filter based on counts
        filter_features_for_wrapper = run_filter(X_train, y_train, k=n_wrapper)
        filter_features_for_embedded = run_filter(X_train, y_train, k=n_embedded)
        
        # === Matchup 1: Wrapper vs Filter ===
        print(f"\n[Matchup 1: Wrapper vs Filter (N={n_wrapper})]")
        
        dt_wrap = DecisionTreeClassifier(criterion='entropy', random_state=SEED, class_weight='balanced')
        dt_wrap.fit(X_train[wrapper_features], y_train)
        metrics_wrap = evaluate_model(dt_wrap, X_test[wrapper_features], y_test)
        
        joblib.dump(dt_wrap, f"{model_dir}/wrapper_model.pkl")
        tv.visualize_decision_tree_matplotlib(dt_wrap, save_path=f"{results_dir}/wrapper_tree.png", feature_names=wrapper_features, class_names=['Not Passed', 'Passed'], max_depth=10)
        
        dt_filt_1 = DecisionTreeClassifier(criterion='entropy', random_state=SEED, class_weight='balanced')
        dt_filt_1.fit(X_train[filter_features_for_wrapper], y_train)
        metrics_filt_1 = evaluate_model(dt_filt_1, X_test[filter_features_for_wrapper], y_test)
        joblib.dump(dt_filt_1, f"{model_dir}/filter_model_wrapper.pkl")
        tv.visualize_decision_tree_matplotlib(dt_filt_1, save_path=f"{results_dir}/filter_tree_wrapper.png", feature_names=filter_features_for_wrapper, class_names=['Not Passed', 'Passed'], max_depth=10)
        
        overlap_1 = calculate_overlap(wrapper_features, filter_features_for_wrapper)
        print(f"Wrapper F1: {metrics_wrap['f1_score']:.4f} | Filter F1: {metrics_filt_1['f1_score']:.4f} | Overlap: {overlap_1:.2f}")

        # === Matchup 2: Embedded vs Filter ===
        print(f"\n[Matchup 2: Embedded vs Filter (N={n_embedded})]")
        
        dt_embed = DecisionTreeClassifier(criterion='entropy', random_state=SEED, class_weight='balanced')
        dt_embed.fit(X_train[embedded_features], y_train)
        metrics_embed = evaluate_model(dt_embed, X_test[embedded_features], y_test)
        
        joblib.dump(dt_embed, f"{model_dir}/embedded_model.pkl")
        tv.visualize_decision_tree_matplotlib(dt_embed, save_path=f"{results_dir}/embedded_tree.png", feature_names=embedded_features, class_names=['Not Passed', 'Passed'], max_depth=10)
        
        
        dt_filt_2 = DecisionTreeClassifier(criterion='entropy', random_state=SEED, class_weight='balanced')
        dt_filt_2.fit(X_train[filter_features_for_embedded], y_train)
        metrics_filt_2 = evaluate_model(dt_filt_2, X_test[filter_features_for_embedded], y_test)
        
        joblib.dump(dt_filt_2, f"{model_dir}/filter_model_embedded.pkl")
        tv.visualize_decision_tree_matplotlib(dt_filt_2, save_path=f"{results_dir}/filter_tree_embedded.png", feature_names=filter_features_for_embedded, class_names=['Not Passed', 'Passed'], max_depth=10)
        
        
        overlap_2 = calculate_overlap(embedded_features, filter_features_for_embedded)
        print(f"Embedded F1: {metrics_embed['f1_score']:.4f} | Filter F1: {metrics_filt_2['f1_score']:.4f} | Overlap: {overlap_2:.2f}")

        # === Package and Save Results ===
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
        
        output_file = f"{results_dir}/exp1_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(exp1_results, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Experiment 1 complete! Results neatly logged to {output_file}")

    elif exp_idx == 2:
        print("\n--- Running Experiment 2 (The Bottleneck Phase) ---")
        model_dir = f"model/exp_{exp_idx}"
        os.makedirs(model_dir, exist_ok=True)
        
        #Step A: 找出第一輪的最小值 (Minimum N)
        n_wrapper_auto = len(run_wrapper(X_train, y_train))
        n_embedded_auto = len(run_embedded(X_train, y_train))
        min_n = min(n_wrapper_auto, n_embedded_auto)
        
        min_n = max(1, min_n)
        print(f"Wrapper wanted {n_wrapper_auto}, Embedded wanted {n_embedded_auto}.")
        print(f"Bottleneck activated: Forcing all algorithms to select exactly {min_n} features.")
        # Step B: 三種方法各自選出 min_n 個特徵
        features_filter = run_filter(X_train, y_train, k=min_n)
        features_wrapper = run_wrapper_k(X_train, y_train, k=min_n)
        features_embedded = run_embedded_k(X_train, y_train, k=min_n)
        
        # === 訓練與評估三種模型 ===
        
        # 1. Filter Model
        dt_filt = DecisionTreeClassifier(criterion='entropy', random_state=SEED, class_weight='balanced')
        dt_filt.fit(X_train[features_filter], y_train)
        metrics_filt = evaluate_model(dt_filt, X_test[features_filter], y_test)
        
        joblib.dump(dt_filt, f"{model_dir}/filter_model.pkl")
        tv.visualize_decision_tree_matplotlib(dt_filt, save_path=f"{results_dir}/filter_tree.png", feature_names=features_filter, class_names=['Not Passed', 'Passed'], max_depth=10)

        # 2. Wrapper Model
        dt_wrap = DecisionTreeClassifier(criterion='entropy', random_state=SEED, class_weight='balanced')
        dt_wrap.fit(X_train[features_wrapper], y_train)
        metrics_wrap = evaluate_model(dt_wrap, X_test[features_wrapper], y_test)
        
        joblib.dump(dt_wrap, f"{model_dir}/wrapper_model.pkl")
        tv.visualize_decision_tree_matplotlib(dt_wrap, save_path=f"{results_dir}/wrapper_tree.png", feature_names=features_wrapper, class_names=['Not Passed', 'Passed'], max_depth=10)

        # 3. Embedded Model
        dt_embed = DecisionTreeClassifier(criterion='entropy', random_state=SEED, class_weight='balanced')
        dt_embed.fit(X_train[features_embedded], y_train)
        metrics_embed = evaluate_model(dt_embed, X_test[features_embedded], y_test)
        
        joblib.dump(dt_embed, f"{model_dir}/embedded_model.pkl")
        tv.visualize_decision_tree_matplotlib(dt_embed, save_path=f"{results_dir}/embedded_tree.png", feature_names=features_embedded, class_names=['Not Passed', 'Passed'], max_depth=10)

        # 列印結果比較
        print(f"\n[Bottleneck Results (N={min_n})]")
        print(f"Filter F1:   {metrics_filt['f1_score']:.4f}")
        print(f"Wrapper F1:  {metrics_wrap['f1_score']:.4f}")
        print(f"Embedded F1: {metrics_embed['f1_score']:.4f}")

        # === Package and Save JSON Results ===
        exp2_results = {
            "bottleneck_n": min_n,
            "filter": {
                "selected_features": features_filter,
                "metrics": metrics_filt
            },
            "wrapper": {
                "selected_features": features_wrapper,
                "metrics": metrics_wrap
            },
            "embedded": {
                "selected_features": features_embedded,
                "metrics": metrics_embed
            }
        }
        
        output_file = f"{results_dir}/exp2_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(exp2_results, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Experiment 2 complete! Results logged to {output_file}")
        
    elif exp_idx == 3:
        pass