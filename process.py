import process.download_process as dp
import process.split_data as sd
import util.feature_distribution as fd
import util.data_compose as dc
import os
import json
import pandas as pd
import matplotlib.pyplot as plt




if __name__ == "__main__":

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    dataset_path = config["data_path"]
    visual_path = config["feature_visual"]
    med_mean_path = config["median_mean_cac"]
    missing_value_path = config["missing_value"]
    split_ratio = config.get("data_split_ratio", 0.8)

    dp.download_dataset("ai-impact-on-student-performance", dataset_path)
    data = pd.read_csv(f"{dataset_path}/ai_impact_student_performance_dataset.csv", keep_default_na=False, na_values=['', 'NA', 'NaN', 'NULL'])


    dp.process_missing_values(data, dataset_path, missing_value_path)
    data = pd.read_csv(f"{dataset_path}/ai_impact_student_performance_dataset-filtered.csv")

    dp.col_process(data, cols=["grade_level", "ai_tools_used"], store_path=dataset_path)
    data_convert = pd.read_csv(f"{dataset_path}/ai_impact_student_performance_dataset-convert.csv")
    
    for col in data_convert.columns:
        dp.caculate_mean(data_convert, dataset_path, col, med_mean_path)
        dp.caculate_median(data_convert, dataset_path, col, med_mean_path)

    fd.draw_all_pictures(data_convert, med_mean_path, missing_value_path, visual_path)

    # 分割訓練集和測試集
    split_save_path = "dataset/split"
    (train_data, train_label), (test_data, test_label) = sd.split_data(split_save_path, split_ratio=split_ratio)
    
    # 可視化訓練集和測試集的分佈對比
    train_full = pd.read_csv(f"{split_save_path}/train.csv")
    test_full = pd.read_csv(f"{split_save_path}/test.csv")
    dc.visualize_train_test_distribution(train_full, test_full, "results/stastic/compose/image")
    dc.record_feature_statistics(train_full, test_full, "results/stastic/compose")



