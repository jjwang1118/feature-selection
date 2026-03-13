import process.download_process as dp
import util.feature_distribution as fd
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

