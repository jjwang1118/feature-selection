from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np




def split_data(save_path, split_ratio=0.8):
    """
    分割數據為訓練集和測試集，並保存到文件
    
    參數:
        save_path: str，保存路徑
        split_ratio: float，訓練集比例（默認 0.8）
    
    返回:
        (train_data, train_label), (test_data, test_label)
    """
    data = pd.read_csv("dataset/ai_impact_student_performance_dataset-convert.csv")
    feature_cols = data.drop(columns=["passed"])    
    target_col = data["passed"]

    train_data, test_data, train_label, test_label = train_test_split(
        feature_cols,
        target_col,
        test_size=1-split_ratio,
        random_state=42,
        stratify=target_col
    )
    
   
    os.makedirs(save_path, exist_ok=True)
    
 
    train_set = pd.concat([train_data, train_label], axis=1)
    test_set = pd.concat([test_data, test_label], axis=1)
    
    save_dir = Path(save_path)
    train_set.to_csv(save_dir / "train.csv", index=False)
    test_set.to_csv(save_dir / "test.csv", index=False)

    split_info = {
        "train_size": len(train_data),
        "test_size": len(test_data),
        "split_ratio": split_ratio,
        "random_state": 42
    }
    
    with open(save_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Train set ({len(train_data)} samples) saved to {save_dir / 'train.csv'}")
    print(f"✅ Test set ({len(test_data)} samples) saved to {save_dir / 'test.csv'}")
    print(f"✅ Split info saved to {save_dir / 'split_info.json'}")


    return (train_data, train_label), (test_data, test_label)




