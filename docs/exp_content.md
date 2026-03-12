# 實驗內容

## 📋 實驗目標
- 先決定實驗內容

## 📊 使用資料集
- **資料來源**: https://www.kaggle.com/datasets/ankushnarwade/ai-impact-on-student-performance
- **資料說明**: 33個特徵，判斷學生是否及格

## 🔬 實驗內容
- type1 or type2

## 🛠️ 使用工具
- `torch`
- `pandas`
- `scikit-learn`
- `matplotlib`

## 👥 工作分配
- 資料處理、資料分析、模型訓練 *1
- 模型訓練 *2

## ⚙️ 固定項目
- `random.seed(42)`
- 資料分割比例: **80%** 訓練集，**20%** 測試集

## 📁 目錄架構

```
feature selection/
├── checkpoint/          # 模型檢查點存放目錄
├── dataset/            # 資料集存放和處理後的資料目錄
├── docs/               # 文檔目錄
│   └── exp_content.md  # 實驗內容說明
├── model/              # 模型定義目錄
├── process/            # 資料處理相關模組
├── util/               # 工具函數目錄
├── process.py          # 資料處理主程式
└── train.py            # 訓練主程式
```

