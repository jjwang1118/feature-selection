# 實驗內容

## 📋 實驗目標
- target :及格/不及格

## 📊 使用資料集
- **資料來源**: https://www.kaggle.com/datasets/ankushnarwade/ai-impact-on-student-performance
- **資料說明**: 26個特徵，判斷學生是否及格

## 🔬 實驗內容
- type2
    - threshold : choose top-k 個特徵(k=7)
    - method (n為演算法選取的特徵數量)
        - baseline : 直接選全部特徵
        - exp1 如果不一樣會產生兩輪
            - 假設Wrapper 得出 n = 15 最佳， Filter 也使用 n = 15 去比較
            - 若Embedded 得出 n = 8 ..， Filter 也使用 n = 8 …..
        - exp2
            - 選第一輪**最小值**的特徵數量，其他相同
        - exp3
            - wrapper : 拿到k個特徵,**另外用其他模型幫忙找出特徵**
            - embed : 拿到k個特徵，**另外用其他模型幫忙找出特徵**
            - FILTER : 直接拿k個特徵
    - classifier (choose $?$)
        - **decision tree(base model)**
    -  evaluation
        - accuracy
        - precision
        - recall
        - F1-score
        - featue quality
            - 選出特徵數量
            - 選出特徵重疊程度


## 🛠️ 使用工具
- `torch`
- `pandas`
- `scikit-learn` **主要訓練用**
- `matplotlib`
- `json`
- `os`
- `pathlib`
-  實際情況請參考 `requirements.txt`

## 🐍 環境建立
```bash
# 建立 conda 環境
conda create -n feature_selection python=3.13 -y

# 啟動環境
conda activate feature_selection

# 安裝套件（不含 torch 與 scikit-learn）
pip install -r requirements.txt
```


## 👥 工作分配
- 資料處理、資料分析、模型訓練 *1【baseline】 (wang)
- 模型訓練 *3 (東)

## ⚙️ 固定項目
- `random.seed(42)`
- 資料分割比例: **80%** 訓練集，**20%** 測試集

## 工作
- **資料處理**
    - 時間處理
    - 編碼(one-hot encoding)
    - 拆分使用工具欄位
    - 處理缺失值(丟/補)
    - 篩選影響欄位('student_id','passed', 'performance_category', 'final_score')
    - split data 內使用 stratify 以保持資料分佈一致

- **資料分析**: 
    - 個特徵平均值、中位數、缺失值數量
    - 特徵分佈圖
    - 分析資料是否平衡 
        - 根據 #stastic/med_mean.json 欄位    "passed{"mean": 0.88925,"median": 1.0} 可以知道資料不平衡，因為大多數學生都及格了
    - 分析訓練集和測試集的分佈是否相似

- **模型訓練**: baseline、wrapper、filter、embedded
    - 要點跟**實驗內容**一致
    - 使用random seed 確保實驗可重現
    - 將樹狀結構視覺化
- **模型評估**
    - 評估指標: accuracy、precision、recall、F1-score
    - 各實驗相似度 : jaccard similarity , jaccard distance

## 📁 目錄架構

```
feature selection/
├── checkpoint/         # 模型檢查點存放目錄
├── dataset/            # 資料集存放和處理後的資料目錄
├── docs/               # 文檔目錄
│   └── exp_content.md  # 實驗內容說明
├── model/              # 模型定義目錄
├── process/            # 資料處理相關模組
├── util/               # 工具函數目錄
|── results/            # 儲存實驗結果的目錄
|   ├── exp1/           
│   ├── exp2/
│   └── exp3/
├── process.py          # 資料處理主程式
└── train.py            # 訓練主程式
├── config.json         # 實驗配置文件
└── eval.py             # 評估主程式
```
