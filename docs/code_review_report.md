# 程式碼審查報告

## 一、實驗規格符合性檢查

根據 `README.md` 的實驗內容，比對 `train.py` 與 `feature_selectors.py` 的實作結果。

### ✅ 符合的部分

| 規格 | 程式碼對應 |
|------|-----------|
| baseline：全部特徵 | `train.py` `exp_idx==0` 使用全部特徵訓練 |
| exp1：Wrapper/Embedded 各自選 n，Filter 也用對應 n 比較 | `train.py` 分別取 `n_wrapper` 和 `n_embedded`，Filter 各用對應數量 |
| exp1：不一樣會產生兩輪 | `train.py` Matchup 1 (Wrapper vs Filter) 與 Matchup 2 (Embedded vs Filter) |
| exp2：選第一輪最小值 | `train.py` `min_n = min(n_wrapper_auto, n_embedded_auto)` |
| exp3：Wrapper/Embedded 用其他模型選 k 個特徵 | `train.py` Wrapper 用 RandomForest，Embedded 用 GradientBoosting |
| exp3：Filter 直接選 k 個 | `train.py` `run_filter(X_train, y_train, k=k_value)` |
| classifier：Decision Tree (base model) | 所有實驗的最終分類器都用 `DecisionTreeClassifier` |
| evaluation：accuracy, precision, recall, F1 | `evaluation.py` 四項指標齊全 |
| 特徵重疊程度 (Jaccard similarity) | `evaluation.py` `calculate_overlap` 使用 Jaccard similarity |
| random.seed(42) | `config.json` `"seed": 42`，所有模型和特徵選擇器均使用 |
| 資料分割 80/20 | `config.json` `"data_split_ratio": 0.8` |
| k=7 | `config.json` `"k": 7` |
| drop 特定欄位 | `train.py` 移除 `student_id`, `passed`, `performance_category`, `final_score` |
| one-hot encoding | `train.py` `pd.get_dummies(X_train, drop_first=True)` |

### ❌ 不符合 / 缺漏的部分

1. **Exp2 缺少特徵重疊度計算**
   - README 要求 evaluation 包含「選出特徵重疊程度」，但 exp2 結果只儲存了三種方法的 `selected_features` 和 `metrics`，沒有計算 filter / wrapper / embedded 之間的 overlap。
   - **to do**：在 exp2 中加入兩兩 overlap 計算（filter↔wrapper、filter↔embedded、wrapper↔embedded）

2. **Exp3 缺少特徵重疊度計算**
   - 同上，exp3 的結果也沒有計算三種方法之間的特徵重疊程度。
   - **to do**：在 exp3 中加入兩兩 overlap 計算

3. **Jaccard distance 被註解掉**
   - README 提到「jaccard similarity, jaccard distance」，但 `evaluation.py` 中 `jaccard_distance` 被註解掉，函數只回傳 similarity，不回傳 distance。
   - **to do**：將 `calculate_overlap` 同時回傳 jaccard_similarity 和 jaccard_distance

---

## 二、random.seed(42) 使用範圍

### 設定來源
- `config.json` — `"seed": 42` 作為全域設定

### 各檔案使用情況

#### split_data.py（資料分割）
| 位置 | 用途 |
|------|------|
| L35 | `train_test_split()` 的 `random_state=seed` |

#### train.py（模型訓練）
| 實驗 | 位置 | 用途 |
|------|------|------|
| baseline | L74 | `DecisionTreeClassifier(random_state=SEED)` |
| exp1 | L131, L138, L150, L158 | 4 個 DecisionTreeClassifier |
| exp2 | L224, L232, L240 | 3 個 DecisionTreeClassifier |
| exp3 | L292 | `RandomForestClassifier(random_state=SEED)` |
| exp3 | L296 | `GradientBoostingClassifier(random_state=SEED)` |
| exp3 | 分類器 | 3 個 DecisionTreeClassifier |

#### feature_selectors.py（特徵選擇）
| 位置 | 用途 |
|------|------|
| L16 | `mutual_info_classif(X, y, random_state=42)` |
| L31 | `run_wrapper` 預設 DecisionTreeClassifier |
| L47 | `run_embedded` 預設 DecisionTreeClassifier |
| L63 | `run_wrapper_k` 預設 DecisionTreeClassifier |
| L76 | `run_embedded_k` 預設 DecisionTreeClassifier |

### ⚠️ 潛在問題

- `feature_selectors.py` 中的 `random_state=42` 是**硬編碼**的，沒有讀取 config 中的 seed。目前值恰好一致所以不影響結果，但如果未來改 config 中的 seed，`feature_selectors.py` 不會跟著變。
- **to do**：考慮將 `feature_selectors.py` 的 seed 改為參數傳入，統一由 config 控管
