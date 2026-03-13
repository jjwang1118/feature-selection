import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, SelectFromModel, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

def run_filter(X_train, y_train, k):
    """Filter Method: ANOVA (f_classif)"""
    # 如果 k 等於特徵總數，就全選
    k = min(k, X_train.shape[1]) 
    
    mi_func = lambda X, y: mutual_info_classif(X, y, random_state=42)
    selector = SelectKBest(score_func=mi_func, k=k) # f_classif
    selector.fit(X_train, y_train)
    
    # 獲取被選中的特徵名稱
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features

def run_wrapper(X_train, y_train, estimator=None):
    """Wrapper Method: RFECV 自動找出最佳特徵數量 n"""
    if estimator is None:
        estimator = DecisionTreeClassifier(random_state=42)
        
    selector = RFECV(estimator=estimator, step=1, cv=5, scoring='f1')
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features

def run_embedded(X_train, y_train, estimator=None):
    """Embedded Method: Decision Tree Feature Importance"""
    if estimator is None:
        estimator = DecisionTreeClassifier(random_state=42)
        
    # SelectFromModel 預設會挑選 importance 大於平均值的特徵
    selector = SelectFromModel(estimator=estimator)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features