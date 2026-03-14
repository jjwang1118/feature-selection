import pandas as pd
from sklearn.feature_selection import SelectKBest, RFECV, SelectFromModel, mutual_info_classif, RFE
from sklearn.tree import DecisionTreeClassifier

def run_filter(X_train, y_train, k):
    """
    Filter Method: Mutual Information (mutual_info_classif)
    Evaluates the dependency between each feature and the target independently.
    """
    # Safeguard: if k is greater than total features, just select all
    k = min(k, X_train.shape[1]) 
    if k <= 0:
        return []
        
    # Using Mutual Information which handles both continuous and categorical data well
    mi_func = lambda X, y: mutual_info_classif(X, y, random_state=42)
    selector = SelectKBest(score_func=mi_func, k=k)
    selector.fit(X_train, y_train)
    
    # Get the names of the selected columns
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features


def run_wrapper(X_train, y_train, estimator=None):
    """
    Wrapper Method: Recursive Feature Elimination with Cross-Validation (RFECV)
    Automatically finds the optimal number of features by recursively dropping the weakest ones.
    """
    if estimator is None:
        estimator = DecisionTreeClassifier(random_state=42)
        
    # cv=5 means 5-fold cross-validation
    selector = RFECV(estimator=estimator, step=1, cv=5, scoring='f1')
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features


def run_embedded(X_train, y_train, estimator=None):
    """
    Embedded Method: Decision Tree Feature Importance
    Selects features based on the internal node weights/importance of a trained tree.
    """
    if estimator is None:
        estimator = DecisionTreeClassifier(random_state=42)
        
    # SelectFromModel by default picks features with importance > mean importance
    selector = SelectFromModel(estimator=estimator)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features

from sklearn.feature_selection import RFE
import numpy as np

def run_wrapper_k(X_train, y_train, k, estimator=None):
    """Wrapper Method: 強制精確選擇 K 個特徵 (RFE)"""
    if estimator is None:
        from sklearn.tree import DecisionTreeClassifier
        estimator = DecisionTreeClassifier(random_state=42)
        
    # 使用普通的 RFE (非 RFECV) 讓我們可以強制指定 n_features_to_select
    selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    return selected_features

def run_embedded_k(X_train, y_train, k, estimator=None):
    """Embedded Method: 強制精確選擇 K 個特徵 (Decision Tree Importances)"""
    if estimator is None:
        from sklearn.tree import DecisionTreeClassifier
        estimator = DecisionTreeClassifier(random_state=42)
        
    estimator.fit(X_train, y_train)
    importances = estimator.feature_importances_
    
    # 找出重要性最高的前 k 個特徵的 index
    # argsort 會由小到大排序，所以我們取最後 k 個 [-k:]
    top_k_indices = np.argsort(importances)[-k:]
    
    selected_features = X_train.columns[top_k_indices].tolist()
    return selected_features