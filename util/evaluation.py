from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """計算準確率、精確率、召回率和 F1-score"""
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }
    return metrics

def calculate_overlap(features_list_1, features_list_2):
    """計算兩個演算法選出的特徵重疊程度"""
    set1 = set(features_list_1)
    set2 = set(features_list_2)
    
    if not set1 or not set2:
        return 0.0
        
    overlap = len(set1.intersection(set2))
    total_unique = len(set1.union(set2))
    return overlap / total_unique