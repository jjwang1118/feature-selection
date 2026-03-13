import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_split_data(data_path):
    """
    Loads the train/test splits, separates features and targets, 
    and handles any lingering string/categorical columns.
    """
    split_path = f"{data_path}/split"
    
    train_df = pd.read_csv(f"{split_path}/train.csv")
    test_df = pd.read_csv(f"{split_path}/test.csv")
    
    # Separate features (X) and target (y)
    # Dropping 'student_id' because IDs shouldn't be used as predictive features
    X_train = train_df.drop(columns=["passed", "student_id", "performance_category", "final_score"], errors='ignore')
    y_train = train_df["passed"]
    
    X_test = test_df.drop(columns=["passed", "student_id"], errors='ignore')
    y_test = test_df["passed"]
    
    # Failsafe: One-hot encode any remaining text columns (like gender, ai_usage_purpose)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Align columns just in case the test set is missing a category present in the train set
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    print("This is the main entry point for the training script.")

    # ---------------------------------------------------------
    # Load settings from config.json
    # ---------------------------------------------------------
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    exp = config.get("exp_idx", 0)
    data_path = config.get("data_path", "dataset")
    k_folds = config.get("k", 7) # Ready for future experiments!

    # Load the data using the path from config
    X_train, y_train, X_test, y_test = load_split_data(data_path)

    if exp == 0:
        print("\n" + "="*50)
        print("🚀 Running Baseline (Experiment 0): Decision Tree")
        print("="*50)
        
        # Initialize and train
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = dt.predict(X_test)
        
        print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\n📊 Classification Report:\n")
        print(classification_report(y_test, y_pred))

    elif exp == 1:
        print("\n" + "="*50)
        print("🚀 Running Experiment 1: Model Comparison")
        print("="*50)
        
        # Define the three models we want to compare
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results = []

        # Loop through, train, and test each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            results.append({"Model": name, "Accuracy": acc})
            
        print("\n🏆 Final Comparison:")
        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        print(results_df.to_string(index=False))
    
    elif exp == 2:
        pass

    elif exp == 3:
        pass