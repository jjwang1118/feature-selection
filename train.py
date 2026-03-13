
import json
import sklearn
from sklearn import tree
import process.split_data as sd

if __name__ == "__main__":
    print("This is the main entry point for the training script.")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    exp = config.get("exp_idx", 0)

    if exp==0:
        # baseline
        clf=tree.DecisionTreeClassifier(
            criteriom='entropy'
            random_state=42
            class
        )
        (train_data, train_label), (test_data, test_label) = sd.split_data(split_save_path, split_ratio=0.8)
        std_passed_clf=clf.fit(train_data, train_label)


        pass

    elif exp==1:
        pass
    
    elif exp==2:
        pass

    elif exp==3:
        pass