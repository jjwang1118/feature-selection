
import json

if __name__ == "__main__":
    print("This is the main entry point for the training script.")

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    exp = config.get("exp_idx", 0)

    if exp==0:
        # baseline
        pass

    elif exp==1:
        pass
    
    elif exp==2:
        pass

    elif exp==3:
        pass