from sklearn.tree import export_graphviz, export_text, plot_tree
import graphviz
import matplotlib.pyplot as plt
import os


def visualize_decision_tree(model, save_path, feature_names=None, class_names=None):
    """
    使用 graphviz 生成精美的決策樹圖片

    參數:
        model: 已訓練的 DecisionTreeClassifier
        save_path: str，圖片保存路徑（不含副檔名）
        feature_names: list，特徵名稱
        class_names: list，類別名稱（如 ["不及格", "及格"]）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render(save_path, format="png", cleanup=True)
    print(f"✅ Graphviz tree saved to {save_path}.png")


def visualize_decision_tree_matplotlib(model, save_path, feature_names=None, class_names=None, max_depth=None):
    """
    使用 matplotlib 的 plot_tree 生成決策樹圖片

    參數:
        model: 已訓練的 DecisionTreeClassifier
        save_path: str，圖片保存路徑（含副檔名，如 .png）
        feature_names: list，特徵名稱
        class_names: list，類別名稱
        max_depth: int，最大顯示深度（None 顯示全部，樹很大時建議設 3~5）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(30, 15))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        ax=ax,
        max_depth=max_depth,
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Matplotlib tree saved to {save_path}")


def export_tree_text(model, save_path, feature_names=None):
    """
    將決策樹匯出為文字格式

    參數:
        model: 已訓練的 DecisionTreeClassifier
        save_path: str，文字檔保存路徑
        feature_names: list，特徵名稱
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    text = export_text(model, feature_names=feature_names)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Tree text saved to {save_path}")