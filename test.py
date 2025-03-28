import pandas as pd
import numpy as np
import os
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "./"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False


def load_data():
    baseline = pd.read_csv(os.path.join(DATA_DIR, "baseline_results.csv"))
    tree = pd.read_csv(os.path.join(DATA_DIR, "tree_results.csv"))
    custom_model = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))

    baseline["Model"] = "Baseline"
    tree["Model"] = "Tree"
    custom_model["Model"] = "Deep"

    return pd.concat([baseline, tree, custom_model], axis=0)


def perform_statistical_tests(df, metric="MAPE"):
    groups = df["Model"].unique()
    samples = [df[df["Model"] == g][metric] for g in groups]

    # Kruskal-Wallis检验
    h_stat, p_kw = kruskal(*samples)
    print(f"\nKruskal-Wallis H = {h_stat:.2f}, p = {p_kw:.2e}")

    if p_kw < 0.05:
        # Dunn检验
        dunn_matrix = posthoc_dunn(
            df,
            group_col="Model",
            val_col=metric,
            p_adjust="holm"
        )
        print("\nDunn results:")
        print(dunn_matrix.map(lambda x: f"{x:.2e}"))
        return dunn_matrix
    else:
        return None


def visualize_and_save(df, dunn_matrix=None, metric="MAPE"):
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="Model",
        y=metric,
        hue="Model",
        data=df,
        palette="muted",
        legend=False
    )
    plt.title("comparison - MAPE")
    plt.xlabel("model")
    plt.ylabel("MAPE (%)")
    plt.grid(axis="y", alpha=0.3)

    # 保存结果
    plot_path = os.path.join(OUTPUT_DIR, f"model_comparison_{metric}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"保存至：{os.path.abspath(plot_path)}")


if __name__ == "__main__":
    df = load_data()
    print(f"\ndata：\n{df.groupby('Model')['MAPE'].describe()}")
    dunn_matrix = perform_statistical_tests(df, metric="MAPE")
    visualize_and_save(df, dunn_matrix)
