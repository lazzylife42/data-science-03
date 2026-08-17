import sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))

from utils.utils import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


def plot_grid(columns, plot_func, nrows=6, ncols=5):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        plot_func(col, axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()


def main():
    df_test = load_dataset(ROOT / "Test_knight.csv")
    df_train = load_dataset(ROOT / "Train_knight.csv")

    plot_grid(
        df_test.columns,
        lambda col, ax: sns.histplot(data=df_test[col], bins=30, ax=ax)
    )

    def plot_overlay(col, ax):
        sns.histplot(data=df_test[col], bins=30, ax=ax, label="Test", stat="density")
        sns.histplot(data=df_train[col], bins=30, ax=ax, label="Train", stat="density")
        ax.legend()

    plot_grid(df_test.columns, plot_overlay)


if __name__ == "__main__":
    main()