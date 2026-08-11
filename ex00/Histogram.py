import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

p = Path(__file__).parents[1]

def load_dataset(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Fichier introuvable : {path}")
	return pd.read_csv(path)

def main():
	df_test = load_dataset(p / "Test_knight.csv")
	df_train = load_dataset(p / "Train_knight.csv")

	print(f"[df test]\n{df_test.dtypes}")
	print("="*25)
	print(f"[df train]\n{df_train.dtypes}")

	#1 On veut voir la distribution du DF de test
	fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 24))
	axes = axes.flatten()

	for i, col in enumerate(df_test.columns):
		sns.histplot(data=df_test[col], bins=30, ax=axes[i])
		axes[i].set_title(col)

	plt.tight_layout()
	plt.show()

	#2 On regarde si la distribution change en les donnees de test et de train
	fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 24))
	axes = axes.flatten()

	for i, col in enumerate(df_test.columns):
		sns.histplot(data=df_test[col], bins=30, ax=axes[i], label="Test", stat="density")
		sns.histplot(data=df_train[col], bins=30, ax=axes[i], label="Train", stat="density")
		axes[i].set_title(col)
		axes[i].legend()

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()