import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils.utils import load_dataset


def main():
	df_train = load_dataset(ROOT / "Train_knight.csv")

	training, validation = train_test_split(
		df_train,
		test_size=0.2,
		stratify=df_train["knight"],
		random_state=42
	)

	training.to_csv(ROOT / "Training_knight.csv", index=False)
	validation.to_csv(ROOT / "Validation_knight.csv", index=False)

	print(f"Training: {len(training)} lignes ({len(training)/len(df_train)*100:.0f}%)")
	print(f"Validation: {len(validation)} lignes ({len(validation)/len(df_train)*100:.0f}%)")

	fig, ax = plt.subplots(figsize=(8, 5))

	proportions = pd.DataFrame({
		"Train (avant)": df_train["knight"].value_counts(normalize=True),
		"Training": training["knight"].value_counts(normalize=True),
		"Validation": validation["knight"].value_counts(normalize=True),
	})

	proportions.T.plot(kind="bar", ax=ax, color=["red", "blue"])
	
	ax.set_ylabel("Proportion")
	ax.set_title("Répartition Sith/Jedi avant/après split")
	plt.xticks(rotation=45, ha="right")
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main() 