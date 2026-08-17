import sys
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt	
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils.utils import load_dataset


def main():
	df_train = load_dataset(ROOT / "Train_knight.csv")
	df_test = load_dataset(ROOT / "Test_knight.csv")

	feat_separate = ("Empowered", "Prescience")
	feat_mix = ("Deflection", "Survival")

	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))

	sns.scatterplot(data=df_train, x=feat_separate[0], y=feat_separate[1],
					 hue="knight", palette={"Sith": "red", "Jedi": "blue"}, ax=axes[0][0])
	sns.scatterplot(data=df_train, x=feat_mix[0], y=feat_mix[1],
					 hue="knight", palette={"Sith": "red", "Jedi": "blue"}, ax=axes[0][1])
	sns.scatterplot(data=df_test, x=feat_separate[0], y=feat_separate[1],
					 color="green", label="Knight", ax=axes[1][0])
	sns.scatterplot(data=df_test, x=feat_mix[0], y=feat_mix[1],
					 color="green", label="Knight", ax=axes[1][1])

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()