import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt	
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils.utils import load_dataset


def normalize_data(x: pd.DataFrame) -> pd.DataFrame:
	x = x.select_dtypes(include="number")
	return (x - x.min()) / (x.max() - x.min())

def main():
	df_test = load_dataset(ROOT / "Test_knight.csv")
	df_train = load_dataset(ROOT / "Train_knight.csv")
	feat_mix = ("Deflection", "Survival")

	print(df_test)
	df_test_norm = normalize_data(df_test)
	print(df_test_norm)

	print(df_train)
	df_train_norm = normalize_data(df_train)
	df_train_norm["knight"] = df_train["knight"]
	print(df_train_norm)

	sns.scatterplot(data=df_train_norm, x=feat_mix[0], y=feat_mix[1],
					 hue="knight", palette={"Sith": "red", "Jedi": "blue"})
	plt.show()


if __name__ == "__main__":
	main()