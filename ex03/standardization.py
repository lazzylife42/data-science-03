import sys
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt	
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils.utils import load_dataset

def standerdize_data(x: pd.DataFrame) -> pd.DataFrame:
	x = x.select_dtypes(include="number")
	return (x - x.mean()) / x.std()

def main():
	df_test = load_dataset(ROOT / "Test_knight.csv")
	df_train = load_dataset(ROOT / "Train_knight.csv")
	feat_separate = ("Empowered", "Prescience")

	print(df_test)
	df_test_std = standerdize_data(df_test)
	print(df_test_std)

	print(df_train)
	df_train_std = standerdize_data(df_train)
	df_train_std["knight"] = df_train["knight"]
	print(df_train_std)

	sns.scatterplot(data=df_train_std, x=feat_separate[0], y=feat_separate[1],
					 hue="knight", palette={"Sith": "red", "Jedi": "blue"})
	plt.show()


if __name__ == "__main__":
	main()