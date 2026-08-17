import sys
import math
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))
from utils.utils import load_dataset


def correlation_factor(x_val, y_val):
	x_mean = x_val.mean()
	y_mean = y_val.mean()
	covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_val, y_val))
	sum_sq_x = sum((x - x_mean) ** 2 for x in x_val)
	sum_sq_y = sum((y - y_mean) ** 2 for y in y_val)
	return covariance / math.sqrt(sum_sq_x * sum_sq_y)

def main():
	df_train = load_dataset(ROOT / "Train_knight.csv")
	y_val = (df_train["knight"] == "Jedi").astype(int)

	results = {"knight": 1.0}
	for col in df_train.columns:
		if col == "knight":
			continue
		results[col] = correlation_factor(df_train[col], y_val)

	for col, r in sorted(results.items(), key=lambda item: item[1], reverse=True):
		print(f"{col:<15}: {r:.6f}")


if __name__ == "__main__":
	main()