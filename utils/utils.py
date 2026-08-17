import pandas as pd
import math
from pathlib import Path

def load_dataset(path) -> pd.DataFrame:
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Fichier introuvable : {path}")
	return pd.read_csv(path)

def correlation_factor(x_val, y_val):
	x_mean = x_val.mean()
	y_mean = y_val.mean()
	covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_val, y_val))
	sum_sq_x = sum((x - x_mean) ** 2 for x in x_val)
	sum_sq_y = sum((y - y_mean) ** 2 for y in y_val)
	return covariance / math.sqrt(sum_sq_x * sum_sq_y)