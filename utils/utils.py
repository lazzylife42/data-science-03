import pandas as pd
from pathlib import Path

def load_dataset(path) -> pd.DataFrame:
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Fichier introuvable : {path}")
	return pd.read_csv(path)
