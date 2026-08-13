import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import load_dataset

SAMPLE_SIZE = 30

def plot_normal(x, mean, var):
	return (1 / (var * math.sqrt(2 * math.pi))) * (math.exp( - ((x - mean) ** 2) / (2 * (var) ** 2 )))


def main():
	df = load_dataset("Test_knight.csv")
	df_Sensitivity = df["Sensitivity"]

	df_Sensitivity_mean = df_Sensitivity.sum() / len(df_Sensitivity)
	df_Sensitivity_variance = sum((x - df_Sensitivity_mean) ** 2 for x in df_Sensitivity) / len(df_Sensitivity)
	df_Sensitivity_std = math.sqrt(df_Sensitivity_variance)
	print(f"DF mean     		: {df_Sensitivity_mean:.2f}")
	print(f"DF variance 		: {df_Sensitivity_variance:.2f}")
	print(f"DF std      		: {df_Sensitivity_std:.2f}")
	print("-"*50)

	df_sample = df_Sensitivity.sample(n=SAMPLE_SIZE)
	df_sample_mean = df_sample.mean()
	df_sample_variance = sum((x - df_sample_mean) ** 2 for x in df_sample) / (len(df_sample) - 1)
	df_sample_std = math.sqrt(df_sample_variance)
	print(f"DF sample mean		: {df_sample_mean}")
	print(f"DF sample variance	: {df_sample_variance:.2f}")
	print(f"DF sample std		: {df_sample_std:.2f}")

	sns.histplot(data=df_Sensitivity, bins=SAMPLE_SIZE, label="Full DF", stat="density")
	sns.histplot(data=df_sample, bins=SAMPLE_SIZE, label="Sample DF", stat="density")
	x1_values = sorted(df_Sensitivity)
	y1_values = [plot_normal(x1, df_Sensitivity_mean, df_Sensitivity_std) for x1 in x1_values]

	x2_values = sorted(df_sample)
	y2_values = [plot_normal(x2, df_sample_mean, df_sample_std) for x2 in x2_values]
	plt.plot(x1_values, y1_values, color="red", label="Loi normale")
	plt.plot(x2_values, y2_values, color="blue", label="Loi normale estimee")
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()