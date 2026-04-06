from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from analytics.data_cleaning import clean_dataset
from analytics.feature_importance import get_mutual_important_features
from analytics.model_evaluation import evaluate_models


def main() -> None:

	# Set project paths
	project_dir = Path(__file__).resolve().parent
	output_dir = project_dir / "outputs"
	output_dir.mkdir(exist_ok=True)

	# Load training and testing datasets
	train_path = project_dir / "train.csv"

	train_df = pd.read_csv(train_path)

	# Clean raw data
	train_df = clean_dataset(train_df)

	# Set model features and target
	feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
	target_column = "Survived"

	x = train_df[feature_columns]
	y = train_df[target_column]

	# Split training data into different sets (training/validation)
	x_train, x_valid, y_train, y_valid = train_test_split(
		x,
		y,
		random_state=7,
		stratify=y,
	)

	# Train and evaluate both models (KNN and Naive Bayes)
	_, results_df = evaluate_models(x_train, x_valid, y_train, y_valid, output_dir)

	# Analyze feature importance
	importance_df = get_mutual_important_features(x, y)

	# Print summary
	print()
	print("Main features used for prediction:")
	print()
	print("- Passenger class (Pclass)")
	print("- Sex")
	print("- Age")
	print("- Number of siblings/spouses (SibSp)")
	print("- Number of parents/children (Parch)")
	print("- Fare")
	print("- Embarked")
	print()
	print("Model Evaluation:")
	print(results_df.to_string(index=False))
	print()
	print("Most important features by mutual information:")
	print(importance_df.head(5).to_string(index=False))

if __name__ == "__main__":
	main()
