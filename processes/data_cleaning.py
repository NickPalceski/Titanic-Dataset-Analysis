import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
	"""Clean the raw dataset before feature engineering and analysis.

	This standardizes categorical values and fills missing values in the raw data
	so the dataset is in a cleaner state before modeling.
	"""

	cleaned = df.copy()

	if "Sex" in cleaned.columns:
		cleaned["Sex"] = cleaned["Sex"].astype("string").str.strip().str.lower()
		cleaned["Sex"] = cleaned["Sex"].fillna(cleaned["Sex"].mode(dropna=True).iloc[0])

	if "Embarked" in cleaned.columns:
		cleaned["Embarked"] = cleaned["Embarked"].astype("string").str.strip().str.upper()
		cleaned["Embarked"] = cleaned["Embarked"].replace({"": pd.NA})
		cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode(dropna=True).iloc[0])

	for column in ["Age", "Fare", "SibSp", "Parch", "Pclass"]:
		if column in cleaned.columns:
			cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
			if cleaned[column].isna().any():
				cleaned[column] = cleaned[column].fillna(cleaned[column].median())

	return cleaned
