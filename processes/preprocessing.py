from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
	"""Create preprocessing pipeline before modeling.

	- Numeric columns: median imputer and scaler standardization
	- Categorical columns: mode imputer and one-hot encoding
	"""

	numeric_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
	categorical_columns = ["Sex", "Embarked"]

	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
		]
	)

	return ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_columns),
			("cat", categorical_transformer, categorical_columns),
		]
	)
