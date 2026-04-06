import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from analytics.preprocessing import build_preprocessor


def get_mutual_important_features(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
	"""Get the importance of features using mutual information.

	Tells us which features are most informative for predicting survival.
	"""

	preprocessor = build_preprocessor()
	x_processed = preprocessor.fit_transform(x)
	feature_names = preprocessor.get_feature_names_out()

	mi_scores = mutual_info_classif(x_processed, y, random_state=7)
	importance_df = pd.DataFrame(
		{
			"Feature": feature_names,
			"Importance": mi_scores,
		}
	).sort_values(by="Importance", ascending=False)

	return importance_df
