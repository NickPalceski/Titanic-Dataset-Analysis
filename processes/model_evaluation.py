from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	confusion_matrix,
	precision_score,
	recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from processes.preprocessing import build_preprocessor


def evaluate_models(x_train: pd.DataFrame, x_valid: pd.DataFrame,
	                y_train: pd.Series, y_valid: pd.Series,
	                output_dir: Path,
                    ) -> tuple[dict[str, Pipeline], pd.DataFrame]:
	"""Evaluate KNN and Naive Bayes then save confusion matrix plots.

	Returns:
	- pipelines: trained model pipelines by name
	- results: comparison table with metrics and 10-fold CV error rate
	"""

	models = {
		"KNN": KNeighborsClassifier(n_neighbors=7),
		"NaiveBayes": GaussianNB(),
	}

	pipelines: dict[str, Pipeline] = {}
	rows = []

	for name, estimator in models.items():
		pipeline = Pipeline(
			steps=[
				("preprocessor", build_preprocessor()),
				("model", estimator),
			]
		)

		pipeline.fit(x_train, y_train)
		y_pred = pipeline.predict(x_valid)

		accuracy = accuracy_score(y_valid, y_pred)
		precision = precision_score(y_valid, y_pred, zero_division=0)
		recall = recall_score(y_valid, y_pred, zero_division=0)
		cm = confusion_matrix(y_valid, y_pred)

		cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
		cv_accuracy = cross_val_score(
			pipeline,
			pd.concat([x_train, x_valid]),
			pd.concat([y_train, y_valid]),
			cv=cv,
			scoring="accuracy",
		)
		cv_mean_accuracy = cv_accuracy.mean()
		cv_error_rate = 1 - cv_mean_accuracy

		rows.append(
			{
				"Model": name,
				"Accuracy": round(accuracy, 4),
				"Precision": round(precision, 4),
				"Recall": round(recall, 4),
				"CV Mean Accuracy": round(cv_mean_accuracy, 4),
				"CV Error Rate": round(cv_error_rate, 4),
			}
		)

		disp = ConfusionMatrixDisplay(confusion_matrix=cm)
		disp.plot()
		
		plt.title(f"{name}'s Confusion Matrix")
		plt.tight_layout()
		plt.savefig(output_dir / f"{name.lower()}_confusion_matrix.png")
		plt.close()

		pipelines[name] = pipeline

	results = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
	
	return pipelines, results
