import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

DATA_PATH = "data/gripper_sensor_data.csv"
MODEL_PATH = "models/gripper_failure_model.pkl"

FEATURES = [
    "cycle_count",
    "temperature",
    "suction_pressure",
    "response_time",
    "leak_rate",
]

TARGET = "failure"


def train_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found. Run generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("\n========== MODEL PERFORMANCE ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": pipeline,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")

    # Feature importance
    classifier = pipeline.named_steps["classifier"]

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        "models/feature_importance.csv",
        index=False,
    )


if __name__ == "__main__":
    train_model()
