import os
import joblib
import pandas as pd


MODEL_PATH = "models/gripper_failure_model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["features"]


def predict_failure(
    cycle_count,
    temperature,
    suction_pressure,
    response_time,
    leak_rate,
):
    model, features = load_model()

    input_data = pd.DataFrame(
        [
            {
                "cycle_count": cycle_count,
                "temperature": temperature,
                "suction_pressure": suction_pressure,
                "response_time": response_time,
                "leak_rate": leak_rate,
            }
        ]
    )

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return prediction, probability


if __name__ == "__main__":

    prediction, probability = predict_failure(
        cycle_count=7500,
        temperature=58,
        suction_pressure=-0.55,
        response_time=1.9,
        leak_rate=4.5,
    )

    print("\n========== GRIPPER PREDICTION ==========")

    if prediction == 1:
        print("Prediction: FAILURE")
    else:
        print("Prediction: NORMAL")

    print(f"Failure Probability: {probability:.2%}")
