import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 2500

cycle_count = np.random.randint(100, 10000, N_SAMPLES)

temperature = (
    25
    + (cycle_count / 10000) * 30
    + np.random.normal(0, 3, N_SAMPLES)
)

suction_pressure = (
    -0.65
    - (cycle_count / 10000) * 0.25
    + np.random.normal(0, 0.05, N_SAMPLES)
)

response_time = (
    0.8
    + (cycle_count / 10000) * 1.2
    + np.random.normal(0, 0.15, N_SAMPLES)
)

leak_rate = (
    0.5
    + (cycle_count / 10000) * 4
    + np.random.normal(0, 0.6, N_SAMPLES)
)

# Keep generated values within realistic ranges
temperature = np.clip(temperature, 20, 75)
suction_pressure = np.clip(suction_pressure, -1.2, -0.3)
response_time = np.clip(response_time, 0.4, 3.0)
leak_rate = np.clip(leak_rate, 0, 8)

# Failure score
failure_score = (
    0.00025 * cycle_count
    + 0.045 * np.maximum(temperature - 45, 0)
    + 2.5 * np.maximum(suction_pressure + 0.75, 0)
    + 1.8 * np.maximum(response_time - 1.3, 0)
    + 0.65 * np.maximum(leak_rate - 2, 0)
)

# Add random variation
failure_score += np.random.normal(0, 0.7, N_SAMPLES)

# Convert score into binary failure
failure_probability = 1 / (1 + np.exp(-(failure_score - 2.5)))

failure = (failure_probability > 0.5).astype(int)

df = pd.DataFrame(
    {
        "cycle_count": cycle_count,
        "temperature": np.round(temperature, 2),
        "suction_pressure": np.round(suction_pressure, 3),
        "response_time": np.round(response_time, 3),
        "leak_rate": np.round(leak_rate, 3),
        "failure": failure,
    }
)

os.makedirs("data", exist_ok=True)

output_path = "data/gripper_sensor_data.csv"
df.to_csv(output_path, index=False)

print(f"Dataset created: {output_path}")
print(f"Rows: {len(df)}")
print("\nFailure distribution:")
print(df["failure"].value_counts())
