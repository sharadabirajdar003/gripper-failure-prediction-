import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Gripper Failure Prediction",
    page_icon="🤖",
    layout="wide",
)


MODEL_PATH = "models/gripper_failure_model.pkl"
DATA_PATH = "data/gripper_sensor_data.csv"


FEATURES = [
    "cycle_count",
    "temperature",
    "suction_pressure",
    "response_time",
    "leak_rate",
]


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    artifact = joblib.load(MODEL_PATH)

    return artifact["model"]


@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 Gripper Failure Prediction System")

st.markdown(
    """
    **Machine Learning-based predictive maintenance dashboard**

    This application predicts potential suction cup gripper failures
    using industrial sensor parameters.
    """
)


if model is None:

    st.error(
        "Trained model not found. Please run `generate_data.py` "
        "followed by `train_model.py`."
    )

    st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Sensor Input")

cycle_count = st.sidebar.number_input(
    "Cycle Count",
    min_value=0,
    max_value=20000,
    value=5000,
    step=100,
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.5,
)

suction_pressure = st.sidebar.number_input(
    "Suction Pressure (bar)",
    min_value=-1.5,
    max_value=0.0,
    value=-0.75,
    step=0.01,
)

response_time = st.sidebar.number_input(
    "Response Time (s)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.05,
)

leak_rate = st.sidebar.number_input(
    "Leak Rate",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

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

prediction = model.predict(input_data)[0]

probability = model.predict_proba(input_data)[0][1]


# --------------------------------------------------
# Risk calculation
# --------------------------------------------------

if probability >= 0.75:
    risk = "HIGH"
    color = "🔴"

elif probability >= 0.50:
    risk = "MEDIUM"
    color = "🟠"

else:
    risk = "LOW"
    color = "🟢"


# --------------------------------------------------
# Dashboard metrics
# --------------------------------------------------

st.subheader("Prediction Result")

col1, col2, col3 = st.columns(3)

with col1:

    if prediction == 1:
        st.error("⚠️ FAILURE PREDICTED")
    else:
        st.success("✅ NORMAL OPERATION")


with col2:

    st.metric(
        "Failure Probability",
        f"{probability:.1%}",
    )


with col3:

    st.metric(
        "Risk Level",
        f"{color} {risk}",
    )


# --------------------------------------------------
# Sensor overview
# --------------------------------------------------

st.subheader("Sensor Readings")

metric_columns = st.columns(5)

sensor_values = [
    ("Cycle Count", f"{cycle_count:,}"),
    ("Temperature", f"{temperature:.1f} °C"),
    ("Suction Pressure", f"{suction_pressure:.2f} bar"),
    ("Response Time", f"{response_time:.2f} s"),
    ("Leak Rate", f"{leak_rate:.2f}"),
]

for column, (label, value) in zip(
    metric_columns,
    sensor_values,
):

    with column:
        st.metric(label, value)


# --------------------------------------------------
# Recommendation
# --------------------------------------------------

st.subheader("Maintenance Recommendation")

if risk == "HIGH":

    st.error(
        """
        High failure probability detected.

        Recommended actions:
        - Inspect suction cup condition.
        - Check for air leakage.
        - Verify vacuum pressure.
        - Inspect pneumatic connections.
        - Consider preventive maintenance.
        """
    )

elif risk == "MEDIUM":

    st.warning(
        """
        Moderate failure probability detected.

        Recommended actions:
        - Monitor sensor values.
        - Inspect abnormal parameters.
        - Schedule preventive inspection.
        """
    )

else:

    st.success(
        """
        Low failure probability.

        Continue monitoring the gripper sensor parameters.
        """
    )


# --------------------------------------------------
# Dataset Analytics
# --------------------------------------------------

if df is not None:

    st.divider()

    st.header("📊 Dataset Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Samples",
            f"{len(df):,}",
        )

    with col2:
        failures = int(df["failure"].sum())

        st.metric(
            "Failure Samples",
            f"{failures:,}",
        )

    with col3:

        failure_rate = df["failure"].mean()

        st.metric(
            "Failure Rate",
            f"{failure_rate:.1%}",
        )


    # --------------------------------------------------
    # Feature Importance
    # --------------------------------------------------

    st.subheader("Feature Importance")

    classifier = model.named_steps["classifier"]

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": classifier.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color="#1f77b4",
    )

    ax.set_xlabel("Importance")

    ax.set_title(
        "Random Forest Feature Importance"
    )

    st.pyplot(fig)


    # --------------------------------------------------
    # Failure Distribution
    # --------------------------------------------------

    st.subheader("Failure Distribution")

    failure_counts = (
        df["failure"]
        .value_counts()
        .sort_index()
    )

    labels = ["Normal", "Failure"]

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        labels,
        [
            failure_counts.get(0, 0),
            failure_counts.get(1, 0),
        ],
        color=[
            "#2ca02c",
            "#d62728",
        ],
    )

    ax.set_ylabel("Number of Samples")

    ax.set_title(
        "Normal vs Failure Samples"
    )

    st.pyplot(fig)


    # --------------------------------------------------
    # Temperature vs Failure
    # --------------------------------------------------

    st.subheader("Temperature vs Failure")

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    normal = df[df["failure"] == 0]

    failure = df[df["failure"] == 1]

    ax.scatter(
        normal["temperature"],
        normal["leak_rate"],
        alpha=0.35,
        label="Normal",
        color="green",
    )

    ax.scatter(
        failure["temperature"],
        failure["leak_rate"],
        alpha=0.45,
        label="Failure",
        color="red",
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Leak Rate")

    ax.legend()

    st.pyplot(fig)


    # --------------------------------------------------
    # Raw data
    # --------------------------------------------------

    with st.expander("View Dataset"):

        st.dataframe(
            df,
            use_container_width=True,
        )


st.divider()

st.caption(
    "Gripper Failure Prediction System | "
    "Python • Scikit-learn • Pandas • Streamlit"
)
