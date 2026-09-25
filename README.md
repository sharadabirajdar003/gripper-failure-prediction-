# 🤖 Gripper Failure Prediction System

An end-to-end machine learning application for predicting suction cup
gripper failures using industrial sensor data.

The project combines machine learning, sensor analytics, predictive
maintenance concepts, and an interactive Streamlit dashboard.

---

## 🚀 Features

- Synthetic industrial sensor dataset
- Machine learning-based failure prediction
- Random Forest classification
- Feature scaling
- Failure probability estimation
- Risk classification
- Feature importance analysis
- Sensor monitoring dashboard
- Maintenance recommendations
- Dataset analytics
- Failure distribution visualization
- Temperature and leak-rate analysis
- Reusable prediction script

---

## 🧠 Machine Learning Features

The model uses the following sensor parameters:

| Feature | Description |
|---|---|
| cycle_count | Number of gripper operating cycles |
| temperature | Gripper/system temperature |
| suction_pressure | Vacuum/suction pressure |
| response_time | Gripper response time |
| leak_rate | Estimated air leakage |
| failure | Target variable |

---

## 🏗️ Architecture

```text
Sensor Data
     |
     v
Data Generation / Collection
     |
     v
Data Preprocessing
     |
     v
Feature Scaling
     |
     v
Random Forest Classifier
     |
     +-------------------+
     |                   |
     v                   v
Prediction         Feature Importance
     |
     v
Failure Probability
     |
     v
Risk Classification
     |
     v
Streamlit Dashboard
