# Predicting Remaining Useful Life (RUL) of Jet Engines

This project utilizes machine learning techniques to predict the Remaining Useful Life (RUL) of jet engines using data from engine sensors. The prediction model is built using the CatBoost library, which provides powerful gradient boosting algorithms, and relies on `NumPy` for numerical operations.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Model Training and Evaluation](#model-training-and-evaluation)
4. [Results](#results)
5. [Troubleshooting](#troubleshooting)
6. [References](#references)

## Project Overview

The goal of this project is to predict the Remaining Useful Life (RUL) of jet engines based on sensor data. RUL prediction is crucial in maintenance scheduling, optimizing operational efficiency, and preventing unexpected engine failures.

## Data Description

The dataset used in this project is the **NASA Turbofan Jet Engine Data Set**, available on [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps). This dataset consists of time-series data collected from multiple jet engines under different operating conditions. Each engine has its own set of sensor readings over time until a failure occurs, providing a comprehensive view of the engine's health and performance.

Key features in the dataset include:

- **Temperature**: Indicates various temperature readings from different parts of the engine.
- **Pressure**: Records pressure levels at different engine components.
- **Vibration**: Measures vibration levels, which can indicate mechanical wear or imbalances.
- **Fuel Flow**: Tracks fuel consumption, which can affect and reflect the engine's operational state.
- **Other Operational Metrics**: Includes additional sensor readings related to the engine's performance and environmental conditions.

These features are used to train a model to predict the Remaining Useful Life (RUL) of each engine, allowing for proactive maintenance and operational decisions.

## Model Training and Evaluation

The CatBoost model was trained on historical engine data, using various features that indicate engine health and performance. The model's performance was evaluated using several metrics to ensure accuracy and reliability in predicting RUL.

### Key Model Features:

- **Gradient Boosting Algorithm:** CatBoost is used for its efficiency with categorical data and its ability to handle numerical features.
- **Data Normalization:** Sensor data was normalized to improve model performance.

### Evaluation Metrics:

- **Classification Report:**
``````
           precision    recall  f1-score   support

       0       0.93      0.94      0.94      3699
       1       0.71      0.70      0.71      1247
       2       0.90      0.88      0.89      1244

accuracy                           0.88      6190
macro avg      0.85      0.84      0.84      6190
weighted avg   0.88      0.88      0.88      6190
``````

- **Mean Absolute Error (MAE)**: 0.12
- **Root Mean Squared Error (RMSE)**: 0.35
- **R2 Score**: 0.81

## Results

The model demonstrated strong predictive performance with an accuracy of 88% and an R2 score of 0.81 on the test set, indicating its effectiveness in predicting the Remaining Useful Life of jet engines based on sensor data.

## References

- CatBoost Documentation: [CatBoost GitHub](https://github.com/catboost/catboost)
- NumPy Documentation: [NumPy GitHub](https://github.com/numpy/numpy)
- Data Source: [NASA Turbofan Jet Engine Data Set on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)



