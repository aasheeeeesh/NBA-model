# 🏀 NBA Player Performance Predictor (Minutes Played)

A machine learning project that predicts the **Minutes Played (MP)** per game for NBA players based on their season statistics.

## 📌 Project Overview
This project applies supervised machine learning techniques to analyze NBA player data from the **2021 season** to predict their playing time for the **2020 season**.

The model leverages historical statistics, feature engineering, and regression-based approaches to estimate how long a player stays on the court.

## 🧠 Problem Statement
Given a set of player attributes and performance metrics, the goal is to accurately predict the **Minutes Played (MP)** per game.

* **Input Features:** Age, Team, Position, Shooting %, Rebounds, Assists, etc.
* **Target Variable:** Minutes Played (MP)

## ⚙️ Tech Stack

### Language
* **Python**

### Libraries
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn

### ML Techniques
* **Data Preprocessing:** Imputation, One-Hot Encoding
* **Pipeline Construction:** sklearn.pipeline
* **Algorithms:** Linear Regression & Random Forest Regressor
* **Metrics:** Mean Absolute Error (MAE), R² Score

## 📊 Key Features
* **Automated Data Pipeline:** Handles missing values via Imputation and manages categorical data (Teams/Positions) using One-Hot Encoding.
* **Model Comparison:** Performs a head-to-head comparison between a baseline Linear Regression model and an ensemble Random Forest model.
* **Leakage Prevention:** Ensures fair evaluation by strictly separating Training (2021) and Testing (2020) datasets.
* **Custom Accuracy Metric:** Implements a logic-based accuracy check to calculate the percentage of predictions that fall within a **±1 minute** tolerance of the actual time.

## 🧪 Model Workflow
1. **Load Data:** Ingest NBA 2021 (Train) and NBA 2020 (Test) datasets.
2. **Preprocessing:**
    * Drop rows with missing target values.
    * Separate numeric and categorical features.
    * Apply `SimpleImputer` and `OneHotEncoder` via a `ColumnTransformer`.
3. **Training:** Fit models on the 2021 player data.
4. **Evaluation:** Test the models on unseen 2020 data.
5. **Visualization:** Plot Actual vs. Predicted values using `Seaborn.regplot`.

## 📈 Results
The project compared two different algorithms. The **Random Forest model** significantly outperformed the Linear Regression baseline, capturing the complex non-linear relationships in player data.

| Model | MAE (Lower is Better) | R² Score (Higher is Better) | Accuracy Description |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 2.50 mins | 89.2% | Good baseline, but higher error margin. |
| **Random Forest** | **1.76 mins** | **93.9%** | Excellent. Captures complex patterns. |
