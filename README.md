# 🏀 NBA Player Performance Predictor (Minutes Played)

A machine learning project that predicts the **Minutes Played (MP)** per game for NBA players based on their season statistics.

## 📌 Project Overview
This project applies supervised machine learning techniques to analyze NBA player data from the **2021 season** to predict their playing time for the **2020 season**.

The model leverages historical statistics, feature engineering, and advanced regression-based ensemble approaches to estimate how long a player stays on the court.

## 🧠 Problem Statement
Given a set of player attributes and performance metrics, the goal is to accurately predict the **Minutes Played (MP)** per game.

* **Input Features:** Age, Team, Position, Shooting %, Rebounds, Assists, and Engineered Features (TS%, Starter Ratio, etc.).
* **Target Variable:** Minutes Played (MP)

## ⚙️ Tech Stack

### Language
* **Python**

### Libraries
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn

### ML Techniques & Architectures
* **Feature Engineering:** Domain-specific calculations (True Shooting %, Per-game metrics, Assist-to-Turnover ratio, Starter Ratio).
* **ColumnTransformer Pipeline:** Unified preprocessing handling numerical features (scaling & mean imputation) and categorical features (One-Hot Encoding) preventing data leakage.
* **Hyperparameter Tuning:** Hyperparameter optimization via `GridSearchCV` on cross-validated training folds.
* **Stacked Generalization (Stacking):** Ensembles optimized Random Forest and HistGradientBoosting Regressors using a `RidgeCV` meta-learner to achieve high performance.

## 📊 Key Features
* **Automated Data Pipeline:** Unified pipeline architecture ensures robust transformations and prevents train-test leakage.
* **Advanced Domain Feature Engineering:**
    * **True Shooting Percentage (TS%):** Measures scoring efficiency.
    * **Starter Ratio:** Computes the fraction of games started, indicating structural team status.
    * **Normalized Stats:** Custom per-game normalization for counting metrics.
* **Tuned Ensemble Stacking:** Leverages `StackingRegressor` to integrate multiple strong models.
* **Feature Importance Analysis:** Ranks and prints the most influential metrics affecting a player's court minutes.
* **Interactive CLI Tool:** Provides a terminal interface for typing custom player profiles and predicting expected playing time in real-time.

## 🧪 Model Workflow
1. **Load Data:** Ingest NBA 2021 (Train) and NBA 2020 (Test) datasets.
2. **Feature Engineering:** Compute True Shooting %, Starter Ratio, and per-game normalization.
3. **Preprocessing:** Fit `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` via `ColumnTransformer` inside pipelines.
4. **Tuning:** Find optimal base model parameters via `GridSearchCV`.
5. **Stacking Ensemble:** Combine the optimized base regressors under a meta-learner.
6. **Evaluation:** Output metrics on unseen 2020 testing data.
7. **Visualization:** Plot Actual vs. Predicted values using `Seaborn.regplot`.

## 📈 Results

Our optimized pipeline and stacked hybrid ensemble significantly outperform the baseline linear model:

| Model | MAE (Lower is Better) | R² Score (Higher is Better) | Performance Description |
| :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 1.89 mins | 92.76% | Standard linear baseline. |
| **Tuned Random Forest Regressor** | 1.77 mins | 93.55% | Captures non-linear feature splits. |
| **Tuned HistGradientBoosting** | 1.66 mins | 94.17% | High-performance gradient boosting. |
| **Stacked Hybrid Ensemble (Meta)** | **1.65 mins** | **94.20%** | **Best performance. Combines model strengths.** |

### Top 5 Influential Features:
1. **PTS (Points):** Strongest indicator of total volume/impact.
2. **FGA (Field Goal Attempts):** Indicates shot volume and usage.
3. **GS_Ratio (Starter Ratio):** Represents starter status.
4. **FG (Field Goals):** Scoring completion volume.
5. **GS (Games Started):** Raw started count.
