# -*- coding: utf-8 -*-
"""
NBA Playing Time Prediction Model
Author: Aashish Arya
Optimized ML Pipeline with Feature Engineering, Bayesian Hyperparameter Tuning (Hyperopt), and Stacking Ensembles.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Hyperopt Imports
from hyperopt import hp, tpe, fmin, Trials, space_eval, STATUS_OK

# 1. Feature Engineering (Domain-Specific Features)
def engineer_features(df):
    """
    Creates custom, normalized domain-specific features from raw player stats.
    """
    df = df.copy()
    
    # Avoid division by zero by replacing zero games with 1 (where relevant)
    df['G'] = df['G'].replace(0, 1)
    
    # Normalized Per-Game Metrics
    df['PTS_per_G'] = df['PTS'] / df['G']
    df['AST_per_G'] = df['AST'] / df['G']
    df['TOV_per_G'] = df['TOV'] / df['G']
    df['FGA_per_G'] = df['FGA'] / df['G']
    df['FTA_per_G'] = df['FTA'] / df['G']
    df['TRB_per_G'] = df['TRB'] / df['G']
    df['STL_per_G'] = df['STL'] / df['G']
    df['BLK_per_G'] = df['BLK'] / df['G']
    df['PF_per_G'] = df['PF'] / df['G']
    
    # Starter Ratio (fraction of games started - highly correlated with minutes)
    df['GS_Ratio'] = df['GS'] / df['G']
    
    # True Shooting Percentage (TS%) - measures scoring efficiency
    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    denom_ts = 2 * (df['FGA'] + 0.44 * df['FTA'])
    df['TS%'] = df['PTS'] / denom_ts.replace(0, 1)
    
    # Assist-to-Turnover Ratio
    df['AST_TOV_ratio'] = df['AST'] / df['TOV'].replace(0, 1)
    
    return df

def build_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Builds a ColumnTransformer for numeric and categorical feature preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

# Interactive CLI Predictor
def predict_custom_player(model, numeric_cols, categorical_cols):
    print("\n" + "="*60)
    print("=== NBA PLAYER MINUTES PREDICTOR (INTERACTIVE TOOL) ===")
    print("="*60)
    print("Enter the following player stats to forecast expected court minutes:")
    
    try:
        age = float(input("Player Age (e.g., 25): ") or 25)
        g = float(input("Games Played (e.g., 70): ") or 70)
        gs = float(input("Games Started (e.g., 50): ") or 50)
        pts = float(input("Points per Game (e.g., 15.4): ") or 15.4)
        ast = float(input("Assists per Game (e.g., 4.2): ") or 4.2)
        trb = float(input("Rebounds per Game (e.g., 5.1): ") or 5.1)
        fga = float(input("Field Goal Attempts per game (e.g., 12.0): ") or 12.0)
        fta = float(input("Free Throw Attempts per game (e.g., 3.5): ") or 3.5)
        fg_pct = float(input("Field Goal % (e.g., 0.45): ") or 0.45)
        three_pct = float(input("3-Point % (e.g., 0.36): ") or 0.36)
        ft_pct = float(input("Free Throw % (e.g., 0.80): ") or 0.80)
        tov = float(input("Turnovers per Game (e.g., 2.1): ") or 2.1)
        stl = float(input("Steals per Game (e.g., 1.1): ") or 1.1)
        blk = float(input("Blocks per Game (e.g., 0.5): ") or 0.5)
        pf = float(input("Personal Fouls per Game (e.g., 2.2): ") or 2.2)
        pos = input("Position (e.g., PG, SG, SF, PF, C): ") or "SG"
        tm = input("Team Abbr (e.g., LAL, BOS, MIA): ") or "LAL"
        
        # Build raw dict structure matching dataset columns
        custom_data = pd.DataFrame([{
            'Age': age, 'G': g, 'GS': gs,
            'FG': pts * fg_pct / 2.0, 'FGA': fga * g, 'FG%': fg_pct,
            '3P': fga * three_pct / 3.0, '3PA': fga * 0.3 * g, '3P%': three_pct,
            '2P': (fga - fga * 0.3) * fg_pct, '2PA': fga * 0.7 * g, '2P%': fg_pct,
            'eFG%': fg_pct + 0.5 * three_pct, 'FT': fta * ft_pct, 'FTA': fta * g,
            'FT%': ft_pct, 'ORB': trb * 0.3 * g, 'DRB': trb * 0.7 * g,
            'TRB': trb * g, 'AST': ast * g, 'STL': stl * g, 'BLK': blk * g,
            'TOV': tov * g, 'PF': pf * g, 'PTS': pts * g, 'Pos': pos, 'Tm': tm
        }])
        
        # Run engineered feature pipeline
        custom_features = engineer_features(custom_data)
        
        # Predict minutes
        predicted_mp = model.predict(custom_features)[0]
        
        print("\n" + "-"*40)
        print(f"Predicted Expected Playing Time: {predicted_mp:.2f} Minutes/Game")
        print("-"*40 + "\n")
    except Exception as e:
        print(f"Error compiling manual player data: {e}")

def main():
    print("Initializing NBA Playing Time Prediction Pipeline...")
    
    # 2. Loading Datasets
    train_url = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/NBA_2021.csv"
    test_url = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/NBA_2020.csv"
    
    print("Downloading training data from 2021 season...")
    data_train = pd.read_csv(train_url, index_col="Player")
    print("Downloading testing data from 2020 season...")
    data_test = pd.read_csv(test_url, index_col="Player")
    
    # Handle NaNs in target variable
    data_train.dropna(axis=0, subset=['MP'], inplace=True)
    data_test.dropna(axis=0, subset=['MP'], inplace=True)
    
    # Define Target and Split features
    y_train = data_train["MP"]
    X_train = data_train.drop(["MP"], axis=1)
    
    y_test = data_test["MP"]
    X_test = data_test.drop(["MP"], axis=1)
    
    # Apply Feature Engineering
    print("Engineering domain-specific basketball features...")
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    
    # Feature columns specification
    categorical_cols = ['Pos', 'Tm']
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    # Setup preprocessing
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # 3. Model Suite Setup
    print("\n--- Model Training & Parameter Tuning ---")
    
    # Baseline: Linear Regression
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    print(f"Linear Regression Baseline -> R2: {r2_score(y_test, lr_preds):.4f} | MAE: {mean_absolute_error(y_test, lr_preds):.2f} mins")
    
    # Define hyperopt objectives and spaces
    print("\nTuning models using Hyperopt (Bayesian Optimization)...")
    
    # --- Random Forest Tuning ---
    rf_space = {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'max_depth': hp.choice('max_depth', [10, 15, None]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10])
    }
    
    def rf_objective(params):
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42
        )
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        # Maximize R2 (equivalent to minimizing negative R2)
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
        return {'loss': -np.mean(scores), 'status': STATUS_OK}
    
    print("Tuning Random Forest Regressor...")
    rf_trials = Trials()
    best_rf_idx = fmin(
        fn=rf_objective,
        space=rf_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=rf_trials
    )
    best_rf_params = space_eval(rf_space, best_rf_idx)
    print(f"Best RF Parameters: {best_rf_params}")
    
    # Train best RF Model
    best_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=best_rf_params['n_estimators'],
            max_depth=best_rf_params['max_depth'],
            min_samples_split=best_rf_params['min_samples_split'],
            random_state=42
        ))
    ])
    best_rf.fit(X_train, y_train)
    rf_preds = best_rf.predict(X_test)
    print(f"Tuned Random Forest Regressor -> R2: {r2_score(y_test, rf_preds):.4f} | MAE: {mean_absolute_error(y_test, rf_preds):.2f} mins")
    
    # --- HistGradientBoosting Tuning ---
    hgb_space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'max_iter': hp.choice('max_iter', [100, 200, 300]),
        'max_depth': hp.choice('max_depth', [3, 5, 8, 12])
    }
    
    def hgb_objective(params):
        model = HistGradientBoostingRegressor(
            learning_rate=params['learning_rate'],
            max_iter=params['max_iter'],
            max_depth=params['max_depth'],
            random_state=42
        )
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
        return {'loss': -np.mean(scores), 'status': STATUS_OK}
    
    print("Tuning HistGradientBoosting Regressor...")
    hgb_trials = Trials()
    best_hgb_idx = fmin(
        fn=hgb_objective,
        space=hgb_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=hgb_trials
    )
    best_hgb_params = space_eval(hgb_space, best_hgb_idx)
    print(f"Best HGB Parameters: {best_hgb_params}")
    
    # Train best HGB Model
    best_hgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', HistGradientBoostingRegressor(
            learning_rate=best_hgb_params['learning_rate'],
            max_iter=best_hgb_params['max_iter'],
            max_depth=best_hgb_params['max_depth'],
            random_state=42
        ))
    ])
    best_hgb.fit(X_train, y_train)
    hgb_preds = best_hgb.predict(X_test)
    print(f"Tuned HistGradientBoosting    -> R2: {r2_score(y_test, hgb_preds):.4f} | MAE: {mean_absolute_error(y_test, hgb_preds):.2f} mins")
    
    # 4. Ensemble Stacking Regressor
    print("\nTraining Ensemble Stacking Regressor...")
    stacking_regressor = StackingRegressor(
        estimators=[
            ('rf', best_rf.named_steps['model']),
            ('hgb', best_hgb.named_steps['model'])
        ],
        final_estimator=RidgeCV()
    )
    
    stacking_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking_regressor)
    ])
    stacking_pipeline.fit(X_train, y_train)
    stack_preds = stacking_pipeline.predict(X_test)
    print(f"Stacked Hybrid Ensemble       -> R2: {r2_score(y_test, stack_preds):.4f} | MAE: {mean_absolute_error(y_test, stack_preds):.2f} mins")
    
    # 5. Extract Feature Importances (Random Forest)
    print("\n--- Feature Importance Analysis ---")
    best_rf_model = best_rf.named_steps['model']
    cat_encoder = best_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_features = numeric_cols + cat_feature_names
    
    importances = best_rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    print("Top 10 Most Important Features Driving Predicted Minutes:")
    for rank, idx in enumerate(indices, 1):
        print(f" {rank}. {all_features[idx]:<20} Importance Score: {importances[idx]:.4f}")
        
    # 6. Plotting Predictions vs Actuals (Save to workspace)
    print("\nGenerating model prediction plot...")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=stack_preds, scatter_kws={'alpha': 0.6, 'color': '#1d3557'}, line_kws={'color': '#e63946', 'lw': 2})
    plt.title('Stacked Hybrid Model: Predicted vs Actual Minutes Played', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Actual Minutes Played (MP)', fontsize=12)
    plt.ylabel('Predicted Minutes Played (MP)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plot_path = "model_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as: {plot_path}")
    
    # 7. Interactive Prediction Call
    # Only triggers when run directly in an interactive shell
    if sys.stdin.isatty():
        predict_custom_player(stacking_pipeline, numeric_cols, categorical_cols)
    else:
        print("\nInteractive predictor skipped (non-TTY environment).")

if __name__ == "__main__":
    main()
