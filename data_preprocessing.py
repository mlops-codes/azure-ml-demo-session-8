#!/usr/bin/env python3
"""
Data preprocessing and feature engineering for loan approval ML pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


# -----------------------
# IO helpers
# -----------------------

def load_data(path='data/train_data.csv'):
    """Load CSV and report size."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")
    return df


# -----------------------
# Feature engineering
# -----------------------

def _safe_ratio(num, den):
    den = den.replace(0, np.nan)
    r = num / den
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing data."""
    df = df.copy()

    # Robust ratio features
    df['loan_to_income_ratio'] = _safe_ratio(df['loan_amount'], df['income'])
    df['credit_score_to_age_ratio'] = _safe_ratio(df['credit_score'], df['age'])

    # Category buckets (pd.cut produces Categorical; we'll convert to string later)
    df['age_category'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 45, 55, 100],
        labels=['young', 'adult', 'middle_age', 'mature', 'senior'],
        include_lowest=True
    )
    df['income_category'] = pd.cut(
        df['income'],
        bins=[0, 30000, 50000, 80000, np.inf],
        labels=['low', 'medium', 'high', 'very_high'],
        include_lowest=True
    )
    df['credit_category'] = pd.cut(
        df['credit_score'],
        bins=[0, 550, 650, 750, np.inf],
        labels=['poor', 'fair', 'good', 'excellent'],
        include_lowest=True
    )

    # Boolean features (use float64 to handle potential missing values)
    df['high_income'] = (df['income'] >= 75000).astype('float64')
    df['excellent_credit'] = (df['credit_score'] >= 750).astype('float64')
    df['low_debt_ratio'] = (df['debt_to_income'] <= 0.3).astype('float64')
    df['experienced_employment'] = (df['employment_length'] >= 5).astype('float64')

    return df


# -----------------------
# Preprocessing
# -----------------------

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def _add_unknown_to_label_encoder(le: LabelEncoder):
    """Make sure 'unknown' is a valid class for inference-time mapping."""
    if 'unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'unknown')
    return le

def preprocess_data(df: pd.DataFrame, fit_scalers: bool = True, scalers_path: str = 'models'):
    """
    Preprocess data for model training & inference.

    - Creates engineered features
    - Handles NAs (numeric median, categoricals -> 'unknown')
    - Scales numeric features
    - Encodes categoricals with LabelEncoder (safe for unseen via 'unknown')
    """
    _ensure_dir(scalers_path)

    # Create features
    df = create_features(df)

    # Separate features/target
    target_col = 'loan_approved'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]

    # Feature lists
    numerical_features = [
        'age', 'income', 'employment_length', 'credit_score',
        'debt_to_income', 'loan_amount',
        'loan_to_income_ratio', 'credit_score_to_age_ratio'
    ]
    categorical_features = [
        'home_ownership', 'loan_purpose',
        'age_category', 'income_category', 'credit_category'
    ]
    boolean_features = [
        'high_income', 'excellent_credit', 'low_debt_ratio', 'experienced_employment'
    ]

    # Validate columns exist
    missing_cols = [c for c in (numerical_features + categorical_features + boolean_features) if c not in X.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns in data: {missing_cols}")

    # ---- Handle missing values ----
    # Numerics -> median
    X[numerical_features] = X[numerical_features].apply(lambda s: s.fillna(s.median()))

    # Categoricals: convert to pandas 'string' dtype, then fill with 'unknown'
    # (avoids CategoricalDtype fillna bugs)
    X[categorical_features] = X[categorical_features].astype('string')
    X[categorical_features] = X[categorical_features].fillna('unknown')

    # ---- Scale numerics ----
    scaler_path = os.path.join(scalers_path, 'scaler.pkl')
    if fit_scalers:
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X[numerical_features] = scaler.transform(X[numerical_features])

    # ---- Encode categoricals with LabelEncoder per column ----
    label_encoders = {}
    for col in categorical_features:
        le_path = os.path.join(scalers_path, f'label_encoder_{col}.pkl')

        if fit_scalers:
            le = LabelEncoder()
            # ensure strings and include 'unknown' in training set
            col_vals = X[col].astype(str).fillna('unknown')
            le.fit(col_vals)
            _add_unknown_to_label_encoder(le)
            X[col] = le.transform(col_vals)
            label_encoders[col] = le
        else:
            le = joblib.load(le_path)
            # map unseen categories to 'unknown'
            vals = X[col].astype(str)
            mask = vals.isin(le.classes_)
            vals = vals.where(mask, 'unknown')
            # ensure encoder has 'unknown'
            _add_unknown_to_label_encoder(le)
            X[col] = le.transform(vals)

    # Persist label encoders on fit
    if fit_scalers:
        for col, le in label_encoders.items():
            joblib.dump(le, os.path.join(scalers_path, f'label_encoder_{col}.pkl'))

    # Feature info
    feature_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'boolean_features': boolean_features,
        'all_features': list(X.columns),
        'target': target_col
    }

    if fit_scalers:
        with open(os.path.join(scalers_path, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)

    print(f"Preprocessed {len(X)} samples with {len(X.columns)} features")
    return X, y, feature_info


# -----------------------
# Train/Val/Test pipelines
# -----------------------

def prepare_data_for_training(data_path='data/train_data.csv', output_dir='processed_data'):
    """Main preprocessing pipeline for training data."""
    _ensure_dir(output_dir)

    df = load_data(data_path)
    X, y, feature_info = preprocess_data(df, fit_scalers=True, scalers_path='models')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = X_train.copy()
    train_data[feature_info['target']] = y_train

    val_data = X_val.copy()
    val_data[feature_info['target']] = y_val

    train_data.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val_processed.csv'), index=False)

    print(f"Training data: {len(train_data)} samples")
    print(f"Validation data: {len(val_data)} samples")
    print(f"Training approval rate: {y_train.mean():.2%}")
    print(f"Validation approval rate: {y_val.mean():.2%}")

    return train_data, val_data, feature_info


def preprocess_test_data(test_path='data/test_data.csv', output_dir='processed_data'):
    """Preprocess test data using saved scalers/encoders."""
    _ensure_dir(output_dir)

    df = load_data(test_path)
    X, y, feature_info = preprocess_data(df, fit_scalers=False, scalers_path='models')

    test_data = X.copy()
    test_data[feature_info['target']] = y

    test_data.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)

    print(f"Test data: {len(test_data)} samples")
    print(f"Test approval rate: {y.mean():.2%}")

    return test_data


# -----------------------
# CLI
# -----------------------

def main():
    print("Starting data preprocessing...")
    _ensure_dir('processed_data')
    _ensure_dir('models')

    # Train/val
    train_data, val_data, feature_info = prepare_data_for_training()

    # Test
    preprocess_test_data()

    print("\nData preprocessing completed!")
    print("Files created:")
    print("  - processed_data/train_processed.csv")
    print("  - processed_data/val_processed.csv")
    print("  - processed_data/test_processed.csv")
    print("  - models/scaler.pkl")
    print("  - models/feature_info.json")
    print("  - models/label_encoder_*.pkl")


if __name__ == "__main__":
    main()