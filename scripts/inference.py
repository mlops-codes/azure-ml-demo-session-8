"""
Azure ML inference script for loan approval model endpoint.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List

def init():
    """Initialize the model for inference."""
    global model, feature_info, scalers
    
    # Get model path
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR', ''), 'model.pkl')
    
    # Load model
    model = joblib.load(model_path)
    print("Model loaded successfully")
    
    # Load feature info and preprocessing objects
    models_dir = os.path.join(os.getenv('AZUREML_MODEL_DIR', ''), 'preprocessing')
    
    with open(os.path.join(models_dir, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    # Load scalers and encoders
    scalers = {
        'scaler': joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    }
    
    # Load label encoders
    for col in feature_info['categorical_features']:
        encoder_path = os.path.join(models_dir, f'label_encoder_{col}.pkl')
        if os.path.exists(encoder_path):
            scalers[f'encoder_{col}'] = joblib.load(encoder_path)
    
    print("Preprocessing objects loaded successfully")

def create_features(df):
    """Create additional features from raw input data."""
    df = df.copy()
    
    # Create ratio features
    df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
    df['credit_score_to_age_ratio'] = df['credit_score'] / df['age']
    
    # Create age categories
    df['age_category'] = pd.cut(df['age'], 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=['young', 'adult', 'middle_age', 'mature', 'senior'])
    
    # Create income categories
    df['income_category'] = pd.cut(df['income'], 
                                  bins=[0, 30000, 50000, 80000, np.inf], 
                                  labels=['low', 'medium', 'high', 'very_high'])
    
    # Create credit score categories
    df['credit_category'] = pd.cut(df['credit_score'], 
                                  bins=[0, 550, 650, 750, np.inf], 
                                  labels=['poor', 'fair', 'good', 'excellent'])
    
    # Boolean features (use float64 to handle potential missing values)
    df['high_income'] = (df['income'] >= 75000).astype('float64')
    df['excellent_credit'] = (df['credit_score'] >= 750).astype('float64')
    df['low_debt_ratio'] = (df['debt_to_income'] <= 0.3).astype('float64')
    df['experienced_employment'] = (df['employment_length'] >= 5).astype('float64')
    
    return df

def preprocess_input(data):
    """Preprocess input data for prediction."""
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Create features
    df = create_features(df)
    
    # Handle missing values
    numerical_features = feature_info['numerical_features']
    categorical_features = feature_info['categorical_features']
    
    # Fill missing values with defaults
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if len(df) > 1 else 0)
    
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    # Scale numerical features
    scaler = scalers['scaler']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    # Encode categorical features
    for col in categorical_features:
        encoder_key = f'encoder_{col}'
        if encoder_key in scalers and col in df.columns:
            encoder = scalers[encoder_key]
            # Handle unseen categories
            df[col] = df[col].astype(str)
            mask = df[col].isin(encoder.classes_)
            df.loc[~mask, col] = 'unknown' if 'unknown' in encoder.classes_ else encoder.classes_[0]
            df[col] = encoder.transform(df[col])
    
    # Ensure all required features are present
    for feature in feature_info['all_features']:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Select only model features in correct order
    X = df[feature_info['all_features']]
    
    return X

def run(raw_data):
    """Run inference on input data."""
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Format results
        results = []
        for i in range(len(X)):
            result = {
                'loan_approved': int(predictions[i]),
                'approval_probability': float(probabilities[i, 1]) if probabilities is not None else float(predictions[i])
            }
            
            # Add confidence interpretation
            if probabilities is not None:
                prob = probabilities[i, 1]
                if prob >= 0.8:
                    result['confidence'] = 'high'
                elif prob >= 0.6:
                    result['confidence'] = 'medium'
                else:
                    result['confidence'] = 'low'
            
            results.append(result)
        
        return json.dumps(results)
        
    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        return json.dumps({"error": error_msg})

def predict_single(applicant_data):
    """Helper function for single prediction (for testing)."""
    raw_data = json.dumps([applicant_data])
    result = run(raw_data)
    return json.loads(result)[0]

# Test function for local development
def test_inference():
    """Test the inference function locally."""
    # Sample test data
    test_data = {
        "age": 35,
        "income": 65000,
        "employment_length": 5.2,
        "credit_score": 720,
        "debt_to_income": 0.25,
        "loan_amount": 25000,
        "home_ownership": "MORTGAGE",
        "loan_purpose": "debt_consolidation"
    }
    
    # Convert to JSON
    raw_data = json.dumps([test_data])
    
    # Run inference
    result = run(raw_data)
    
    print("Test input:")
    print(json.dumps(test_data, indent=2))
    print("\nTest output:")
    print(result)

if __name__ == "__main__":
    # For local testing
    print("Testing inference script...")
    
    # Initialize (would need actual model files)
    # init()
    
    # Run test
    # test_inference()