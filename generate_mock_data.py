#!/usr/bin/env python3
"""
One-time mock data generation for loan approval ML pipeline.
Run this script once to generate training and test datasets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

def generate_loan_data(n_samples=10000, random_state=42):
    """Generate synthetic loan application data."""
    np.random.seed(random_state)
    
    # Generate features
    data = {
        'age': np.random.normal(40, 12, n_samples).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 500000),
        'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
        'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 0.8,
        'loan_amount': np.random.lognormal(10, 0.6, n_samples).clip(5000, 200000),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.4, 0.3, 0.3]),
        'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'other'], 
                                       n_samples, p=[0.4, 0.25, 0.2, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable with realistic logic
    def determine_approval(row):
        score = 0
        
        # Credit score influence (most important)
        if row['credit_score'] >= 750: score += 40
        elif row['credit_score'] >= 650: score += 20
        elif row['credit_score'] >= 550: score += 5
        else: score -= 20
        
        # Income influence
        if row['income'] >= 80000: score += 25
        elif row['income'] >= 50000: score += 15
        elif row['income'] >= 30000: score += 5
        else: score -= 10
        
        # Debt to income ratio
        if row['debt_to_income'] <= 0.2: score += 20
        elif row['debt_to_income'] <= 0.4: score += 10
        elif row['debt_to_income'] <= 0.6: score -= 5
        else: score -= 20
        
        # Employment length
        if row['employment_length'] >= 5: score += 15
        elif row['employment_length'] >= 2: score += 10
        elif row['employment_length'] >= 1: score += 5
        else: score -= 5
        
        # Home ownership
        if row['home_ownership'] == 'OWN': score += 15
        elif row['home_ownership'] == 'MORTGAGE': score += 10
        
        # Loan amount to income ratio
        loan_to_income = row['loan_amount'] / row['income']
        if loan_to_income <= 0.3: score += 10
        elif loan_to_income <= 0.5: score += 5
        elif loan_to_income > 1.0: score -= 15
        
        # Add some randomness
        score += np.random.normal(0, 10)
        
        return 1 if score > 50 else 0
    
    df['loan_approved'] = df.apply(determine_approval, axis=1)
    
    # Round numerical columns (use float64 for MLflow compatibility)
    df['age'] = df['age'].round().astype('float64')
    df['income'] = df['income'].round().astype('float64')
    df['employment_length'] = df['employment_length'].round(1).astype('float64')
    df['credit_score'] = df['credit_score'].round().astype('float64')
    df['debt_to_income'] = df['debt_to_income'].round(3).astype('float64')
    df['loan_amount'] = df['loan_amount'].round().astype('float64')
    
    return df

def main():
    """Generate and save mock data."""
    print("Generating mock loan application data...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    df = generate_loan_data(n_samples=10000)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['loan_approved'])
    
    # Save datasets
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    df.to_csv('data/full_data.csv', index=False)
    
    # Print statistics
    print(f"Generated {len(df)} total samples")
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Approval rate: {df['loan_approved'].mean():.2%}")
    
    # Save feature info
    feature_info = {
        'numerical_features': ['age', 'income', 'employment_length', 'credit_score', 'debt_to_income', 'loan_amount'],
        'categorical_features': ['home_ownership', 'loan_purpose'],
        'target': 'loan_approved',
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'approval_rate': float(df['loan_approved'].mean())
    }
    
    with open('data/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("\nData generation completed!")
    print("Files created:")
    print("  - data/train_data.csv")
    print("  - data/test_data.csv") 
    print("  - data/full_data.csv")
    print("  - data/feature_info.json")
    
    print(f"\nFeature summary:")
    print(df.describe())

if __name__ == "__main__":
    main()