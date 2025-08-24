"""
Azure ML Feature Store management for loan approval model.
Handles feature group creation, feature ingestion, and retrieval.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

class AzureMLFeatureStore:
    def __init__(self, config_path='config.json'):
        """Initialize Azure ML Feature Store client."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize ML client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.config['azure_ml']['subscription_id'],
            resource_group_name=self.config['azure_ml']['resource_group'],
            workspace_name=self.config['azure_ml']['workspace_name']
        )
        
        # Feature store config
        self.feature_config = self.config.get('feature_store', {})
        self.feature_group_name = self.feature_config.get('feature_group_name', 'loan-approval-features')
        
        # Test connection (but don't fail if workspace doesn't exist)
        try:
            workspace = self.ml_client.workspaces.get(self.config['azure_ml']['workspace_name'])
            print(f"✅ Connected to Azure ML workspace: {workspace.name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to workspace: {self.config['azure_ml']['workspace_name']}")
            print(f"   Error: {str(e)}")
            print("   Feature store will work in local-only mode")

    def create_feature_set(self, data_path='processed_data/train_processed.csv'):
        """Create feature set metadata (simplified for demo)."""
        print(f"Creating feature set metadata: {self.feature_group_name}")
        
        # Load processed data to understand schema
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found: {data_path}")
            print("Creating feature set metadata without schema validation")
            
            # Create basic feature set info
            feature_set_info = {
                "name": self.feature_group_name,
                "version": "1",
                "description": self.feature_config.get('description', 'Loan approval features'),
                "status": "created_metadata_only",
                "created_at": datetime.utcnow().isoformat()
            }
        else:
            df = pd.read_csv(data_path)
            print(f"Loaded {len(df)} samples from {data_path}")
            
            # Create feature set metadata
            target_col = 'loan_approved'
            features = []
            
            for col in df.columns:
                if col == target_col:
                    continue
                    
                # Determine feature type
                if df[col].dtype in ['int64', 'float64']:
                    feature_type = "numeric"
                else:
                    feature_type = "categorical"
                
                features.append({
                    "name": col,
                    "type": feature_type,
                    "description": f"Feature: {col}",
                    "nullable": bool(df[col].isnull().any())
                })
            
            feature_set_info = {
                "name": self.feature_group_name,
                "version": "1", 
                "description": self.feature_config.get('description', 'Loan approval features'),
                "features": features,
                "total_features": len(features),
                "sample_count": len(df),
                "created_at": datetime.utcnow().isoformat(),
                "status": "created_with_schema",
                "tags": {
                    "created_by": "github_actions",
                    "environment": self.config.get('environment', 'dev'),
                    "model_type": "loan_approval"
                }
            }
        
        # Save feature set info locally
        os.makedirs('feature_store', exist_ok=True)
        with open('feature_store/feature_set_info.json', 'w') as f:
            json.dump(feature_set_info, f, indent=2)
        
        print(f"✅ Feature set metadata created: {feature_set_info['name']}")
        print(f"   Features: {feature_set_info.get('total_features', 'unknown')}")
        print(f"   Status: {feature_set_info['status']}")
        
        return feature_set_info

    def upload_features(self, data_path='processed_data'):
        """Upload feature data to Azure ML datastore."""
        print("Uploading feature data to Azure ML...")
        
        try:
            # Upload processed data
            from azure.ai.ml.entities import Data
            from azure.ai.ml.constants import AssetTypes
            
            feature_data = Data(
                name=f"{self.feature_group_name}-data",
                path=data_path,
                type=AssetTypes.URI_FOLDER,
                description="Processed loan approval feature data"
            )
            
            uploaded_data = self.ml_client.data.create_or_update(feature_data)
            print(f"✅ Feature data uploaded: {uploaded_data.name}:{uploaded_data.version}")
            
            return uploaded_data
            
        except Exception as e:
            print(f"⚠️  Warning: Could not upload feature data to Azure ML: {str(e)}")
            print("This might be because:")
            print("  - Azure ML workspace doesn't exist")
            print("  - Insufficient permissions")
            print("  - Network connectivity issues")
            
            # Create a mock upload response for demo purposes
            mock_data = {
                "name": f"{self.feature_group_name}-data",
                "version": "1",
                "status": "upload_failed_but_continuing",
                "error": str(e)
            }
            
            print("📝 Continuing with local feature store metadata only")
            return mock_data

    def get_feature_data(self, version: Optional[str] = None) -> pd.DataFrame:
        """Retrieve feature data from feature store."""
        try:
            # Get feature set
            feature_set = self.ml_client.feature_sets.get(
                name=self.feature_group_name,
                version=version or "1"
            )
            
            print(f"Retrieved feature set: {feature_set.name}:{feature_set.version}")
            
            # For this demo, return the locally processed data
            # In a real scenario, you'd query the feature store directly
            processed_data_path = 'processed_data/train_processed.csv'
            if os.path.exists(processed_data_path):
                df = pd.read_csv(processed_data_path)
                print(f"Loaded {len(df)} feature records")
                return df
            else:
                raise FileNotFoundError("Processed data not found locally")
                
        except Exception as e:
            print(f"Failed to retrieve feature data: {e}")
            raise

    def create_feature_view(self, features: List[str], version: str = "1"):
        """Create a feature view for model training."""
        print(f"Creating feature view with {len(features)} features")
        
        # Define feature view (simplified for demo)
        feature_view_config = {
            "name": f"{self.feature_group_name}-view",
            "version": version,
            "feature_set_name": self.feature_group_name,
            "feature_set_version": "1",
            "features": features,
            "description": "Feature view for loan approval model training"
        }
        
        # Save feature view config
        os.makedirs('feature_store', exist_ok=True)
        with open('feature_store/feature_view.json', 'w') as f:
            json.dump(feature_view_config, f, indent=2)
        
        print("Feature view configuration saved")
        return feature_view_config

    def get_training_data(self, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         features: Optional[List[str]] = None) -> pd.DataFrame:
        """Get training data from feature store with time range filtering."""
        
        print("Retrieving training data from feature store...")
        
        # Get feature data
        df = self.get_feature_data()
        
        # Apply feature selection if specified
        if features:
            available_features = [f for f in features if f in df.columns]
            missing_features = [f for f in features if f not in df.columns]
            
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
            
            if available_features:
                # Always include target if present
                if 'loan_approved' in df.columns:
                    available_features.append('loan_approved')
                df = df[available_features]
        
        # Time filtering (demo - in real feature store, this would be more sophisticated)
        if start_time or end_time:
            print(f"Note: Time filtering not implemented in this demo")
        
        print(f"Retrieved training data: {df.shape}")
        return df

    def validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature data quality."""
        print("Validating feature data quality...")
        
        validation_results = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "missing_values": {},
            "data_types": {},
            "unique_values": {},
            "validation_passed": True,
            "issues": []
        }
        
        for col in df.columns:
            # Check for missing values
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            validation_results["missing_values"][col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2)
            }
            
            # Data types
            validation_results["data_types"][col] = str(df[col].dtype)
            
            # Unique values for categorical features
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                validation_results["unique_values"][col] = int(df[col].nunique())
            
            # Validation rules
            if missing_pct > 50:
                validation_results["issues"].append(f"High missing values in {col}: {missing_pct:.1f}%")
                validation_results["validation_passed"] = False
            
            if df[col].dtype in ['int64', 'float64']:
                if df[col].min() == df[col].max():
                    validation_results["issues"].append(f"No variance in numeric feature: {col}")
        
        # Save validation results
        os.makedirs('feature_store', exist_ok=True)
        with open('feature_store/validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        if validation_results["validation_passed"]:
            print("✅ Feature validation passed")
        else:
            print("⚠️  Feature validation issues found:")
            for issue in validation_results["issues"]:
                print(f"  - {issue}")
        
        return validation_results

    def create_feature_pipeline(self):
        """Create a feature engineering pipeline for real-time inference."""
        print("Creating feature engineering pipeline...")
        
        pipeline_code = '''
import pandas as pd
import numpy as np

def engineer_features(raw_data):
    """Feature engineering pipeline for real-time inference."""
    df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else raw_data.copy()
    
    # Ratio features
    df['loan_to_income_ratio'] = df['loan_amount'] / df['income'].replace(0, np.nan)
    df['credit_score_to_age_ratio'] = df['credit_score'] / df['age'].replace(0, np.nan)
    
    # Handle infinities
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Category features
    df['age_category'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                               labels=['young', 'adult', 'middle_age', 'mature', 'senior'])
    df['income_category'] = pd.cut(df['income'], bins=[0, 30000, 50000, 80000, np.inf], 
                                  labels=['low', 'medium', 'high', 'very_high'])
    df['credit_category'] = pd.cut(df['credit_score'], bins=[0, 550, 650, 750, np.inf], 
                                  labels=['poor', 'fair', 'good', 'excellent'])
    
    # Boolean features (use float64 to handle potential missing values)
    df['high_income'] = (df['income'] >= 75000).astype('float64')
    df['excellent_credit'] = (df['credit_score'] >= 750).astype('float64')
    df['low_debt_ratio'] = (df['debt_to_income'] <= 0.3).astype('float64')
    df['experienced_employment'] = (df['employment_length'] >= 5).astype('float64')
    
    return df
'''
        
        # Save pipeline code
        os.makedirs('feature_store', exist_ok=True)
        with open('feature_store/feature_pipeline.py', 'w') as f:
            f.write(pipeline_code)
        
        print("Feature engineering pipeline saved to feature_store/feature_pipeline.py")

    def list_feature_sets(self):
        """List feature sets (simplified for demo)."""
        print("Listing feature sets (local metadata)...")
        
        feature_sets = []
        
        # Check local feature set info
        if os.path.exists('feature_store/feature_set_info.json'):
            try:
                with open('feature_store/feature_set_info.json', 'r') as f:
                    feature_set = json.load(f)
                feature_sets.append(feature_set)
                print(f"  - {feature_set['name']}:{feature_set['version']} (Status: {feature_set.get('status', 'Unknown')})")
            except Exception as e:
                print(f"Error reading local feature set info: {e}")
        
        # Note: In production, this would query Azure ML Feature Store
        print(f"Found {len(feature_sets)} feature set(s) in local metadata")
        print("Note: This is a simplified implementation. In production, would query Azure ML Feature Store.")
        
        return feature_sets

    def delete_feature_set(self, name: str, version: str):
        """Delete a feature set (simplified for demo)."""
        print(f"Note: Feature set deletion not implemented in this demo")
        print(f"Would archive feature set {name}:{version} in production")

def main():
    """Main feature store management function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure ML Feature Store Management')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--create-feature-group', action='store_true', help='Create feature set')
    parser.add_argument('--upload-features', action='store_true', help='Upload feature data')
    parser.add_argument('--validate-features', action='store_true', help='Validate feature data')
    parser.add_argument('--create-pipeline', action='store_true', help='Create feature engineering pipeline')
    parser.add_argument('--list-features', action='store_true', help='List feature sets')
    parser.add_argument('--data-path', type=str, default='processed_data/train_processed.csv', 
                       help='Path to processed data')
    
    args = parser.parse_args()
    
    # Initialize feature store
    fs = AzureMLFeatureStore(args.config)
    
    if args.create_feature_group:
        fs.create_feature_set(args.data_path)
    
    if args.upload_features:
        fs.upload_features(os.path.dirname(args.data_path))
    
    if args.validate_features:
        df = pd.read_csv(args.data_path)
        fs.validate_features(df)
    
    if args.create_pipeline:
        fs.create_feature_pipeline()
    
    if args.list_features:
        fs.list_feature_sets()
    
    # Save feature store info
    feature_store_info = {
        "feature_group_name": fs.feature_group_name,
        "workspace": fs.config['azure_ml']['workspace_name'],
        "created_at": datetime.utcnow().isoformat(),
        "config": fs.feature_config
    }
    
    with open('feature_store_info.json', 'w') as f:
        json.dump(feature_store_info, f, indent=2)
    
    print("Feature store operations completed!")

if __name__ == "__main__":
    main()