"""
Azure ML Model Validation for quality assurance and performance checks.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class AzureMLModelValidator:
    def __init__(self, config_path='config.json'):
        """Initialize Azure ML Model Validator."""
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
        
        # Validation thresholds
        self.thresholds = {
            'min_accuracy': 0.80,
            'min_precision': 0.75,
            'min_recall': 0.75,
            'min_f1': 0.75,
            'min_auc': 0.75,
            'max_prediction_time_ms': 100,
            'min_training_samples': 1000
        }
        
        print(f"Model validator initialized for workspace: {self.config['azure_ml']['workspace_name']}")

    def validate_model_performance(self) -> Dict[str, Any]:
        """Validate model performance against thresholds."""
        print("Validating model performance...")
        
        validation_results = {
            'performance_validation': {
                'passed': True,
                'issues': [],
                'metrics': {},
                'thresholds': self.thresholds
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            # Load metrics from training
            metrics_path = 'artifacts/model_metrics.json'
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                validation_results['performance_validation']['metrics'] = metrics
                
                # Check each threshold
                checks = [
                    ('val_accuracy', self.thresholds['min_accuracy'], 'accuracy'),
                    ('val_precision', self.thresholds['min_precision'], 'precision'),
                    ('val_recall', self.thresholds['min_recall'], 'recall'),
                    ('val_f1', self.thresholds['min_f1'], 'f1_score'),
                    ('val_auc', self.thresholds['min_auc'], 'auc')
                ]
                
                for metric_key, threshold, display_name in checks:
                    if metric_key in metrics:
                        value = metrics[metric_key]
                        if value < threshold:
                            validation_results['performance_validation']['passed'] = False
                            validation_results['performance_validation']['issues'].append(
                                f"{display_name.title()}: {value:.4f} below threshold {threshold:.4f}"
                            )
                        print(f"✓ {display_name.title()}: {value:.4f} (threshold: {threshold:.4f})")
                    else:
                        validation_results['performance_validation']['issues'].append(
                            f"Missing metric: {metric_key}"
                        )
            else:
                validation_results['performance_validation']['passed'] = False
                validation_results['performance_validation']['issues'].append(
                    "No metrics file found for validation"
                )
        
        except Exception as e:
            validation_results['performance_validation']['passed'] = False
            validation_results['performance_validation']['issues'].append(f"Error validating performance: {str(e)}")
        
        return validation_results

    def validate_data_quality(self, test_data_path: str = 'processed_data/test_processed.csv') -> Dict[str, Any]:
        """Validate test data quality."""
        print("Validating test data quality...")
        
        validation_results = {
            'data_quality': {
                'passed': True,
                'issues': [],
                'statistics': {}
            }
        }
        
        try:
            if not os.path.exists(test_data_path):
                validation_results['data_quality']['passed'] = False
                validation_results['data_quality']['issues'].append(f"Test data file not found: {test_data_path}")
                return validation_results
            
            df = pd.read_csv(test_data_path)
            
            # Basic statistics
            validation_results['data_quality']['statistics'] = {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }
            
            # Validation checks
            if len(df) < self.thresholds['min_training_samples']:
                validation_results['data_quality']['passed'] = False
                validation_results['data_quality']['issues'].append(
                    f"Insufficient test samples: {len(df)} < {self.thresholds['min_training_samples']}"
                )
            
            # Check for excessive missing values
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if missing_percentage > 10:
                validation_results['data_quality']['passed'] = False
                validation_results['data_quality']['issues'].append(
                    f"Excessive missing values: {missing_percentage:.1f}%"
                )
            
            # Check for target distribution
            if 'loan_approved' in df.columns:
                target_dist = df['loan_approved'].value_counts(normalize=True)
                if target_dist.min() < 0.1:  # Less than 10% minority class
                    validation_results['data_quality']['issues'].append(
                        f"Imbalanced target distribution: {target_dist.to_dict()}"
                    )
            
            print(f"✓ Data quality validation: {len(df)} samples, {missing_percentage:.1f}% missing")
            
        except Exception as e:
            validation_results['data_quality']['passed'] = False
            validation_results['data_quality']['issues'].append(f"Error validating data quality: {str(e)}")
        
        return validation_results

    def validate_model_artifacts(self) -> Dict[str, Any]:
        """Validate required model artifacts exist."""
        print("Validating model artifacts...")
        
        validation_results = {
            'artifacts': {
                'passed': True,
                'issues': [],
                'found_artifacts': []
            }
        }
        
        # Required artifacts
        required_artifacts = [
            'models/model.pkl',
            'models/scaler.pkl',
            'models/feature_info.json',
            'artifacts/model_metrics.json'
        ]
        
        optional_artifacts = [
            'artifacts/registered_model.json',
            'feature_store_info.json',
            'feature_store/validation_results.json'
        ]
        
        # Check required artifacts
        for artifact in required_artifacts:
            if os.path.exists(artifact):
                validation_results['artifacts']['found_artifacts'].append(artifact)
                print(f"✓ Found required artifact: {artifact}")
            else:
                validation_results['artifacts']['passed'] = False
                validation_results['artifacts']['issues'].append(f"Missing required artifact: {artifact}")
                print(f"✗ Missing required artifact: {artifact}")
        
        # Check optional artifacts
        for artifact in optional_artifacts:
            if os.path.exists(artifact):
                validation_results['artifacts']['found_artifacts'].append(artifact)
                print(f"✓ Found optional artifact: {artifact}")
        
        return validation_results

    def validate_model_inference(self) -> Dict[str, Any]:
        """Test model inference functionality."""
        print("Validating model inference...")
        
        validation_results = {
            'inference': {
                'passed': True,
                'issues': [],
                'performance': {}
            }
        }
        
        try:
            # Load model and preprocessing objects
            if not os.path.exists('models/model.pkl'):
                validation_results['inference']['passed'] = False
                validation_results['inference']['issues'].append("Model file not found")
                return validation_results
            
            model = joblib.load('models/model.pkl')
            
            # Create test input
            test_input = pd.DataFrame([{
                'age': 35,
                'income': 65000,
                'employment_length': 5.2,
                'credit_score': 720,
                'debt_to_income': 0.25,
                'loan_amount': 25000,
                'high_income': 0,
                'excellent_credit': 0,
                'low_debt_ratio': 1,
                'experienced_employment': 1,
                'loan_to_income_ratio': 0.385,
                'credit_score_to_age_ratio': 20.57,
                'home_ownership': 2,  # Encoded
                'loan_purpose': 0,    # Encoded
                'age_category': 2,    # Encoded
                'income_category': 2, # Encoded
                'credit_category': 2  # Encoded
            }])
            
            # Test prediction
            import time
            start_time = time.time()
            
            # Make prediction
            prediction = model.predict(test_input)
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(test_input)
                validation_results['inference']['performance']['has_probabilities'] = True
            else:
                validation_results['inference']['performance']['has_probabilities'] = False
            
            end_time = time.time()
            prediction_time_ms = (end_time - start_time) * 1000
            
            validation_results['inference']['performance']['prediction_time_ms'] = prediction_time_ms
            validation_results['inference']['performance']['prediction_result'] = int(prediction[0])
            
            # Check prediction time
            if prediction_time_ms > self.thresholds['max_prediction_time_ms']:
                validation_results['inference']['issues'].append(
                    f"Slow prediction time: {prediction_time_ms:.2f}ms > {self.thresholds['max_prediction_time_ms']}ms"
                )
            
            print(f"✓ Inference test passed: {prediction_time_ms:.2f}ms, prediction: {prediction[0]}")
            
        except Exception as e:
            validation_results['inference']['passed'] = False
            validation_results['inference']['issues'].append(f"Inference test failed: {str(e)}")
            print(f"✗ Inference test failed: {str(e)}")
        
        return validation_results

    def validate_feature_store_integration(self) -> Dict[str, Any]:
        """Validate feature store integration."""
        print("Validating feature store integration...")
        
        validation_results = {
            'feature_store': {
                'passed': True,
                'issues': [],
                'info': {}
            }
        }
        
        try:
            # Check if feature store info exists
            if os.path.exists('feature_store_info.json'):
                with open('feature_store_info.json', 'r') as f:
                    fs_info = json.load(f)
                
                validation_results['feature_store']['info'] = fs_info
                print(f"✓ Feature store integration found: {fs_info.get('feature_group_name', 'Unknown')}")
            else:
                validation_results['feature_store']['issues'].append(
                    "Feature store info file not found"
                )
            
            # Check feature validation results
            if os.path.exists('feature_store/validation_results.json'):
                with open('feature_store/validation_results.json', 'r') as f:
                    validation_data = json.load(f)
                
                if not validation_data.get('validation_passed', True):
                    validation_results['feature_store']['passed'] = False
                    validation_results['feature_store']['issues'].extend(
                        validation_data.get('issues', [])
                    )
                print(f"✓ Feature validation results found")
            
        except Exception as e:
            validation_results['feature_store']['issues'].append(
                f"Error validating feature store integration: {str(e)}"
            )
        
        return validation_results

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Running comprehensive model validation...")
        print("=" * 50)
        
        results = {
            'overall_passed': True,
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'validator_version': '1.0.0',
            'checks': {}
        }
        
        # Run all validation checks
        validation_checks = [
            ('performance', self.validate_model_performance),
            ('data_quality', self.validate_data_quality),
            ('artifacts', self.validate_model_artifacts),
            ('inference', self.validate_model_inference),
            ('feature_store', self.validate_feature_store_integration)
        ]
        
        for check_name, check_func in validation_checks:
            print(f"\n📋 Running {check_name} validation...")
            try:
                check_result = check_func()
                results['checks'][check_name] = check_result
                
                # Update overall status
                for key in check_result:
                    if isinstance(check_result[key], dict) and 'passed' in check_result[key]:
                        if not check_result[key]['passed']:
                            results['overall_passed'] = False
                            print(f"❌ {check_name} validation failed")
                            for issue in check_result[key].get('issues', []):
                                print(f"   - {issue}")
                        else:
                            print(f"✅ {check_name} validation passed")
                
            except Exception as e:
                print(f"❌ {check_name} validation error: {str(e)}")
                results['overall_passed'] = False
                results['checks'][check_name] = {
                    'error': str(e),
                    'passed': False
                }
        
        # Save validation results
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 50)
        if results['overall_passed']:
            print("🎉 ALL VALIDATIONS PASSED")
        else:
            print("⚠️  VALIDATION ISSUES FOUND")
        
        print(f"📄 Validation report saved to: artifacts/validation_results.json")
        
        return results

def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure ML Model Validation')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance validation')
    parser.add_argument('--data-quality-only', action='store_true', help='Run only data quality validation')
    parser.add_argument('--artifacts-only', action='store_true', help='Run only artifacts validation')
    parser.add_argument('--inference-only', action='store_true', help='Run only inference validation')
    parser.add_argument('--feature-store-only', action='store_true', help='Run only feature store validation')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = AzureMLModelValidator(args.config)
    
    # Run specific validation or comprehensive
    if args.performance_only:
        result = validator.validate_model_performance()
    elif args.data_quality_only:
        result = validator.validate_data_quality()
    elif args.artifacts_only:
        result = validator.validate_model_artifacts()
    elif args.inference_only:
        result = validator.validate_model_inference()
    elif args.feature_store_only:
        result = validator.validate_feature_store_integration()
    else:
        result = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    if isinstance(result, dict):
        if result.get('overall_passed', True):
            exit(0)
        else:
            exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()