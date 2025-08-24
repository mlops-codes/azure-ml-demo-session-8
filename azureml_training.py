"""
Azure ML training job management for loan approval model.
"""

import os
import json
import time
from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.entities import Environment, Data, Model, AmlCompute
from azure.ai.ml import Output
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import mlflow


class AzureMLTrainer:
    def __init__(self, config_path="config.json"):
        """Initialize Azure ML trainer."""
        with open(config_path, "r") as f:
            self.config = json.load(f)

        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.config["azure_ml"]["subscription_id"],
            resource_group_name=self.config["azure_ml"]["resource_group"],
            workspace_name=self.config["azure_ml"]["workspace_name"],
        )

        try:
            workspace = self.ml_client.workspaces.get(self.config["azure_ml"]["workspace_name"])
            print(f"✅ Connected to Azure ML workspace: {workspace.name}")
            self.workspace_available = True
        except Exception as e:
            print(
                f"⚠️  Warning: Could not connect to workspace: {self.config['azure_ml']['workspace_name']}\n   Error: {str(e)}\n   Training will fall back to local-only mode"
            )
            self.workspace_available = False

    def create_environment(self):
        """Create or update Azure ML environment."""
        env_name = "loan-approval-env"
        environment = Environment(
            name=env_name,
            description="Environment for loan approval model training",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file="environment.yml",
        )
        env = self.ml_client.environments.create_or_update(environment)
        print(f"Environment '{env_name}' created/updated")
        return env_name

    def create_compute_cluster(self, compute_name=None):
        """Create or get compute cluster."""
        if not self.workspace_available:
            print("⚠️  Skipping compute cluster creation - workspace not available")
            return None

        if compute_name is None:
            compute_name = self.config.get("compute", {}).get("training_compute", "cpu-cluster")

        try:
            compute = self.ml_client.compute.get(compute_name)
            print(f"✅ Using existing compute: {compute_name}")
            return compute
        except Exception:
            print(f"Compute {compute_name} not found, creating new one...")
            try:
                compute_config = AmlCompute(
                    name=compute_name,
                    type="amlcompute",
                    size=self.config.get("compute", {}).get("training_vm_size", "STANDARD_DS3_V2"),
                    min_instances=0,
                    max_instances=4,
                    idle_time_before_scale_down=300,
                )
                compute = self.ml_client.compute.begin_create_or_update(compute_config).result()
                print(f"✅ Compute cluster created: {compute_name}")
                return compute
            except Exception as create_error:
                print(f"❌ Failed to create compute cluster: {create_error}")
                return None

    def upload_data(self, local_path="processed_data"):
        """Upload data to Azure ML workspace."""
        data_name = "loan-approval-data"
        data = Data(name=data_name, path=local_path, type=AssetTypes.URI_FOLDER, description="Preprocessed loan approval data")
        data_asset = self.ml_client.data.create_or_update(data)
        print(f"Data uploaded: {data_asset.name}:{data_asset.version}")
        return data_asset

    def submit_training_job(self, data_asset):
        """Submit training job to Azure ML."""
        experiment_name = self.config["experiment"]["experiment_name"]
        self.create_compute_cluster()
        hyperparams = self.config["model_hyperparameters"]

        job = command(
            code="./scripts",
            command=(
                "python train.py "
                "--data_path ${{inputs.data}} "
                f"--model_type {hyperparams['model_type']} "
                f"--n_estimators {hyperparams['n_estimators']} "
                f"--max_depth {hyperparams['max_depth']} "
                f"--random_state {hyperparams['random_state']} "
                "--use_feature_store true "
                "--output_model ${{outputs.model}}"  # named output -> MLflow dir
            ),
            environment="loan-approval-env@latest",
            inputs={"data": Input(type="uri_folder", path=data_asset.id)},
            outputs={"model": Output(type=AssetTypes.MLFLOW_MODEL)},  # Azure will expect an MLflow folder here
            compute=self.config["compute"]["training_compute"],
            instance_count=1,
            experiment_name=experiment_name,
            display_name="Loan approval model training",
            description="Loan approval model training with feature store",
            tags={
                "model_type": hyperparams["model_type"],
                "environment": self.config.get("environment", "dev"),
                "feature_store_enabled": "true",
            },
        )

        job_result = self.ml_client.jobs.create_or_update(job)
        print(f"Training job submitted: {job_result.name}")
        print(f"Job status: {job_result.status}")
        if hasattr(job_result, "studio_url"):
            print(f"Studio URL: {job_result.studio_url}")
        return job_result

    def wait_for_job_completion(self, job_name, timeout_minutes=60):
        """Wait for training job to complete."""
        print(f"Waiting for job {job_name} to complete...")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            try:
                job = self.ml_client.jobs.get(job_name)
                status = job.status
                print(f"Job status: {status}")

                if status in ["Completed", "Failed", "Canceled"]:
                    if status == "Completed":
                        print(f"✅ Job {job_name} completed successfully")
                        return True
                    else:
                        print(f"❌ Job {job_name} finished with status: {status}")
                        return False

                if time.time() - start_time > timeout_seconds:
                    print(f"⏰ Job {job_name} timeout after {timeout_minutes} minutes")
                    return False

                time.sleep(30)
            except Exception as e:
                print(f"Error checking job status: {e}")
                return False

    def register_model(self, job_name, model_name=None):
        """Register trained model from job outputs."""
        if model_name is None:
            model_name = self.config["experiment"]["model_name"]

        model = Model(
            name=model_name,
            path=f"azureml://jobs/{job_name}/outputs/model",  # named output 'model'
            description="Loan approval prediction model (MLflow directory) from Azure ML job",
            type=AssetTypes.MLFLOW_MODEL,
            tags={
                "training_job": job_name,
                "model_type": self.config.get("model_hyperparameters", {}).get("model_type", "unknown"),
                "feature_store_enabled": "true",
                "created_by": "github_actions",
            },
        )

        registered_model = self.ml_client.models.create_or_update(model)
        print(f"Model registered: {registered_model.name}:{registered_model.version}")

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/registered_model.json", "w") as f:
            json.dump(
                {
                    "name": registered_model.name,
                    "version": registered_model.version,
                    "training_job": job_name,
                    "registration_time": time.time(),
                },
                f,
                indent=2,
            )

        return registered_model

    # local MLflow training retained as-is (optional for you)
    def train_local_with_mlflow(self):
        print("Starting local training with MLflow tracking...")
        mlflow.set_experiment(self.config["experiment"]["experiment_name"])
       # Import training components
        from data_preprocessing import prepare_data_for_training
        import pandas as pd
        import numpy as np
        import json
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import joblib
        
        with mlflow.start_run():
            # Prepare data - check if processed data already exists (from workflow)
            print("Loading and preprocessing data...")
            
            # In workflow context, processed data might already exist
            if os.path.exists('processed_data/train_processed.csv'):
                print("Found preprocessed data, using it directly")
                train_df = pd.read_csv('processed_data/train_processed.csv')
                val_df = pd.read_csv('processed_data/val_processed.csv') if os.path.exists('processed_data/val_processed.csv') else None
                
                # Load feature info
                if os.path.exists('models/feature_info.json'):
                    with open('models/feature_info.json', 'r') as f:
                        feature_info = json.load(f)
                else:
                    feature_info = {"target": "loan_approved"}
            else:
                # Fall back to full preprocessing
                train_df, val_df, feature_info = prepare_data_for_training()
            
            # Separate features and target
            X_train = train_df.drop(columns=['loan_approved'])
            y_train = train_df['loan_approved']
            
            if val_df is not None:
                X_val = val_df.drop(columns=['loan_approved'])
                y_val = val_df['loan_approved']
            else:
                # Split training data if no validation data exists
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            
            # Get hyperparameters
            hyperparams = self.config['model_hyperparameters']
            mlflow.log_params(hyperparams)
            
            # Train model
            print(f"Training {hyperparams['model_type']} model...")
            if hyperparams['model_type'] == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=hyperparams['n_estimators'],
                    max_depth=hyperparams['max_depth'],
                    random_state=hyperparams['random_state'],
                    n_jobs=-1
                )
            elif hyperparams['model_type'] == 'logistic_regression':
                model = LogisticRegression(
                    random_state=hyperparams['random_state'],
                    max_iter=1000
                )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'train_precision': precision_score(y_train, y_train_pred),
                'train_recall': recall_score(y_train, y_train_pred),
                'train_f1': f1_score(y_train, y_train_pred)
            }
            
            val_metrics = {
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred),
                'val_recall': recall_score(y_val, y_val_pred),
                'val_f1': f1_score(y_val, y_val_pred)
            }
            
            # Add AUC if model supports probability prediction
            if hasattr(model, 'predict_proba'):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_val_proba = model.predict_proba(X_val)[:, 1]
                train_metrics['train_auc'] = roc_auc_score(y_train, y_train_proba)
                val_metrics['val_auc'] = roc_auc_score(y_val, y_val_proba)
            
            # Log metrics
            mlflow.log_metrics({**train_metrics, **val_metrics})
            
            # Print results
            print("\nTraining Results:")
            for metric, value in train_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print("\nValidation Results:")
            for metric, value in val_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model_path = 'models/model.pkl'
            joblib.dump(model, model_path)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                input_example=X_train.iloc[:5],
                signature=mlflow.models.infer_signature(X_train, y_train)
            )
            
            # Save metrics for validation
            os.makedirs('artifacts', exist_ok=True)
            with open('artifacts/model_metrics.json', 'w') as f:
                json.dump({**train_metrics, **val_metrics}, f, indent=2)
            
            print(f"\nModel saved locally to: {model_path}")
            print("Local training completed successfully!")
            
            return model, train_metrics, val_metrics

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Azure ML Training")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--create-compute", action="store_true")
    parser.add_argument("--upload-data", action="store_true")
    parser.add_argument("--submit-job", action="store_true")
    parser.add_argument("--wait-for-completion", action="store_true")
    parser.add_argument("--job-name", type=str)
    parser.add_argument("--register-model", type=str)
    args = parser.parse_args()

    trainer = AzureMLTrainer(args.config)

    if args.local_only:
        print("🏠 Running in local-only mode")
        trainer.train_local_with_mlflow()
        return

    if not trainer.workspace_available:
        print("🔄 Workspace not available, running local training instead")
        trainer.train_local_with_mlflow()
        return

    print("🚀 Running Azure ML training workflow")

    if args.create_compute:
        compute = trainer.create_compute_cluster()
        if not compute:
            return

    data_asset = None
    if args.upload_data:
        try:
            data_asset = trainer.upload_data()
        except Exception as e:
            print(f"❌ Data upload failed: {e}")
            return

    if args.submit_job:
        try:
            trainer.create_environment()
            if data_asset is None:
                # If not explicitly uploaded this run, use latest
                data_assets = trainer.ml_client.data.list(name="loan-approval-data")
                data_asset = next(iter(data_assets))
            job = trainer.submit_training_job(data_asset)
            print(f"Training job submitted: {job.name}")
            os.makedirs("artifacts", exist_ok=True)
            with open("artifacts/last_training_job.json", "w") as f:
                json.dump(
                    {"job_name": job.name, "studio_url": getattr(job, "studio_url", None), "status": job.status}, f, indent=2
                )
        except Exception as e:
            print(f"❌ Job submission failed: {e}")
            print("🔄 Falling back to local training")
            trainer.train_local_with_mlflow()

    if args.wait_for_completion:
        if not args.job_name:
            print("Error: --job-name required for wait-for-completion")
            return
        if not trainer.wait_for_job_completion(args.job_name):
            print("Training job did not complete successfully")
            return

    if args.register_model:
        try:
            model = trainer.register_model(args.register_model)
            print(f"Model registered: {model.name}:{model.version}")
        except Exception as e:
            print(f"❌ Model registration failed: {e}")


if __name__ == "__main__":
    main()
