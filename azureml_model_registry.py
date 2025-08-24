"""
Azure ML Model Registry management for loan approval model.
Handles model registration, versioning, tagging, and lifecycle management.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


class AzureMLModelRegistry:
    def __init__(self, config_path="config.json"):
        """Initialize Azure ML Model Registry client."""
        with open(config_path, "r") as f:
            self.config = json.load(f)

        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.config["azure_ml"]["subscription_id"],
            resource_group_name=self.config["azure_ml"]["resource_group"],
            workspace_name=self.config["azure_ml"]["workspace_name"],
        )
        print(f"Connected to Azure ML workspace: {self.config['azure_ml']['workspace_name']}")

    def register_model_from_local(self, model_path: str = "models", model_name: Optional[str] = None) -> Model:
        """Register model from local training artifacts (pickle/custom folder)."""
        if model_name is None:
            model_name = self.config.get("experiment", {}).get("model_name", "loan-approval-model")

        print(f"Registering model from local artifacts: {model_path}")

        # Expecting at least a pickle
        model_file = os.path.join(model_path, "model.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Local model file not found: {model_file}")

        metrics = {}
        metrics_file = os.path.join("artifacts", "model_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

        model = Model(
            name=model_name,
            path=model_path,
            description="Loan approval prediction model trained locally",
            type=AssetTypes.CUSTOM_MODEL,  # local path is a pickle folder
            tags={
                "training_mode": "local",
                "model_type": self.config.get("model_hyperparameters", {}).get("model_type", "random_forest"),
                "framework": "scikit-learn",
                "environment": self.config.get("environment", "dev"),
                "feature_store_enabled": "true",
                "created_by": "github_actions",
                "created_at": datetime.utcnow().isoformat(),
                "version_description": "Model from local training",
                **{f"metric_{k}": str(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            },
            properties={
                "training_dataset": "loan-approval-data",
                "validation_performed": "true",
                "model_format": "scikit-learn-pickle",
                "inference_ready": "true",
                "local_training": "true",
            },
        )

        registered_model = self.ml_client.models.create_or_update(model)
        print(f"✅ Model registered successfully: {registered_model.name}:{registered_model.version}")
        self._save_model_info(registered_model, "local_training")
        return registered_model

    def register_model_from_job(self, job_name: str, model_name: Optional[str] = None) -> Model:
        """Register model from completed training job (MLflow directory at outputs/model)."""
        if model_name is None:
            model_name = self.config.get("experiment", {}).get("model_name", "loan-approval-model")

        print(f"Registering model from job: {job_name}")

        # Ensure job succeeded
        job = self.ml_client.jobs.get(job_name)
        if getattr(job, "status", None) != "Completed":
            raise ValueError(f"Job {job_name} has not completed successfully. Status: {job.status}")

        model = Model(
            name=model_name,
            path=f"azureml://jobs/{job_name}/outputs/model",  # <- matches named output
            description=f"Loan approval prediction model trained from job {job_name}",
            type=AssetTypes.MLFLOW_MODEL,  # <- MLflow model directory
            tags={
                "training_job": job_name,
                "model_type": self.config.get("model_hyperparameters", {}).get("model_type", "random_forest"),
                "framework": "scikit-learn",
                "environment": self.config.get("environment", "dev"),
                "feature_store_enabled": "true",
                "created_by": "github_actions",
                "created_at": datetime.utcnow().isoformat(),
                "version_description": f"Model from training job {job_name}",
            },
            properties={
                "training_dataset": "loan-approval-data",
                "validation_performed": "true",
                "model_format": "mlflow",
                "inference_ready": "true",
            },
        )

        registered_model = self.ml_client.models.create_or_update(model)
        print(f"✅ Model registered successfully: {registered_model.name}:{registered_model.version}")
        self._save_model_info(registered_model, job_name)
        return registered_model

    def _save_model_info(self, model: Model, job_name: str):
        os.makedirs("artifacts", exist_ok=True)
        model_info = {
            "name": model.name,
            "version": model.version,
            "id": model.id,
            "training_job": job_name,
            "registration_time": datetime.utcnow().isoformat(),
            "tags": model.tags,
            "properties": model.properties,
            "description": model.description,
        }
        with open("artifacts/registered_model.json", "w") as f:
            json.dump(model_info, f, indent=2)
        print("Model info saved to artifacts/registered_model.json")

    def list_models(self, model_name: Optional[str] = None) -> List[Model]:
        try:
            if model_name:
                models = list(self.ml_client.models.list(name=model_name))
                print(f"Found {len(models)} versions of model '{model_name}'")
            else:
                models = list(self.ml_client.models.list())
                print(f"Found {len(models)} models in registry")

            for m in models:
                created_time = getattr(getattr(m, "creation_context", None), "created_at", "Unknown")
                print(f"  - {m.name}:{m.version} (Created: {created_time})")
                if getattr(m, "tags", None):
                    env = m.tags.get("environment", "Unknown")
                    mtype = m.tags.get("model_type", "Unknown")
                    print(f"    Environment: {env}, Type: {mtype}")
            return models
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []

    def add_model_tags(self, model_name: str, version: str, tags: Dict[str, str]):
        try:
            model = self.ml_client.models.get(name=model_name, version=version)
            existing_tags = model.tags or {}
            existing_tags.update(tags)

            updated_model = Model(
                name=model.name,
                version=model.version,
                path=model.path,
                description=model.description,
                type=model.type,
                tags=existing_tags,
                properties=model.properties,
            )

            result = self.ml_client.models.create_or_update(updated_model)
            print(f"✅ Tags added to model {model_name}:{version}")
            for k, v in tags.items():
                print(f"  - {k}: {v}")
            return result
        except Exception as e:
            print(f"❌ Failed to add tags to model: {e}")
            raise


def main():
    """Main model registry management function."""
    import argparse

    parser = argparse.ArgumentParser(description="Azure ML Model Registry Management")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file")
    parser.add_argument("--register-from-job", type=str, help="Register model from training job")
    parser.add_argument("--register-from-local", action="store_true", help="Register model from local artifacts")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--list-models", action="store_true", help="List models")
    parser.add_argument("--add-tags", action="store_true", help="Add tags to model")
    parser.add_argument("--tags", nargs="*", help="Tags in format key=value")
    parser.add_argument("--model-info", type=str, help="Model info in format name:version")
    args = parser.parse_args()

    registry = AzureMLModelRegistry(args.config)

    if args.register_from_job:
        model = registry.register_model_from_job(args.register_from_job, args.model_name)
        print(f"Model registered: {model.name}:{model.version}")

    if args.register_from_local:
        model = registry.register_model_from_local("models", args.model_name)
        print(f"Model registered: {model.name}:{model.version}")

    if args.list_models:
        _ = registry.list_models(args.model_name)

    if args.add_tags:
        if args.model_info and args.tags:
            name, version = args.model_info.split(":")
            tags_dict = {}
            for tag in args.tags:
                if "=" in tag:
                    k, v = tag.split("=", 1)
                    tags_dict[k] = v
            registry.add_model_tags(name, version, tags_dict)
        else:
            print("Error: --model-info and --tags required for add-tags")


if __name__ == "__main__":
    main()
