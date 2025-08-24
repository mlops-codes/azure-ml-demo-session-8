#!/usr/bin/env python3
"""
Main Azure ML pipeline orchestration for loan approval prediction.
This script manages the complete ML pipeline from data preprocessing to model deployment.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional

# ----------------------------
# Logging & Config Utilities
# ----------------------------

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """Load configuration from JSON file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    return json.loads(p.read_text())

def ensure_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required config sections/keys exist with sensible defaults."""
    config.setdefault('pipeline_settings', {})
    ps = config['pipeline_settings']
    ps.setdefault('use_azure_ml', False)
    ps.setdefault('deploy_endpoint', False)
    ps.setdefault('continue_on_failure', False)

    config.setdefault('experiment', {})
    ex = config['experiment']
    ex.setdefault('model_name', 'loan-approval-model')
    # model_version is optional and typically discovered post-training
    config.setdefault('data', {})
    data = config['data']
    data.setdefault('base_dir', 'data')
    return config

# ----------------------------
# Data Checks & Preprocessing
# ----------------------------

def check_data_exists(config: Dict[str, Any]) -> List[str]:
    """Check if required data files exist based on config."""
    base = Path(config.get('data', {}).get('base_dir', 'data'))
    required_files = [
        base / 'train_data.csv',
        base / 'test_data.csv'
    ]
    return [str(p) for p in required_files if not p.exists()]

def run_data_preprocessing(logger: logging.Logger) -> bool:
    """Run data preprocessing step."""
    logger.info("Starting data preprocessing...")
    try:
        from data_preprocessing import main as preprocess_main
        preprocess_main()
        logger.info("Data preprocessing completed successfully")
        return True
    except Exception:
        logger.exception("Data preprocessing failed")
        return False

# ----------------------------
# Training (Local & Azure)
# ----------------------------

def run_local_training(config, logger):
    """Run local model training with MLflow."""
    logger.info("Starting local model training...")
    try:
        from azureml_training import AzureMLTrainer
        trainer = AzureMLTrainer()  # no config in __init__

        # If your trainer supports config via a method or attribute, set it:
        if hasattr(trainer, "set_config") and callable(getattr(trainer, "set_config")):
            trainer.set_config(config)
        elif hasattr(trainer, "config"):
            setattr(trainer, "config", config)

        # Try passing config if the method supports it; otherwise call without args
        try:
            model, train_metrics, val_metrics = trainer.train_local_with_mlflow(config=config)
        except TypeError:
            model, train_metrics, val_metrics = trainer.train_local_with_mlflow()

        logger.info("Local training completed successfully")

        acc = val_metrics.get('val_accuracy')
        logger.info("Validation accuracy: %s", f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A")

        auc = val_metrics.get('val_auc')
        logger.info("Validation AUC: %s", f"{auc:.4f}" if isinstance(auc, (int, float)) else "N/A")

        # Persist metrics for downstream steps
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/local_training_metrics.json").write_text(
            json.dumps({"train": train_metrics, "val": val_metrics}, indent=2)
        )

        return True, model, train_metrics, val_metrics
    except Exception:
        logger.exception("Local training failed")
        return False, None, None, None


def run_azure_ml_training(config, logger):
    """Run Azure ML training job."""
    logger.info("Starting Azure ML training job...")
    try:
        from azureml_training import AzureMLTrainer
        from azureml_deployment import create_training_environment

        trainer = AzureMLTrainer()  # no config in __init__
        if hasattr(trainer, "set_config") and callable(getattr(trainer, "set_config")):
            trainer.set_config(config)
        elif hasattr(trainer, "config"):
            setattr(trainer, "config", config)

        # Create environment assets/files
        create_training_environment()

        # Upload data
        data_asset = trainer.upload_data('processed_data')  # adjust path/name as your trainer expects
        logger.info("Data uploaded: %s:%s", getattr(data_asset, "name", "unknown"), getattr(data_asset, "version", "unknown"))

        # Create or reference environment
        env_name = trainer.create_environment()
        logger.info("Environment created: %s", env_name)

        # Submit job (try passing config if supported)
        try:
            job = trainer.submit_training_job(data_asset, config=config)
        except TypeError:
            job = trainer.submit_training_job(data_asset)

        logger.info("Training job submitted: %s", getattr(job, "name", "unknown"))
        if hasattr(job, "studio_url"):
            logger.info("Studio URL: %s", job.studio_url)

        # Persist job refs
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/last_training_job.json").write_text(json.dumps({
            "job_name": getattr(job, "name", None),
            "studio_url": getattr(job, "studio_url", None)
        }, indent=2))

        # Capture registered model info if the trainer exposes it
        if hasattr(trainer, "get_registered_model_info"):
            info = trainer.get_registered_model_info()
            if info:
                Path("artifacts/registered_model.json").write_text(json.dumps(info, indent=2))
                logger.info("Registered model captured: %s", info)

        return True, job
    except Exception:
        logger.exception("Azure ML training failed")
        return False, None


# ----------------------------
# Deployment
# ----------------------------

def run_model_deployment(config: Dict[str, Any], model_name: str, model_version: str, logger: logging.Logger) -> Tuple[bool, Optional[Any], Optional[Any]]:
    """Deploy model to Azure ML endpoint."""
    logger.info("Starting model deployment...")
    try:
        from azureml_deployment import AzureMLDeployer, create_inference_environment

        # Create environment files/assets for inference
        create_inference_environment()

        deployer = AzureMLDeployer(config=config)

        # Create endpoint (idempotent if you implement it that way)
        endpoint = deployer.create_endpoint()
        logger.info("Endpoint created: %s", getattr(endpoint, "name", "unknown"))

        # Deploy model
        deployment = deployer.create_deployment(model_name, model_version)
        logger.info("Model deployed: %s", getattr(deployment, "name", "unknown"))

        # Smoke test
        response = deployer.test_endpoint()
        if response:
            logger.info("Endpoint test successful")
        else:
            logger.warning("Endpoint test failed")

        # Persist endpoint/deployment info
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/last_deployment.json").write_text(json.dumps({
            "endpoint_name": getattr(endpoint, "name", None),
            "deployment_name": getattr(deployment, "name", None),
            "model_name": model_name,
            "model_version": model_version
        }, indent=2))

        return True, endpoint, deployment
    except Exception:
        logger.exception("Model deployment failed")
        return False, None, None

# ----------------------------
# Step Orchestrator
# ----------------------------

def run_pipeline_step(step: str,
                      config: Dict[str, Any],
                      logger: logging.Logger,
                      continue_on_failure: bool = False) -> bool:
    """Run a specific pipeline step."""
    logger.info("=" * 50)
    logger.info("STEP: %s", step.upper())
    logger.info("=" * 50)

    ps = config.get('pipeline_settings', {})
    use_azure_ml = ps.get('use_azure_ml', False)
    deploy_endpoint = ps.get('deploy_endpoint', False)

    try:
        if step == 'data':
            missing_files = check_data_exists(config)
            if missing_files:
                logger.warning("Missing data files: %s", missing_files)
                logger.info("Please run 'python generate_mock_data.py' first to generate data")
                return False
            logger.info("Data files found")
            return True

        elif step == 'preprocess':
            return run_data_preprocessing(logger)

        elif step == 'train':
            if use_azure_ml:
                success, job = run_azure_ml_training(config, logger)
                return success
            else:
                success, model, train_metrics, val_metrics = run_local_training(config, logger)
                return success

        elif step == 'deploy':
            if not deploy_endpoint:
                logger.info("Model deployment disabled in configuration")
                return True

            # Determine model name & version
            model_name = config.get('experiment', {}).get('model_name', 'loan-approval-model')
            model_version = config.get('experiment', {}).get('model_version')  # optional

            # Try to read from artifacts if not provided
            if not model_version:
                reg = Path("artifacts/registered_model.json")
                if reg.exists():
                    try:
                        info = json.loads(reg.read_text())
                        model_version = str(info.get("version")) if info.get("version") is not None else None
                        if not model_name and info.get("name"):
                            model_name = info["name"]
                    except Exception:
                        logger.warning("Could not parse artifacts/registered_model.json")

            if not model_version:
                logger.error("No model_version found for deployment (set in config.experiment.model_version or persist from training).")
                return False

            success, _, _ = run_model_deployment(config, model_name, model_version, logger)
            return success

        else:
            logger.error("Unknown pipeline step: %s", step)
            return False

    except Exception:
        logger.exception("Unhandled exception in step '%s'", step)
        return True if continue_on_failure else False

# ----------------------------
# CLI Entry
# ----------------------------

def main() -> int:
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(description='Azure ML Loan Approval Pipeline')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--step', type=str, choices=['data', 'preprocess', 'train', 'deploy'],
                        help='Run specific pipeline step')
    parser.add_argument('--local-only', action='store_true', help='Run pipeline locally without Azure ML services')
    parser.add_argument('--continue-on-failure', action='store_true', help='Continue pipeline execution even if steps fail')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Load configuration
    try:
        config = load_config(args.config)
        config = ensure_defaults(config)
        logger.info("Configuration loaded from %s", args.config)
    except Exception:
        logger.exception("Failed to load configuration")
        return 1

    # Override config with command line arguments
    if args.local_only:
        config['pipeline_settings']['use_azure_ml'] = False
        config['pipeline_settings']['deploy_endpoint'] = False
        logger.info("Running in local-only mode")

    if args.continue_on_failure:
        config['pipeline_settings']['continue_on_failure'] = True

    ps = config.get('pipeline_settings', {})
    continue_on_failure = ps.get('continue_on_failure', False)

    # Pipeline start
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("AZURE ML LOAN APPROVAL PIPELINE STARTING")
    logger.info("Start time: %s", start_time)
    logger.info("=" * 60)

    # Define pipeline steps
    if args.step:
        steps = [args.step]
        logger.info("Running single step: %s", args.step)
    else:
        steps = ['data', 'preprocess', 'train']
        if ps.get('deploy_endpoint', False):
            steps.append('deploy')
        logger.info("Running full pipeline: %s", " -> ".join(steps))

    # Execute pipeline steps
    failed_steps: List[str] = []
    successful_steps: List[str] = []

    for step in steps:
        success = False
        try:
            success = run_pipeline_step(step, config, logger, continue_on_failure)
        except Exception:
            logger.exception("Unexpected error in step '%s'", step)
            success = False

        if success:
            successful_steps.append(step)
        else:
            failed_steps.append(step)
            if not continue_on_failure:
                break

    # Pipeline completion
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 60)
    logger.info("AZURE ML LOAN APPROVAL PIPELINE COMPLETED")
    logger.info("End time: %s", end_time)
    logger.info("Duration: %s", duration)
    logger.info("Successful steps: %s", successful_steps)
    if failed_steps:
        logger.error("Failed steps: %s", failed_steps)
    logger.info("=" * 60)

    # Exit codes
    if not failed_steps:
        logger.info("🎉 Pipeline completed successfully!")
        return 0
    elif successful_steps:
        logger.warning("⚠️  Pipeline completed with %d failed step(s)", len(failed_steps))
        return 1
    else:
        logger.error("❌ Pipeline failed completely")
        return 2

if __name__ == "__main__":
    sys.exit(main())