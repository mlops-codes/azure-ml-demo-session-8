#!/usr/bin/env python3
"""
Selects an Azure ML registered model (latest or a specific version)
and writes its metadata to selected_model.json
"""

import os
import sys
import json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient


def get_env(name: str, required: bool = True, default: str = "") -> str:
    val = os.environ.get(name, default)
    if required and not val:
        print(f"❌ Missing required env var: {name}")
        sys.exit(1)
    return val


def main():
    try:
        subscription_id = get_env("AZURE_SUBSCRIPTION_ID")
        resource_group = get_env("AZURE_RESOURCE_GROUP")
        workspace_name = get_env("AZURE_ML_WORKSPACE")

        model_name = get_env("MODEL_NAME")
        model_version = os.environ.get("MODEL_VERSION")  # optional

        print(f"🔍 Checking model: {model_name}, version: {model_version or 'latest'}")

        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        models = list(ml_client.models.list(name=model_name))
        if not models:
            print(f"❌ No models found with name '{model_name}'")
            sys.exit(1)

        # Pick a version
        if model_version:
            target = next((m for m in models if str(m.version) == str(model_version)), None)
            if not target:
                avail = [m.version for m in models]
                print(f"❌ Version '{model_version}' not found for '{model_name}'. Available: {avail}")
                sys.exit(1)
        else:
            # sort by numeric version desc and take latest
            target = sorted(models, key=lambda m: int(m.version), reverse=True)[0]

        info = {
            "name": target.name,
            "version": str(target.version),
            "id": target.id,
            "tags": getattr(target, "tags", {}) or {},
            "stage": (getattr(target, "tags", {}) or {}).get("stage", "Unknown"),
        }

        with open("selected_model.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"✅ Selected model {info['name']}:{info['version']} (stage={info['stage']})")
    except Exception as e:
        print(f"❌ Error selecting model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
