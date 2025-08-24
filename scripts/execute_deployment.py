#!/usr/bin/env python3
"""
Creates/updates/deletes/tests a Managed Online Endpoint + Deployment for a registered model.
Writes a summary to artifacts/deployment_result.json
"""

import os
import sys
import json
import time
from typing import Optional, Dict

import requests
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.core.exceptions import ResourceNotFoundError


def env(name: str, required: bool = True, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default if default is not None else "")
    if required and (val is None or val == ""):
        print(f"❌ Missing required env var: {name}")
        sys.exit(1)
    return val


def get_ml_client() -> MLClient:
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=env("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=env("AZURE_RESOURCE_GROUP"),
        workspace_name=env("AZURE_ML_WORKSPACE"),
    )


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        ep = ml_client.online_endpoints.get(endpoint_name)
        print(f"🔄 Endpoint exists: {ep.name} (state={ep.provisioning_state})")
        return ep
    except ResourceNotFoundError:
        print(f"✨ Creating endpoint '{endpoint_name}' (key auth)...")
        ep = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key",  # key auth so you can call with Bearer <key>
        )
        poller = ml_client.begin_create_or_update(ep)
        ep = poller.result()
        print(f"✅ Endpoint created: {ep.name}")
        return ep


def delete_endpoint(ml_client: MLClient, endpoint_name: str):
    try:
        print(f"🗑️ Deleting endpoint '{endpoint_name}' ...")
        poller = ml_client.online_endpoints.begin_delete(endpoint_name)
        poller.result()
        print("✅ Endpoint deleted")
    except ResourceNotFoundError:
        print("ℹ️ Endpoint not found; nothing to delete")


def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    instance_type: str,
    instance_count: int,
) -> ManagedOnlineDeployment:
    # Fetch model reference
    model = ml_client.models.get(name=model_name, version=str(model_version))
    print(f"📦 Using model {model.name}:{model.version} (id={model.id})")

    dep = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,                   # Let Azure ML mount MLflow/sklearn model folder
        instance_type=instance_type,
        instance_count=instance_count,
    )

    print(f"🚀 Creating/Updating deployment '{deployment_name}' ...")
    poller = ml_client.online_deployments.begin_create_or_update(dep)
    dep = poller.result()
    print(f"✅ Deployment ready: {dep.name} (state={dep.provisioning_state})")

    # Set traffic 100% to this deployment unless traffic is handled elsewhere
    ep = ml_client.online_endpoints.get(endpoint_name)
    traffic = ep.traffic or {}
    traffic[deployment_name] = 100
    ep.traffic = traffic
    ml_client.begin_create_or_update(ep).result()
    print(f"🚦 Traffic set: {traffic}")

    return dep


def get_scoring_url_and_key(ml_client: MLClient, endpoint_name: str) -> Dict[str, str]:
    ep = ml_client.online_endpoints.get(endpoint_name)
    # get primary key
    keys = ml_client.online_endpoints.list_keys(endpoint_name)
    auth_key = keys.primary_key
    return {"url": ep.scoring_uri, "key": auth_key}


def try_smoke_test(scoring_url: str, key: str) -> bool:
    """
    Tries /score with a payload. If you have a sample payload file, set SAMPLE_PAYLOAD=path.json in env.
    Otherwise it will attempt a very generic 'split' frame with common columns.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    payload_path = os.environ.get("SAMPLE_PAYLOAD", "")
    if payload_path and os.path.exists(payload_path):
        print(f"🧪 Using sample payload file: {payload_path}")
        with open(payload_path, "r") as f:
            data = json.load(f)
    else:
        # Fallback: generic "split" payload shape expected by AzureML MLflow scoring
        # You SHOULD replace these columns with your actual training columns for a stronger test.
        columns = [
            "age", "income", "employment_length", "credit_score",
            "debt_to_income", "loan_amount",
            # engineered/example columns that your model likely expects:
            "high_income", "excellent_credit", "low_debt_ratio",
            "experienced_employment", "loan_to_income_ratio", "credit_score_to_age_ratio",
            "home_ownership", "loan_purpose", "age_category", "income_category", "credit_category"
        ]
        row = [
            35.0, 65000.0, 5.0, 720.0,
            0.25, 25000.0,
            0, 0, 1,
            1, 0.385, 20.57,
            2, 0, 2, 2, 2
        ]
        data = {"input_data": {"columns": columns, "index": [0], "data": [row]}}

    print(f"🔗 Scoring URL: {scoring_url}")
    try:
        resp = requests.post(scoring_url, headers=headers, json=data, timeout=20)
        print(f"HTTP {resp.status_code} -> {resp.text[:400]}")
        return resp.ok
    except Exception as e:
        print(f"⚠️ Smoke test request failed: {e}")
        return False


def main():
    action = env("ACTION")
    endpoint_name = env("ENDPOINT_NAME")

    ml_client = get_ml_client()

    result = {"action": action, "endpoint_name": endpoint_name, "status": "unknown"}

    if action == "delete":
        delete_endpoint(ml_client, endpoint_name)
        result["status"] = "deleted"

    elif action == "test-only":
        try:
            info = get_scoring_url_and_key(ml_client, endpoint_name)
        except ResourceNotFoundError:
            print("❌ Endpoint not found; cannot test.")
            sys.exit(1)
        ok = try_smoke_test(info["url"], info["key"])
        result["status"] = "passed" if ok else "failed"
        if not ok:
            sys.exit(1)

    elif action in ("create", "update"):
        # Read model and deployment params from env
        model_name = env("MODEL_NAME")
        model_version = env("MODEL_VERSION", required=False, default="")
        if not model_version:
            # if not provided, pick latest via selector output
            # Prefer selected_model.json if present
            if os.path.exists("selected_model.json"):
                with open("selected_model.json", "r") as f:
                    sel = json.load(f)
                model_version = sel["version"]
                if sel.get("name") and sel["name"] != model_name:
                    # sync name in case caller passed a different alias
                    model_name = sel["name"]
            else:
                # last resort: compute latest here
                models = list(ml_client.models.list(name=model_name))
                if not models:
                    print(f"❌ No models found: {model_name}")
                    sys.exit(1)
                model_version = str(sorted(models, key=lambda m: int(m.version), reverse=True)[0].version)

        deployment_name = os.environ.get("DEPLOYMENT_NAME", f"{endpoint_name}-dep")
        instance_type = env("INSTANCE_TYPE")
        instance_count = int(env("INITIAL_INSTANCE_COUNT", required=False, default="1"))

        # Ensure endpoint
        ensure_endpoint(ml_client, endpoint_name)
        # Create/Update deployment
        dep = create_or_update_deployment(
            ml_client=ml_client,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            model_name=model_name,
            model_version=model_version,
            instance_type=instance_type,
            instance_count=instance_count,
        )

        # Optionally set blue/green traffic percentage (informational only here)
        traffic_pct = int(os.environ.get("DEPLOYMENT_PERCENTAGE", "100"))
        if traffic_pct != 100:
            print(f"ℹ️ Requested blue/green traffic {traffic_pct}% — implement split as desired in your flow.")

        # Smoke test
        info = get_scoring_url_and_key(ml_client, endpoint_name)
        ok = try_smoke_test(info["url"], info["key"])

        result.update(
            {
                "status": "active" if ok else "active_with_test_warning",
                "deployment_name": dep.name,
                "model_name": model_name,
                "model_version": str(model_version),
                "instance_type": instance_type,
                "instance_count": instance_count,
                "scoring_url": info["url"],
            }
        )
        if not ok:
            print("⚠️ Smoke test failed; see logs above.")
    else:
        print(f"❌ Unsupported ACTION: {action}")
        sys.exit(1)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/deployment_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("📄 Wrote artifacts/deployment_result.json")


if __name__ == "__main__":
    main()
