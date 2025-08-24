#!/usr/bin/env python3
import json
import os
from pathlib import Path

def main():
    cfg = {
        "azure_ml": {
            "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
            "resource_group": os.environ["AZURE_RESOURCE_GROUP"],
            "workspace_name": os.environ["AZURE_ML_WORKSPACE"],
            "region": os.environ.get("AZURE_REGION", "eastus"),
        },
        "experiment": {
            "endpoint_name": os.environ["ENDPOINT_NAME"],
            "deployment_name": f'{os.environ["ENDPOINT_NAME"]}-deployment-{os.environ.get("GITHUB_RUN_NUMBER","local")}',
        },
        "deployment": {
            "instance_type": os.environ.get("INSTANCE_TYPE", "Standard_DS2_v2"),
            "instance_count": int(os.environ.get("INITIAL_INSTANCE_COUNT", "1")),
            "auto_scale": {
                "enabled": os.environ.get("ENABLE_AUTOSCALING", "false").lower() == "true",
                "min_instances": int(os.environ.get("MIN_CAPACITY", "1") or "1"),
                "max_instances": int(os.environ.get("MAX_CAPACITY", "5") or "5"),
            },
            "traffic_percentage": int(os.environ.get("DEPLOYMENT_PERCENTAGE", "100") or "100"),
        },
        "environment": os.environ.get("ENVIRONMENT", "dev"),
    }

    Path("artifacts").mkdir(exist_ok=True)
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("✅ config.json written:")
    print(json.dumps(cfg, indent=2))

if __name__ == "__main__":
    main()
