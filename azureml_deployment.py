"""
Azure ML model deployment for loan approval model.
"""

import os
import json
import time
import requests
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

class AzureMLDeployer:
    def __init__(self, config_path='config.json'):
        """Initialize Azure ML deployer."""
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
        
        print(f"Connected to Azure ML workspace: {self.config['azure_ml']['workspace_name']}")

    def create_endpoint(self, endpoint_name=None):
        """Create managed online endpoint."""
        if endpoint_name is None:
            endpoint_name = self.config['experiment']['endpoint_name']
        
        print(f"Creating endpoint: {endpoint_name}")
        
        # Create endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Loan approval prediction endpoint",
            auth_mode="key"
        )
        
        # Create or update endpoint
        endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Endpoint created: {endpoint_result.name}")
        
        return endpoint_result

    def create_deployment(self, model_name, model_version, endpoint_name=None, deployment_name=None):
        """Deploy model to endpoint."""
        if endpoint_name is None:
            endpoint_name = self.config['experiment']['endpoint_name']
        if deployment_name is None:
            deployment_name = self.config['experiment']['deployment_name']
        
        print(f"Creating deployment: {deployment_name}")
        
        # Get deployment configuration
        deploy_config = self.config['deployment']
        
        # Create inference environment
        inference_env = Environment(
            name="loan-approval-inference-env",
            description="Environment for loan approval model inference",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file="inference_environment.yml"
        )
        
        # Create or update environment
        env = self.ml_client.environments.create_or_update(inference_env)
        
        # Create deployment
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=f"{model_name}:{model_version}",
            environment=inference_env,
            code_configuration=CodeConfiguration(
                code="./scripts",
                scoring_script="inference.py"
            ),
            instance_type=deploy_config.get('instance_type', 'Standard_DS2_v2'),
            instance_count=deploy_config['instance_count']
        )
        
        # Deploy model
        deployment_result = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        print(f"Deployment created: {deployment_result.name}")
        
        # Set traffic to 100% for this deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        print(f"Traffic set to 100% for deployment: {deployment_name}")
        
        return deployment_result

    def test_endpoint(self, endpoint_name=None, test_data=None):
        """Test the deployed endpoint."""
        if endpoint_name is None:
            endpoint_name = self.config['experiment']['endpoint_name']
        
        # Get endpoint
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        
        # Default test data
        if test_data is None:
            test_data = [
                {
                    "age": 35,
                    "income": 65000,
                    "employment_length": 5.2,
                    "credit_score": 720,
                    "debt_to_income": 0.25,
                    "loan_amount": 25000,
                    "home_ownership": "MORTGAGE",
                    "loan_purpose": "debt_consolidation"
                },
                {
                    "age": 28,
                    "income": 45000,
                    "employment_length": 2.5,
                    "credit_score": 650,
                    "debt_to_income": 0.35,
                    "loan_amount": 15000,
                    "home_ownership": "RENT",
                    "loan_purpose": "home_improvement"
                }
            ]
        
        print("Testing endpoint with sample data...")
        
        # Test endpoint
        try:
            response = self.ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                request_file=None,
                deployment_name=None
            )
            
            print("Endpoint test successful!")
            print(f"Response: {response}")
            
            return response
            
        except Exception as e:
            print(f"Endpoint test failed: {str(e)}")
            return None

    def get_endpoint_logs(self, endpoint_name=None, deployment_name=None, lines=50):
        """Get deployment logs."""
        if endpoint_name is None:
            endpoint_name = self.config['experiment']['endpoint_name']
        if deployment_name is None:
            deployment_name = self.config['experiment']['deployment_name']
        
        try:
            logs = self.ml_client.online_deployments.get_logs(
                name=deployment_name,
                endpoint_name=endpoint_name,
                lines=lines
            )
            
            print(f"Logs for deployment {deployment_name}:")
            print(logs)
            
            return logs
            
        except Exception as e:
            print(f"Failed to get logs: {str(e)}")
            return None

    def delete_endpoint(self, endpoint_name=None):
        """Delete endpoint and all deployments."""
        if endpoint_name is None:
            endpoint_name = self.config['experiment']['endpoint_name']
        
        print(f"Deleting endpoint: {endpoint_name}")
        
        try:
            self.ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
            print(f"Endpoint {endpoint_name} deleted successfully")
            
        except Exception as e:
            print(f"Failed to delete endpoint: {str(e)}")

    def list_endpoints(self):
        """List all endpoints in the workspace."""
        print("Listing all endpoints...")
        
        endpoints = self.ml_client.online_endpoints.list()
        
        for endpoint in endpoints:
            print(f"Endpoint: {endpoint.name}")
            print(f"  Status: {endpoint.provisioning_state}")
            print(f"  URL: {endpoint.scoring_uri}")
            print(f"  Created: {endpoint.creation_context.created_at}")
            print()

    def get_endpoint_details(self, endpoint_name=None):
        """Get detailed information about an endpoint."""
        if endpoint_name is None:
            endpoint_name = self.config['experiment']['endpoint_name']
        
        try:
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            
            print(f"Endpoint Details: {endpoint_name}")
            print(f"  Status: {endpoint.provisioning_state}")
            print(f"  Scoring URI: {endpoint.scoring_uri}")
            print(f"  Swagger URI: {endpoint.openapi_uri}")
            print(f"  Traffic: {endpoint.traffic}")
            
            # List deployments
            deployments = self.ml_client.online_deployments.list(endpoint_name=endpoint_name)
            print("  Deployments:")
            for deployment in deployments:
                print(f"    - {deployment.name}: {deployment.provisioning_state}")
            
            return endpoint
            
        except Exception as e:
            print(f"Failed to get endpoint details: {str(e)}")
            return None

def create_inference_environment():
    """Create inference environment YAML file."""
    env_content = """name: loan-approval-inference
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pip
  - pip:
    - azure-ai-ml>=1.12.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
    - joblib>=1.3.0
"""
    
    with open('inference_environment.yml', 'w') as f:
        f.write(env_content)
    
    print("Inference environment file created: inference_environment.yml")

def create_training_environment():
    """Create training environment YAML file."""
    env_content = """name: loan-approval-training
channels:
  - conda-forge
dependencies:
  - python=3.9
  - pip
  - pip:
    - azure-ai-ml>=1.12.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scikit-learn>=1.3.0
    - joblib>=1.3.0
    - mlflow>=2.8.0
    - matplotlib>=3.7.0
    - seaborn>=0.12.0
"""
    
    with open('environment.yml', 'w') as f:
        f.write(env_content)
    
    print("Training environment file created: environment.yml")

def main():
    """Main deployment function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure ML Deployment')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--create-endpoint', action='store_true', help='Create endpoint')
    parser.add_argument('--deploy-model', nargs=2, metavar=('MODEL_NAME', 'MODEL_VERSION'), 
                       help='Deploy model (provide model name and version)')
    parser.add_argument('--test-endpoint', action='store_true', help='Test endpoint')
    parser.add_argument('--get-logs', action='store_true', help='Get deployment logs')
    parser.add_argument('--list-endpoints', action='store_true', help='List all endpoints')
    parser.add_argument('--endpoint-details', action='store_true', help='Get endpoint details')
    parser.add_argument('--delete-endpoint', action='store_true', help='Delete endpoint')
    parser.add_argument('--create-env-files', action='store_true', help='Create environment YAML files')
    
    args = parser.parse_args()
    
    if args.create_env_files:
        create_inference_environment()
        create_training_environment()
        return
    
    # Initialize deployer
    deployer = AzureMLDeployer(args.config)
    
    if args.create_endpoint:
        endpoint = deployer.create_endpoint()
        print(f"Endpoint created: {endpoint.name}")
    
    if args.deploy_model:
        model_name, model_version = args.deploy_model
        deployment = deployer.create_deployment(model_name, model_version)
        print(f"Model deployed: {deployment.name}")
    
    if args.test_endpoint:
        response = deployer.test_endpoint()
        if response:
            print(f"Test response: {response}")
    
    if args.get_logs:
        logs = deployer.get_endpoint_logs()
    
    if args.list_endpoints:
        deployer.list_endpoints()
    
    if args.endpoint_details:
        details = deployer.get_endpoint_details()
    
    if args.delete_endpoint:
        deployer.delete_endpoint()

if __name__ == "__main__":
    main()