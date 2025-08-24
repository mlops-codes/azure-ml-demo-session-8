AZURE_CREDENTIALS: Copy the entire JSON output from step 4
AZURE_SUBSCRIPTION_ID: Your subscription ID from step 1
AZURE_RESOURCE_GROUP: Your resource group name from step 2
AZURE_ML_WORKSPACE: Your workspace name from step 3

az ad sp create-for-rbac \
  --name "github-actions-azureml-demo" \
  --role "Contributor" \
  --scopes "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/YOUR_RESOURCE_GROUP" \
  --json-auth