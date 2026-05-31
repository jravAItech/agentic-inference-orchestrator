#!/usr/bin/env bash
# Provision + deploy on Azure. Run line-by-line the first time so you see
# each resource get created (good for learning the CLI). Free/low tiers used
# throughout. Requires: az login (Azure CLI).
set -euo pipefail

# ---- 0. Variables (edit these) ----
RG=agentic-orch-rg
LOC=eastus
STORAGE=agentorchstore$RANDOM
SEARCH=agentorch-search
ACR=agentorchacr$RANDOM
ACA_ENV=agentorch-env
APP=agentic-orchestrator

# ---- 1. Resource group ----
az group create --name "$RG" --location "$LOC"

# ---- 2. Blob storage (corpus + incident inputs) ----
az storage account create --name "$STORAGE" --resource-group "$RG" \
  --location "$LOC" --sku Standard_LRS
CONN=$(az storage account show-connection-string --name "$STORAGE" \
  --resource-group "$RG" --query connectionString -o tsv)
az storage container create --name runbooks --connection-string "$CONN"
echo "AZURE_STORAGE_CONNECTION_STRING=$CONN"   # set this in the app env

# ---- 3. Azure AI Search (free tier: --sku free) ----
az search service create --name "$SEARCH" --resource-group "$RG" \
  --sku free --location "$LOC"
SEARCH_KEY=$(az search admin-key show --service-name "$SEARCH" \
  --resource-group "$RG" --query primaryKey -o tsv)
echo "AZURE_SEARCH_ENDPOINT=https://$SEARCH.search.windows.net"
echo "AZURE_SEARCH_KEY=$SEARCH_KEY"
# NOTE: create the 'runbooks' index with a semantic config named 'default'
# in the portal or via the REST API, then index the Blob documents.

# ---- 4. Container registry + build image ----
az acr create --name "$ACR" --resource-group "$RG" --sku Basic --admin-enabled true
az acr build --registry "$ACR" --image "$APP:v1" .

# ---- 5. Container Apps environment + deploy ----
az extension add --name containerapp --upgrade
az containerapp env create --name "$ACA_ENV" --resource-group "$RG" \
  --location "$LOC"
az containerapp create --name "$APP" --resource-group "$RG" \
  --environment "$ACA_ENV" \
  --image "$ACR.azurecr.io/$APP:v1" \
  --registry-server "$ACR.azurecr.io" \
  --target-port 8000 --ingress external \
  --env-vars \
    "ANTHROPIC_API_KEY=secretref:anthropic-key" \
    "CORPUS_BACKEND=azure_blob" \
    "RETRIEVER_BACKEND=azure_search" \
    "AZURE_STORAGE_CONNECTION_STRING=$CONN" \
    "AZURE_SEARCH_ENDPOINT=https://$SEARCH.search.windows.net" \
    "AZURE_SEARCH_KEY=$SEARCH_KEY"

echo "Deployed. Get the URL:"
az containerapp show --name "$APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv
