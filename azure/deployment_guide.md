# ☁️ Azure Deployment Guide

Step-by-step instructions to provision all Azure services needed for the Border Surveillance AI project.

---

## Prerequisites

- Azure account (free student account at https://azure.microsoft.com/en-in/free/students)
- Azure CLI installed: https://docs.microsoft.com/cli/azure/install-azure-cli
- Azure Functions Core Tools v4: `npm install -g azure-functions-core-tools@4`

---

## Step 1 — Resource Group

All services must live in the same resource group.

```bash
az login

az group create \
  --name border-surveillance-rg \
  --location eastus
```

---

## Step 2 — Storage Account + Containers

```bash
# Create storage account (name must be globally unique, lowercase only)
az storage account create \
  --name bordersurveillanceai \
  --resource-group border-surveillance-rg \
  --location eastus \
  --sku Standard_LRS

# Get connection string — save this in config.yaml
az storage account show-connection-string \
  --name bordersurveillanceai \
  --resource-group border-surveillance-rg \
  --query connectionString -o tsv

# Create the three containers
for container in surveillance-videos processed-frames alert-logs; do
  az storage container create \
    --name $container \
    --connection-string "<YOUR_CONNECTION_STRING>"
done
```

---

## Step 3 — Cosmos DB (NoSQL)

```bash
# Create Cosmos DB account (free tier — 1000 RU/s + 25 GB)
az cosmosdb create \
  --name border-surveillance-cosmos \
  --resource-group border-surveillance-rg \
  --default-consistency-level Session \
  --enable-free-tier true

# Create database and container
az cosmosdb sql database create \
  --account-name border-surveillance-cosmos \
  --resource-group border-surveillance-rg \
  --name surveillance_db

az cosmosdb sql container create \
  --account-name border-surveillance-cosmos \
  --resource-group border-surveillance-rg \
  --database-name surveillance_db \
  --name alerts \
  --partition-key-path "/priority" \
  --throughput 400

# Get endpoint and key — save these in config.yaml
az cosmosdb show \
  --name border-surveillance-cosmos \
  --resource-group border-surveillance-rg \
  --query documentEndpoint -o tsv

az cosmosdb keys list \
  --name border-surveillance-cosmos \
  --resource-group border-surveillance-rg \
  --query primaryMasterKey -o tsv
```

---

## Step 4 — Function App

```bash
# Create Function App (Python 3.9, Consumption plan = serverless)
az functionapp create \
  --name border-surveillance-fn \
  --resource-group border-surveillance-rg \
  --storage-account bordersurveillanceai \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.9 \
  --functions-version 4 \
  --os-type linux

# Set environment variables (secrets)
az functionapp config appsettings set \
  --name border-surveillance-fn \
  --resource-group border-surveillance-rg \
  --settings \
    COSMOS_ENDPOINT="<YOUR_COSMOS_ENDPOINT>" \
    COSMOS_KEY="<YOUR_COSMOS_KEY>"

# Deploy from local
cd azure/functions
func azure functionapp publish border-surveillance-fn
```

---

## Step 5 — GitHub Actions CI/CD

1. In Azure Portal → Function App → **Deployment Center** → **Manage publish profile** → Download
2. In GitHub → repo **Settings** → **Secrets and variables** → **Actions** → New repository secret:
   - Name: `AZURE_FUNCTIONAPP_PUBLISH_PROFILE`
   - Value: paste the entire XML content of the downloaded publish profile
3. The `.github/workflows/ci_cd.yml` file will now auto-deploy on every push to `main`.

---

## Cost Estimate (Free Tier)

| Service           | Free Tier Limit                     | Typical Project Usage |
|-------------------|-------------------------------------|-----------------------|
| Blob Storage      | 5 GB                                | ~500 MB               |
| Cosmos DB         | 1,000 RU/s + 25 GB                  | ~100 MB               |
| Azure Functions   | 1,000,000 executions/month          | ~5,000 executions     |
| Computer Vision   | 5,000 transactions/month (optional) | ~1,000                |

Everything stays in the free tier for a 20-day internship project.

---

## Verify Deployment

```bash
# Test the Function endpoint
curl -X POST \
  "https://border-surveillance-fn.azurewebsites.net/api/anomaly_trigger?code=<FUNCTION_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "frame_path": "frame_00001.jpg",
    "confidence": 0.88,
    "detections": [{"class": "person", "conf": 0.88}],
    "location":   "sector_01"
  }'

# Expected response:
# {"status":"ok","alert_id":"XXXX","anomaly_score":0.44,"priority":"MEDIUM","timestamp":"..."}
```
