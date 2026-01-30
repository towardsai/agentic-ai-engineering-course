# Nova MCP Server - GCP Cloud Run Deployment Guide

This guide walks you through deploying the Nova MCP Server to Google Cloud Run with Cloud SQL PostgreSQL.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Google Cloud Platform                     │
│                                                                   │
│  ┌─────────────────┐      ┌─────────────────┐                    │
│  │ Artifact        │      │  Secret         │                    │
│  │ Registry        │      │  Manager        │                    │
│  │ (Docker images) │      │  (API keys)     │                    │
│  └────────┬────────┘      └────────┬────────┘                    │
│           │                        │                              │
│           ▼                        ▼                              │
│  ┌─────────────────────────────────────────┐                     │
│  │              Cloud Run                   │                     │
│  │         (Nova MCP Server)                │                     │
│  └────────────────┬────────────────────────┘                     │
│                   │                                               │
│                   │ Cloud SQL Connector                           │
│                   ▼                                               │
│  ┌─────────────────────────────────────────┐                     │
│  │            Cloud SQL                     │                     │
│  │         (PostgreSQL 16)                  │                     │
│  └─────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Workload Identity Federation
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                            │
│                    (Manual Deployment)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- A Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured
- GitHub repository with the Nova codebase
- The following API keys ready:
  - `GOOGLE_API_KEY` (Gemini)
  - `PPLX_API_KEY` (Perplexity)
  - `FIRECRAWL_API_KEY` (Firecrawl)
  - Optional: `GITHUB_TOKEN`, `OPENAI_API_KEY`, `OPIK_API_KEY`

## Step 1: Set Up GCP Project

### 1.1 Set Environment Variables

```bash
# Replace with your values
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"  # or your preferred region
export GITHUB_ORG="your-github-org"  # or username
export GITHUB_REPO="your-repo-name"

# Set the project
gcloud config set project $PROJECT_ID
```

### 1.2 Enable Required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com
```

## Step 2: Create Artifact Registry Repository

```bash
gcloud artifacts repositories create nova-mcp \
  --repository-format=docker \
  --location=$REGION \
  --description="Nova MCP Server Docker images"
```

## Step 3: Create Cloud SQL Instance

### 3.1 Create PostgreSQL Instance

```bash
# Create a Cloud SQL PostgreSQL 16 instance (Enterprise edition for cost savings)
gcloud sql instances create nova-postgres \
  --database-version=POSTGRES_16 \
  --edition=ENTERPRISE \
  --tier=db-f1-micro \
  --region=$REGION \
  --storage-type=SSD \
  --storage-size=10GB \
  --availability-type=zonal \
  --backup-start-time=03:00 \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=04

# Note: For production, consider:
# - --edition=ENTERPRISE_PLUS with --tier=db-perf-optimized-N-2 (or larger)
# - --availability-type=regional (for HA)
# - --storage-size=50GB (or larger)
```

### 3.2 Create Database and User

```bash
# Create the database
gcloud sql databases create nova_research \
  --instance=nova-postgres

# Generate a secure password
DB_PASSWORD=$(openssl rand -base64 24)
echo "Save this password securely: $DB_PASSWORD"

# Create the database user
gcloud sql users create nova \
  --instance=nova-postgres \
  --password=$DB_PASSWORD
```

### 3.3 Get Connection Name

```bash
# Get the instance connection name (needed for Cloud Run)
gcloud sql instances describe nova-postgres \
  --format="value(connectionName)"

# Output format: PROJECT_ID:REGION:INSTANCE_NAME
# Example: my-project:us-central1:nova-postgres
```

## Step 4: Create Secrets in Secret Manager

Store all sensitive configuration in Secret Manager:

```bash
# Database credentials
echo -n "nova" | gcloud secrets create db-user --data-file=-
echo -n "$DB_PASSWORD" | gcloud secrets create db-password --data-file=-

# API Keys (replace with your actual keys)
echo -n "your-google-api-key" | gcloud secrets create google-api-key --data-file=-
echo -n "your-pplx-api-key" | gcloud secrets create pplx-api-key --data-file=-
echo -n "your-firecrawl-api-key" | gcloud secrets create firecrawl-api-key --data-file=-

# Descope authentication (required for authentication to work)
echo -n "your-descope-project-id" | gcloud secrets create descope-project-id --data-file=-

# Optional secrets
echo -n "your-github-token" | gcloud secrets create github-token --data-file=-
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-
echo -n "your-opik-api-key" | gcloud secrets create opik-api-key --data-file=-
```

## Step 5: Set Up Workload Identity Federation

This allows GitHub Actions to authenticate with GCP without storing service account keys.

### 5.1 Create Workload Identity Pool

```bash
gcloud iam workload-identity-pools create "github-actions-pool" \
  --location="global" \
  --display-name="GitHub Actions Pool"
```

### 5.2 Create Workload Identity Provider

```bash
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == '${GITHUB_ORG}'" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

> **Note**: The `--attribute-condition` is required and restricts authentication to repositories owned by your GitHub organization/user. Replace `${GITHUB_ORG}` with your actual GitHub username or organization name if not using the environment variable.

### 5.3 Create Service Account for Deployments

```bash
# Create the service account
gcloud iam service-accounts create github-actions-deployer \
  --display-name="GitHub Actions Deployer"

# Get the service account email
SA_EMAIL="github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
```

### 5.4 Grant Required Permissions

```bash
# Cloud Run Admin (deploy services)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/run.admin"

# Artifact Registry Writer (push images)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.writer"

# Service Account User (to use the Cloud Run service account)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"

# Secret Manager Accessor (to read secrets during deployment)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/secretmanager.secretAccessor"

# Cloud SQL Client (for database connections)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/cloudsql.client"
```

### 5.5 Allow GitHub to Impersonate the Service Account

```bash
# Get the Workload Identity Pool ID
POOL_ID=$(gcloud iam workload-identity-pools describe github-actions-pool \
  --location="global" \
  --format="value(name)")

# IMPORTANT: Verify your repository name matches exactly!
# Run: git remote -v
# The format should be: GITHUB_ORG/GITHUB_REPO (e.g., myorg/my-repo)
echo "Binding for repository: ${GITHUB_ORG}/${GITHUB_REPO}"

# Allow GitHub Actions from your repository to use this service account
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"

# Also grant token creator role (required for obtaining access tokens)
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --role="roles/iam.serviceAccountTokenCreator" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"
```

> **Critical**: The repository name in the binding must **exactly match** your GitHub repository (case-sensitive). Verify with `git remote -v` before running these commands. A mismatch will cause "Permission denied" errors.

> **Important**: IAM changes can take up to 7 minutes to propagate. Wait a few minutes before running the GitHub Actions workflow.

### 5.6 Get the Workload Identity Provider Resource Name

```bash
# This value is needed for GitHub Actions
gcloud iam workload-identity-pools providers describe github-provider \
  --location="global" \
  --workload-identity-pool="github-actions-pool" \
  --format="value(name)"

# Output format:
# projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-actions-pool/providers/github-provider
```

## Step 6: Configure Cloud Run Service Account

The Cloud Run service needs its own permissions to access Cloud SQL and secrets:

```bash
# Get the project number (Cloud Run uses Compute Engine default service account)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# The default Cloud Run service account is the Compute Engine default
CLOUDRUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant Cloud SQL Client role
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$CLOUDRUN_SA" \
  --role="roles/cloudsql.client"

# Grant Secret Manager access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$CLOUDRUN_SA" \
  --role="roles/secretmanager.secretAccessor"
```

## Step 7: Configure GitHub Repository

### 7.1 Add Repository Variables

Go to your GitHub repository → Settings → Secrets and variables → Actions → Variables tab.

Add the following repository variables:

| Variable Name | Value | Example |
|---------------|-------|---------|
| `GCP_PROJECT_ID` | Your GCP project ID | `my-project-123` |
| `GCP_REGION` | Your preferred region | `us-central1` |
| `CLOUD_SQL_INSTANCE` | Cloud SQL connection name | `my-project:us-central1:nova-postgres` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full provider resource name | `projects/123456/locations/global/workloadIdentityPools/github-actions-pool/providers/github-provider` |
| `GCP_SERVICE_ACCOUNT` | Deployer service account email | `github-actions-deployer@my-project.iam.gserviceaccount.com` |
| `SERVICE_URL` | Public URL of the deployed service | `https://nova-mcp-server-72470131094.us-central1.run.app` |

> **Note**: The `SERVICE_URL` variable is required for Descope authentication to work. After your first deployment, get the service URL with:
> ```bash
> gcloud run services describe nova-mcp-server --region=$REGION --format="value(status.url)"
> ```
> Then add it as a repository variable before redeploying to enable authentication.

### 7.2 Verify Configuration

You can verify your configuration with:

```bash
# Get all values needed for GitHub
echo "GCP_PROJECT_ID: $PROJECT_ID"
echo "GCP_REGION: $REGION"
echo "CLOUD_SQL_INSTANCE: $(gcloud sql instances describe nova-postgres --format='value(connectionName)')"
echo "GCP_WORKLOAD_IDENTITY_PROVIDER: $(gcloud iam workload-identity-pools providers describe github-provider --location=global --workload-identity-pool=github-actions-pool --format='value(name)')"
echo "GCP_SERVICE_ACCOUNT: github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
echo "SERVICE_URL: (will be available after first deployment - see note above)"
```

## Step 8: Deploy

### 8.1 Trigger the Deployment

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Select "Deploy Nova MCP Server" workflow
4. Click "Run workflow"
5. Select the branch (usually `main`)
6. Click "Run workflow"

### 8.2 Monitor the Deployment

The workflow will:
1. Authenticate with GCP using Workload Identity Federation
2. Build the Docker image with Cloud Run configuration
3. Push the image to Artifact Registry
4. Deploy to Cloud Run with Cloud SQL connection
5. Run a smoke test

## Step 9: Verify Deployment

### 9.1 Check Cloud Run Service

```bash
# Get the service URL
gcloud run services describe nova-mcp-server \
  --region=$REGION \
  --format="value(status.url)"

# Check service status
gcloud run services describe nova-mcp-server \
  --region=$REGION
```

### 9.2 View Logs

```bash
# Stream logs
gcloud run services logs tail nova-mcp-server --region=$REGION

# Or view in Cloud Console
echo "https://console.cloud.google.com/run/detail/${REGION}/nova-mcp-server/logs?project=${PROJECT_ID}"
```

### 9.3 Test the Service

```bash
SERVICE_URL=$(gcloud run services describe nova-mcp-server --region=$REGION --format="value(status.url)")

# Test the MCP endpoint
curl -X POST "${SERVICE_URL}/" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

## Troubleshooting

### Database Connection Issues

```bash
# Check if Cloud SQL is accessible
gcloud sql instances describe nova-postgres

# Check Cloud Run service logs for connection errors
gcloud run services logs read nova-mcp-server --region=$REGION --limit=50

# Verify the Cloud SQL instance connection name matches
gcloud sql instances describe nova-postgres --format="value(connectionName)"
```

### Secret Access Issues

```bash
# Verify secrets exist
gcloud secrets list

# Check if the service account has access
gcloud secrets get-iam-policy db-password

# Test secret access
gcloud secrets versions access latest --secret=db-password
```

### Deployment Failures

**Docker build fails with "file not found"**: Check `.dockerignore` - it may be excluding required files like `docker-entrypoint-cloudrun.sh`. The `.dockerignore` should include exceptions:
```
*.sh
!docker-entrypoint.sh
!docker-entrypoint-cloudrun.sh
```

```bash
# Check Cloud Build logs (if using Cloud Build)
gcloud builds list --limit=5

# Check Artifact Registry for the image
gcloud artifacts docker images list \
  $REGION-docker.pkg.dev/$PROJECT_ID/nova-mcp/nova-mcp-server

# Verify IAM permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:github-actions-deployer@"
```

### Workload Identity Federation Issues

If you get `Permission 'iam.serviceAccounts.getAccessToken' denied`:

```bash
# 1. Verify the pool and provider exist
gcloud iam workload-identity-pools list --location=global
gcloud iam workload-identity-pools providers list \
  --location=global \
  --workload-identity-pool=github-actions-pool

# 2. Check service account IAM bindings - verify repository name matches!
gcloud iam service-accounts get-iam-policy \
  github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com

# 3. Verify your actual repository name
git remote -v
# Should show: git@github.com:OWNER/REPO.git

# 4. If repository name is wrong in bindings, fix it:
# First get the pool ID
POOL_ID=$(gcloud iam workload-identity-pools describe github-actions-pool \
  --location="global" --format="value(name)")

# Remove wrong binding (replace WRONG_ORG/WRONG_REPO with actual wrong values)
gcloud iam service-accounts remove-iam-policy-binding \
  github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/WRONG_ORG/WRONG_REPO"

# Add correct binding (use actual GITHUB_ORG and GITHUB_REPO)
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"

gcloud iam service-accounts add-iam-policy-binding \
  github-actions-deployer@${PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.serviceAccountTokenCreator" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}"
```

> **Note**: Wait 5-7 minutes after IAM changes for propagation before retrying.

### Authentication Issues

If Descope authentication is not working (clients connect without authentication prompt):

**Symptoms:**
- MCP clients connect successfully without authentication
- No OAuth callback flow is triggered
- Server logs show "🔐 Authentication disabled (missing configuration)"

**Cause:** Missing `SERVER_URL` or `DESCOPE_PROJECT_ID` configuration.

**Solution:**

```bash
# 1. Verify the Descope secret exists
gcloud secrets versions access latest --secret=descope-project-id

# 2. Check if SERVER_URL is set in GitHub repository variables
# Go to: GitHub repo → Settings → Secrets and variables → Actions → Variables
# Ensure SERVICE_URL variable is set to your Cloud Run URL

# 3. Get your actual service URL
gcloud run services describe nova-mcp-server \
  --region=$REGION \
  --format="value(status.url)"
# Example output: https://nova-mcp-server-72470131094.us-central1.run.app

# 4. Add SERVICE_URL as a GitHub repository variable with the URL from step 3

# 5. Redeploy the service via GitHub Actions

# 6. Verify environment variables in the deployed service
gcloud run services describe nova-mcp-server \
  --region=$REGION \
  --format="yaml(spec.template.spec.containers[0].env)"

# Look for SERVER_URL and DESCOPE_PROJECT_ID in the output
# Both must be present for authentication to work

# 7. Check the service logs after redeployment
gcloud run services logs read nova-mcp-server --region=$REGION --limit=20

# Look for: "🔐 Descope authentication enabled"
# If you see "🔐 Authentication disabled", check the env vars again
```

**Post-deployment checklist:**
1. ✅ `descope-project-id` secret created in Secret Manager
2. ✅ `SERVICE_URL` variable added to GitHub repository variables
3. ✅ Service redeployed after adding `SERVICE_URL`
4. ✅ Server logs show "🔐 Descope authentication enabled"
5. ✅ MCP client triggers OAuth flow when connecting

## Cost Optimization

### Cloud Run
- Uses scale-to-zero (no cost when idle)
- Set `--min-instances=0` for development
- Set `--min-instances=1` for production to avoid cold starts

### Cloud SQL
- `db-f1-micro` tier with ENTERPRISE edition is cheapest (~$10/month)
- For production, consider ENTERPRISE_PLUS with `db-perf-optimized-N-2`
- Use scheduled start/stop for dev instances

### Artifact Registry
- Old images are retained indefinitely
- Set up cleanup policy to delete old images:

```bash
gcloud artifacts repositories set-cleanup-policy nova-mcp \
  --location=$REGION \
  --policy=delete-tagged \
  --tag-state=untagged \
  --older-than=30d
```

## Security Best Practices

1. **Never commit secrets** - All API keys should be in Secret Manager
2. **Use Workload Identity** - No service account keys in GitHub
3. **Least privilege** - Only grant necessary IAM roles
4. **Private networking** - Consider VPC connector for production
5. **Enable Cloud Audit Logs** - Monitor access to resources
6. **Regular rotation** - Rotate database passwords and API keys periodically

## Updating the Deployment

To deploy a new version:
1. Push code changes to the repository
2. Go to Actions → Deploy Nova MCP Server → Run workflow
3. The new image will be built and deployed automatically

To rollback:
```bash
# List revisions
gcloud run revisions list --service=nova-mcp-server --region=$REGION

# Route traffic to a previous revision
gcloud run services update-traffic nova-mcp-server \
  --region=$REGION \
  --to-revisions=REVISION_NAME=100
```

## Local Development vs Cloud Run

| Aspect | Local (Docker Compose) | Cloud Run |
|--------|------------------------|-----------|
| Dockerfile | `Dockerfile` | `Dockerfile.cloudrun` |
| Database | Local PostgreSQL container | Cloud SQL |
| Port | 8000 | 8080 (Cloud Run default) |
| Entrypoint | `docker-entrypoint.sh` | `docker-entrypoint-cloudrun.sh` |
| Secrets | `.env` file | Secret Manager |
| Scaling | Single instance | 0-10 instances |

The codebase automatically detects the environment:
- If `CLOUD_SQL_INSTANCE` is set → Uses Cloud SQL Python Connector
- If not set → Uses `DATABASE_URL` for local PostgreSQL

### File Structure

```
src/nova/mcp_server/
├── Dockerfile                      # For local development (docker compose)
├── Dockerfile.cloudrun             # For Cloud Run deployment
├── docker-entrypoint.sh            # Local startup script
├── docker-entrypoint-cloudrun.sh   # Cloud Run startup script
├── docker-compose.yml              # Local development orchestration
└── ...
```

