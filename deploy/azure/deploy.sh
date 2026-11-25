#!/usr/bin/env bash
# Deploy longdoc to Azure Container Apps. Run it in Azure Cloud Shell (bash), already signed in:
#
#   git clone https://github.com/rahpalrah/lost-in-the-middle && cd lost-in-the-middle
#   bash deploy/azure/deploy.sh
#
# Creates: resource group, container registry, storage account + file share (for accounts and
# documents), Container Apps environment, and one app with a public HTTPS URL.
# Safe to re-run: names derive from the subscription ID and create/update calls are idempotent.
#
# Settings (environment variables, all optional):
#   SUBSCRIPTION   subscription ID or name (default: the signed-in default)
#   LOCATION       Azure region (default: eastus)
#   NAME           base name (default: longdoc)
#   DRY_RUN=1      only print the app definition; change nothing
set -euo pipefail

NAME="${NAME:-longdoc}"
LOCATION="${LOCATION:-eastus}"
TAG="v$(date +%Y%m%d%H%M%S)"
PORT=8000
REPO_URL="https://github.com/rahpalrah/lost-in-the-middle"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

log() { printf '\n==> %s\n' "$*"; }

# Render the Container Apps definition. Uses: ENV_ID IMAGE REG_BLOCK COMMAND_BLOCK ADMIN_TOKEN
# SECRETS_EXTRA PUBLIC_URL STORAGE_NAME LOCATION
render_yaml() {
  cat <<YAML
location: ${LOCATION}
properties:
  managedEnvironmentId: ${ENV_ID}
  configuration:
    activeRevisionsMode: Single
    ingress:
      external: true
      targetPort: ${PORT}
      transport: auto
      allowInsecure: false
${REG_BLOCK}
    secrets:
      - name: admin-token
        value: "${ADMIN_TOKEN}"
${SECRETS_EXTRA}
  template:
    containers:
      - name: ${NAME}
        image: ${IMAGE}
${COMMAND_BLOCK}
        resources:
          cpu: 0.5
          memory: 1Gi
        env:
          - { name: LONGDOC_STORE_DIR, value: /data/documents }
          - { name: LONGDOC_ACCOUNTS_FILE, value: /data/accounts.json }
          - { name: LONGDOC_EVENTS_FILE, value: /data/events.log }
          - { name: LONGDOC_PUBLIC_URL, value: "${PUBLIC_URL}" }
          - { name: LONGDOC_ADMIN_TOKEN, secretRef: admin-token }
          - { name: LONGDOC_TRUST_PROXY, value: "1" }
          - { name: LONGDOC_RATE_LIMIT_PER_MIN, value: "120" }
          - { name: LONGDOC_MAX_UPLOAD_MB, value: "25" }
        volumeMounts:
          - { volumeName: data, mountPath: /data }
        probes:
          - type: Startup
            httpGet: { path: /health, port: ${PORT} }
            periodSeconds: 5
            failureThreshold: 60
          - type: Liveness
            httpGet: { path: /health, port: ${PORT} }
            periodSeconds: 30
    scale:
      minReplicas: 1
      maxReplicas: 1
    volumes:
      - name: data
        storageType: AzureFile
        storageName: ${STORAGE_NAME}
        mountOptions: "uid=10001,gid=10001,dir_mode=0777,file_mode=0666,nobrl,cache=strict"
YAML
}

if [ "${DRY_RUN:-}" = "1" ]; then
  ENV_ID=/subscriptions/0/resourceGroups/rg/providers/Microsoft.App/managedEnvironments/env
  IMAGE=acr.azurecr.io/${NAME}:${TAG}; ADMIN_TOKEN=dry-run; PUBLIC_URL=https://example.invalid
  STORAGE_NAME=${NAME}storage; SECRETS_EXTRA=""; COMMAND_BLOCK=""
  REG_BLOCK=$'    registries:\n      - server: acr.azurecr.io\n        username: u\n        passwordSecretRef: acr-password'
  SECRETS_EXTRA=$'      - name: acr-password\n        value: "x"'
  render_yaml
  exit 0
fi

command -v az >/dev/null || { echo "Azure CLI not found. Run this in Azure Cloud Shell."; exit 1; }
az config set extension.use_dynamic_install=yes_without_prompt extension.dynamic_install_allow_preview=true >/dev/null

if [ -n "${SUBSCRIPTION:-}" ]; then az account set --subscription "$SUBSCRIPTION"; fi
SUB_ID="$(az account show --query id -o tsv)"
SUB_NAME="$(az account show --query name -o tsv)"
SUFFIX="$(printf '%s' "$SUB_ID" | md5sum | cut -c1-6)"
RG="${NAME}-rg"
ACR="${NAME}acr${SUFFIX}"
STORAGE="${NAME}st${SUFFIX}"
SHARE="${NAME}data"
ENV_NAME="${NAME}-env"
log "Subscription: ${SUB_NAME} (${SUB_ID}), region: ${LOCATION}"

log "Registering resource providers (first use of a subscription; can take a few minutes)"
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.Storage; do
  az provider register --namespace "$ns" --wait >/dev/null
done

log "Resource group"
az group create -n "$RG" -l "$LOCATION" -o none

log "Storage for accounts and documents"
az storage account create -n "$STORAGE" -g "$RG" -l "$LOCATION" --sku Standard_LRS \
  --kind StorageV2 --min-tls-version TLS1_2 --allow-blob-public-access false -o none
STORAGE_KEY="$(az storage account keys list -n "$STORAGE" -g "$RG" --query '[0].value' -o tsv)"
az storage share-rm create --storage-account "$STORAGE" -g "$RG" -n "$SHARE" --quota 5 -o none

log "Container Apps environment"
az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" -o none
ENV_ID="$(az containerapp env show -n "$ENV_NAME" -g "$RG" --query id -o tsv)"
STORAGE_NAME="${NAME}storage"
az containerapp env storage set -n "$ENV_NAME" -g "$RG" --storage-name "$STORAGE_NAME" \
  --azure-file-account-name "$STORAGE" --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "$SHARE" --access-mode ReadWrite -o none

REG_BLOCK=""; SECRETS_EXTRA=""; COMMAND_BLOCK=""
log "Building the image in Azure (no local Docker needed)"
if az acr create -n "$ACR" -g "$RG" -l "$LOCATION" --sku Basic --admin-enabled true -o none \
   && az acr build -r "$ACR" -t "${NAME}:${TAG}" . ; then
  LOGIN_SERVER="$(az acr show -n "$ACR" --query loginServer -o tsv)"
  ACR_USER="$(az acr credential show -n "$ACR" --query username -o tsv)"
  ACR_PASS="$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)"
  IMAGE="${LOGIN_SERVER}/${NAME}:${TAG}"
  REG_BLOCK=$'    registries:\n'"      - server: ${LOGIN_SERVER}"$'\n'"        username: ${ACR_USER}"$'\n        passwordSecretRef: acr-password'
  SECRETS_EXTRA=$'      - name: acr-password\n'"        value: \"${ACR_PASS}\""
else
  log "Registry build is not available on this subscription; running from the public Python image"
  IMAGE="docker.io/library/python:3.12-slim"
  COMMAND_BLOCK=$'        command: ["sh", "-c"]\n'"        args: [\"pip install --no-cache-dir ${REPO_URL}/archive/refs/heads/main.zip && exec longdoc serve --host 0.0.0.0 --port ${PORT}\"]"
fi

ADMIN_FILE="$HOME/${NAME}-admin-token.txt"
if [ -s "$ADMIN_FILE" ]; then ADMIN_TOKEN="$(cat "$ADMIN_FILE")"
else
  ADMIN_TOKEN="$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40)"
  (umask 077; printf '%s' "$ADMIN_TOKEN" > "$ADMIN_FILE")
fi

APP="${NAME}-app"
DOMAIN="$(az containerapp env show -n "$ENV_NAME" -g "$RG" --query properties.defaultDomain -o tsv)"
PUBLIC_URL="https://${APP}.${DOMAIN}"
YAML_FILE="$(mktemp)"; trap 'rm -f "$YAML_FILE"' EXIT
render_yaml > "$YAML_FILE"

log "Deploying the app"
if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  # Re-run: change only the image. Replacing the whole definition would erase settings and
  # secrets added since (email, alerts) because they are not in this script's YAML.
  if [ -n "$REG_BLOCK" ]; then
    az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" -o none
  else
    echo "App already exists and uses the public Python image: nothing to update."
  fi
else
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" --yaml "$YAML_FILE" -o none
fi
FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
PUBLIC_URL="https://${FQDN}"

log "Waiting for the app to answer at ${PUBLIC_URL}/health"
ok=""
for _ in $(seq 1 40); do
  if curl -fsS --max-time 5 "${PUBLIC_URL}/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 6
done
if [ -z "$ok" ]; then
  echo "The app did not answer in time. Recent logs:"
  az containerapp logs show -n "$APP" -g "$RG" --tail 50 2>&1 || true
  exit 1
fi

cat <<DONE

Deployed.
  App:        ${PUBLIC_URL}
  Health:     ${PUBLIC_URL}/health
  Register:   ${PUBLIC_URL}/register
  Admin:      ${PUBLIC_URL}/admin        (admin token is saved in ${ADMIN_FILE})
  API docs:   ${PUBLIC_URL}/openapi.json

Verification emails are NOT sent yet (no SMTP server is configured), so new sign-ups see no
email. Until you add one, read verification links from the app log:
  az containerapp logs show -n ${APP} -g ${RG} --tail 100
Add email and owner alerts: see docs/ACCOUNTS.md and run 'az containerapp update' with the
LONGDOC_SMTP_* / LONGDOC_GITHUB_* settings (put passwords in --secrets).
Delete everything later:  az group delete -n ${RG}
DONE
