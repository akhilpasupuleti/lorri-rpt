#!/bin/bash
set -e

# CONFIG
LOW_SPACE_THRESHOLD_MB=2048   # Warn if available space < 2GB

# Step 1: Check disk space
AVAILABLE_MB=$(df --output=avail / | tail -1)
AVAILABLE_MB=$((AVAILABLE_MB / 1024))

echo "Available disk space: ${AVAILABLE_MB} MB"

if [ "$AVAILABLE_MB" -lt "$LOW_SPACE_THRESHOLD_MB" ]; then
  echo "WARNING: Low disk space (< ${LOW_SPACE_THRESHOLD_MB} MB). Consider running: docker system prune -a --volumes"
  read -p "Continue anyway? (y/N): " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborting deploy due to low space."
    exit 1
  fi
fi

# Step 2: Pull latest code
echo "Pulling latest from release..."
git checkout release
git fetch origin
git pull origin release

# Step 3: Stop current containers
echo "Stopping and removing old containers..."
docker compose down

# Step 4: Build new image
echo "Building new image..."
docker compose build

# Step 5: Bring up new container
echo "Starting updated container..."
docker compose up -d

# Step 6: Done
echo "Redeployment complete."
echo "To see logs: docker compose logs -f"
