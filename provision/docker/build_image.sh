#!/usr/bin/env bash
# Build the CUA-Gym desktop Docker image.
# Run from the CUA-Gym repo root:
#   bash provision/docker/build_image.sh
set -euo pipefail

IMAGE_NAME="${1:-cua-gym-desktop:latest}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Building ${IMAGE_NAME} from ${REPO_ROOT} ..."
docker build \
  --platform linux/amd64 \
  -f "${REPO_ROOT}/provision/docker/Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${REPO_ROOT}"

echo "Done. Image: ${IMAGE_NAME}"
