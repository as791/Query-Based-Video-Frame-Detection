#!/bin/sh
set -e

echo "Waiting for MinIO to be ready..."
until mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" 2>/dev/null; do
  sleep 2
done

echo "Creating bucket: $MINIO_BUCKET"
mc mb --ignore-existing "local/$MINIO_BUCKET"

# Block all public access
mc anonymous set none "local/$MINIO_BUCKET"

echo "MinIO bucket '$MINIO_BUCKET' ready."
