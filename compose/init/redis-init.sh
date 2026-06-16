#!/bin/sh
set -e

REDIS_HOST="${REDIS_HOST:-redis}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "Waiting for Redis..."
until redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; do
  sleep 2
done

# Create the main pipeline stream with all three consumer groups.
# MKSTREAM creates the stream if it doesn't exist.
# '0' means consumer group reads from the beginning of the stream.
# '|| true' ignores "already exists" errors on re-runs.
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XGROUP CREATE pipeline.events cg-chunker 0 MKSTREAM || true
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XGROUP CREATE pipeline.events cg-normalizer 0 MKSTREAM || true
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XGROUP CREATE pipeline.events cg-extractor 0 MKSTREAM || true

echo "Redis Streams consumer groups ready: cg-chunker, cg-normalizer, cg-extractor"
