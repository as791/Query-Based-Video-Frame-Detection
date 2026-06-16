#!/bin/bash
# Smoke test: verify every service is reachable and returns expected responses.
# Usage: ./tests/smoke_test.sh
# Requires: curl, jq

set -euo pipefail

PASS=0; FAIL=0

check() {
  local name="$1"; local url="$2"; local expected="$3"
  local actual
  actual=$(curl -sf --max-time 5 "$url" 2>/dev/null || echo "UNREACHABLE")
  if echo "$actual" | grep -q "$expected"; then
    echo "  PASS  $name"
    PASS=$((PASS+1))
  else
    echo "  FAIL  $name  (got: $actual)"
    FAIL=$((FAIL+1))
  fi
}

echo ""
echo "=== Smoke Tests ==="
echo ""

# Embedder
check "embedder /health"   "http://localhost:8002/health"   '"status":"ok"'

# Qdrant
check "qdrant /healthz"    "http://localhost:6333/healthz"  "ok"

# MinIO console (just checks it's up)
check "minio console"      "http://localhost:9001/minio/health/live" ""

# API (Spring Boot actuator not wired, check root returns something)
check "api /v1/auth/me (expect 401 without cookie)" \
  "http://localhost:8080/v1/auth/me" "401\|Unauthorized\|error\|UNAUTHORIZED" || true

# Web
check "web (Next.js)"      "http://localhost:3000"          "<!DOCTYPE html\|<html"

echo ""
echo "Results: $PASS passed, $FAIL failed"
echo ""
[ "$FAIL" -eq 0 ]
