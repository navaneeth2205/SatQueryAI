#!/usr/bin/env bash
# ==============================================================================
# SatQuery AI — Standalone Docker Packaging & Live Runtime Validation Script
# ==============================================================================
# This script builds the production Docker container, starts the web service,
# validates the /api/catalog healthcheck, and runs end-to-end inference queries
# with strict status and payload schema assertions.
# ==============================================================================

set -euo pipefail

IMAGE_NAME="satquery:validation"
CONTAINER_NAME="satquery-val-test"
PORT=5000
TMP_DIR="${TMPDIR:-/tmp}"

echo "=== 1. Building Production Docker Image (CPU Profile) ==="
docker build --build-arg TARGET_ENV=cpu -t "${IMAGE_NAME}" .

echo "=== 2. Starting Containerized Application ==="
# Stop and remove any prior lingering container
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run -d --name "${CONTAINER_NAME}" -p "${PORT}:5000" "${IMAGE_NAME}"

cleanup() {
    echo "=== Teardown: Stopping and removing container ==="
    docker logs "${CONTAINER_NAME}" --tail 25 2>/dev/null || true
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    rm -f "${TMP_DIR}/satquery_cat.json" "${TMP_DIR}/satquery_route.json" "${TMP_DIR}/satquery_unavail.json" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== 3. Awaiting Service Healthcheck (http://localhost:${PORT}/api/catalog) ==="
HEALTHY=false
for i in {1..20}; do
    STATUS=$(curl -s -o "${TMP_DIR}/satquery_cat.json" -w "%{http_code}" "http://localhost:${PORT}/api/catalog" || true)
    if [ "${STATUS}" = "200" ]; then
        echo "Healthcheck passed on attempt ${i}!"
        # Assert valid JSON and catalog metadata
        python3 -c "import json; data = json.load(open('${TMP_DIR}/satquery_cat.json')); assert data.get('status') == 'success', f'Catalog response status invalid: {data}'; assert 'preset_queries' in data; assert 'total_cd_scenes' in data"
        HEALTHY=true
        break
    fi
    echo "Attempt ${i}: HTTP status ${STATUS}. Retrying in 2 seconds..."
    sleep 2
done

if [ "${HEALTHY}" != "true" ]; then
    echo "ERROR: Service failed to start within timeout."
    docker logs "${CONTAINER_NAME}" --tail 50 || true
    exit 1
fi

echo "=== 4. Live Container Inference Test — Task Routing ==="
curl -fsS -X POST "http://localhost:${PORT}/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Detect surface changes between before and after flood imagery", "mode": "query_only"}' \
    | tee "${TMP_DIR}/satquery_route.json"
echo ""

# Explicit assertion on routing response
python3 -c "import json; res = json.load(open('${TMP_DIR}/satquery_route.json')); assert res.get('status') in ('SUCCESS', 'success'), f'Routing status unexpected: {res}'; assert 'predictions' in res, 'Missing predictions in response'; assert 'evidence_report' in res, 'Missing evidence_report in response'"
echo "--> Task Routing inference verified: Status SUCCESS, predictions present."

echo "=== 5. Live Container Inference Test — Unsupported Query Handling ==="
curl -fsS -X POST "http://localhost:${PORT}/api/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Count the number of vessels in the harbor", "mode": "query_only"}' \
    | tee "${TMP_DIR}/satquery_unavail.json"
echo ""

# Explicit assertion on unsupported query response
python3 -c "import json; res = json.load(open('${TMP_DIR}/satquery_unavail.json')); assert res.get('status') == 'UNAVAILABLE', f'Expected UNAVAILABLE status, got: {res}'"
echo "--> Unsupported task handling verified: Status UNAVAILABLE."

echo "=== 6. Recent Container Logs ==="
docker logs "${CONTAINER_NAME}" --tail 25 || true

echo "=== Live Docker Container Validation SUCCESSFUL! ==="
