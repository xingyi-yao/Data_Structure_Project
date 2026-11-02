#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m ui.model_service "$@"
