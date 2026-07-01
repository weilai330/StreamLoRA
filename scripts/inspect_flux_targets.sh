#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-black-forest-labs/FLUX.1-dev}
PRESET=${2:-all_common}
PYTHONPATH=src python -m streamlora.flux.inspect_flux_modules \
  --model "$MODEL" \
  --preset "$PRESET" \
  --output "reports/${PRESET}_module_match.json"
