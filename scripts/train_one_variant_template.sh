#!/usr/bin/env bash
set -euo pipefail
# Template launcher. The exact training backend can be Diffusers' FLUX LoRA script
# or SimpleTuner/ai-toolkit. Use scripts/export_target_configs.sh to generate
# explicit target module lists and pass them to your backend.
CONFIG=${1:-configs/t2_attention_only.yaml}
echo "TODO: wire $CONFIG into selected FLUX LoRA backend after confirming local training environment."
echo "First run: bash scripts/export_target_configs.sh"
