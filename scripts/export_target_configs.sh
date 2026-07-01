#!/usr/bin/env bash
set -euo pipefail
for preset in all_common attention_only single_stream double_stream double_image_branch double_context_branch late_single_blocks; do
  PYTHONPATH=src python -m streamlora.train.make_target_config \
    --preset "$preset" \
    --output "outputs/target_modules/${preset}.json"
done
