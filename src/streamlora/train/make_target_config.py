#!/usr/bin/env python
"""Export one StreamLoRA target preset to JSON for training launchers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from streamlora.flux.target_modules import FluxBlockSpec, preset_target_modules, summarize_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--num_double_blocks", type=int, default=19)
    parser.add_argument("--num_single_blocks", type=int, default=38)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    targets = preset_target_modules(args.preset, FluxBlockSpec(args.num_double_blocks, args.num_single_blocks))
    payload = {"preset": args.preset, "target_modules": targets, "summary": summarize_targets(targets)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(targets)} targets -> {args.output}")


if __name__ == "__main__":
    main()
