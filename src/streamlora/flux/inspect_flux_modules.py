#!/usr/bin/env python
"""Inspect FLUX transformer module names and match StreamLoRA target presets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffusers import FluxTransformer2DModel

from streamlora.flux.target_modules import FluxBlockSpec, preset_target_modules, summarize_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local path containing transformer subfolder")
    parser.add_argument("--subfolder", default="transformer")
    parser.add_argument("--preset", default="all_common")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    transformer = FluxTransformer2DModel.from_pretrained(args.model, subfolder=args.subfolder)
    names = [name for name, _ in transformer.named_modules()]
    targets = preset_target_modules(args.preset, FluxBlockSpec())
    existing = [t for t in targets if t in names]
    missing = [t for t in targets if t not in names]
    report = {
        "model": args.model,
        "preset": args.preset,
        "summary": summarize_targets(targets),
        "matched": len(existing),
        "missing": len(missing),
        "missing_examples": missing[:30],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
