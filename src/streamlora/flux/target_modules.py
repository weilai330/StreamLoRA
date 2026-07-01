"""LoRA target-module presets for FLUX hybrid-stream DiT experiments.

The names follow Diffusers' FluxTransformer2DModel convention:
- transformer_blocks: double-stream / MMDiT blocks
- single_transformer_blocks: single-stream blocks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DOUBLE_ATTN = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
]
DOUBLE_MLP_IMAGE = ["ff.net.0.proj", "ff.net.2"]
DOUBLE_MLP_CONTEXT = ["ff_context.net.0.proj", "ff_context.net.2"]
SINGLE_ATTN = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
SINGLE_MLP = ["proj_mlp", "proj_out"]


@dataclass(frozen=True)
class FluxBlockSpec:
    num_double_blocks: int = 19
    num_single_blocks: int = 38


def expand(prefix: str, block_ids: Iterable[int], modules: Iterable[str]) -> list[str]:
    return [f"{prefix}.{i}.{m}" for i in block_ids for m in modules]


def preset_target_modules(name: str, spec: FluxBlockSpec = FluxBlockSpec()) -> list[str]:
    """Return explicit LoRA target module names for an experiment preset."""
    double_ids = range(spec.num_double_blocks)
    single_ids = range(spec.num_single_blocks)
    name = name.lower()
    if name == "attention_only":
        return expand("transformer_blocks", double_ids, DOUBLE_ATTN) + expand(
            "single_transformer_blocks", single_ids, SINGLE_ATTN
        )
    if name == "single_stream":
        return expand("single_transformer_blocks", single_ids, SINGLE_ATTN + SINGLE_MLP)
    if name == "double_stream":
        return expand("transformer_blocks", double_ids, DOUBLE_ATTN + DOUBLE_MLP_IMAGE + DOUBLE_MLP_CONTEXT)
    if name == "double_image_branch":
        return expand("transformer_blocks", double_ids, ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] + DOUBLE_MLP_IMAGE)
    if name == "double_context_branch":
        return expand("transformer_blocks", double_ids, ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out"] + DOUBLE_MLP_CONTEXT)
    if name == "late_single_blocks":
        late = [i for i in [24, 28, 32, 36] if i < spec.num_single_blocks]
        return expand("single_transformer_blocks", late, SINGLE_ATTN + SINGLE_MLP)
    if name == "all_common":
        return preset_target_modules("single_stream", spec) + preset_target_modules("double_stream", spec)
    raise ValueError(f"Unknown preset: {name}")


def summarize_targets(targets: list[str]) -> dict[str, int]:
    return {
        "num_targets": len(targets),
        "double_targets": sum(t.startswith("transformer_blocks") for t in targets),
        "single_targets": sum(t.startswith("single_transformer_blocks") for t in targets),
        "attention_targets": sum("attn." in t for t in targets),
        "mlp_or_proj_targets": sum(("ff" in t) or ("proj_mlp" in t) or ("proj_out" in t) for t in targets),
    }
