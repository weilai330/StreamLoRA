# StreamLoRA

**StreamLoRA** studies where to insert LoRA adapters in **FLUX hybrid-stream Diffusion Transformer** fine-tuning.

Instead of training a single FLUX LoRA, this project compares LoRA target placement strategies across:

- double-stream / MMDiT blocks;
- single-stream blocks;
- attention modules;
- MLP / projection modules;
- image branch vs text/context branch;
- selected late single-stream blocks.

## Research question

> In FLUX-like hybrid-stream DiT models, how does LoRA target placement affect style learning, prompt adherence, overfitting risk, and parameter efficiency?

## Presets

| ID | Preset | Purpose |
|---|---|---|
| T0 | no LoRA | baseline |
| T1 | all_common | upper-bound capacity |
| T2 | attention_only | lightweight stable baseline |
| T3 | single_stream | global style learning hypothesis |
| T4 | double_stream | semantic binding hypothesis |
| T5 | double_image_branch | visual branch contribution |
| T6 | double_context_branch | text/context branch contribution |
| T7 | late_single_blocks | sparse late-block adaptation |

## Quick start

```bash
pip install -r requirements.txt

# Export explicit target module lists for all presets
bash scripts/export_target_configs.sh

# Optional: inspect whether targets match your local FLUX model
bash scripts/inspect_flux_targets.sh /path/to/FLUX.1-dev all_common
```

Generated target lists are written to `outputs/target_modules/*.json`.

## Planned evaluation

| Variant | PickScore | HPSv2 | Aesthetic | CLIP | Style Sim | Diversity | Params | Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no LoRA | | | | | | | | |
| attention_only | | | | | | | | |
| single_stream | | | | | | | | |
| double_stream | | | | | | | | |

## Deliverables

- FLUX module map
- LoRA target presets
- ablation table
- qualitative grids
- parameter-efficiency trade-off plot
- overfitting/failure-case analysis
