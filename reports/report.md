# StreamLoRA Experiment Report

## Motivation

Community FLUX LoRA recipes often specify target modules empirically, but there is limited systematic comparison of single-stream vs double-stream LoRA placement.  This project turns the choice of LoRA target modules into a controlled ablation study.

## Hypotheses

1. Single-stream blocks may be more effective for global cover style learning.
2. Double-stream blocks may better preserve text-image semantic binding.
3. Attention-only LoRA may be more stable, while MLP/projection LoRA may increase capacity and overfitting risk.
4. Late single-stream blocks may provide a better quality/cost trade-off for style adaptation.

## To fill after experiments

| Variant | Quality | Prompt adherence | Style sim | Diversity | Params | Notes |
|---|---|---|---|---|---|---|
| all_common | | | | | | |
| attention_only | | | | | | |
| single_stream | | | | | | |
| double_stream | | | | | | |
