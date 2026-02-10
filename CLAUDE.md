# CLAUDE.md

## Project Overview

**Prompt3r**: Visual Prompt Tuning for relative pose prediction on frozen CUT3R.

**Core Idea**: Freeze CUT3R, add learnable prompt tokens (~1% params) to query relative pose from existing representations.

## Environment

```bash
. /project2/linjohnss/miniconda3/bin/activate cut3r_gpu7
```

## Key Files

| File | Purpose |
|------|---------|
| `src/dust3r/model.py` | Main model, prompt tokens, freeze logic |
| `src/dust3r/losses.py` | Loss functions (relative pose loss) |
| `src/dust3r/utils/camera.py` | `RelativePoseDecoder` |
| `src/dust3r/heads/dpt_head.py` | Head with relative pose output |

## Training

### Standard Training
```bash
cd src/
accelerate launch --multi_gpu train.py --config-name train_prompt3r
```

### Overfit Experiment

For overfitting experiments on a single sequence:

- **Target**: `../dataset/mast3r_data/processed_tartanair2/office/Easy/P000/`
- **Config**: `config/train_prompt3r_fit.yaml`

```bash
cd src/
python train.py --config-name train_prompt3r_fit
```

Key settings in `train_prompt3r_fit.yaml`:
- `train_dataset: 100 @ ${dataset1}` (small dataset size for overfitting)
- `batch_size: 8`, `epochs: 40`
- `exp_name: 'prompt3r_fit70'` (increment for new experiments)

## Inference

```bash
python demo.py \
  --model_path src/checkpoints/prompt3r_fit70/checkpoint-last.pth \
  --seq_path ../dataset/mast3r_data/processed_tartanair2/office/Easy/P000/ \
  --device cuda --downsample_factor 100 --vis_threshold 10.0
```

### Analysis

```bash
python analysis/analyze_prompt_attention.py \
  --model_path src/checkpoints/prompt3r_fit142/checkpoint-last.pth \
  --seq_path ../dataset/mast3r_data/processed_tartanair2/office/Easy/P000/ \
  --device cuda --downsample_factor 100 --vis_threshold 10.0
```

All analysis scripts and results must be run from the `analysis/` directory.

## Reference

All reference methods are in `reference/`.

## Key Concepts

### Trainable Components (freeze='encoder_and_decoder_and_head')
- `relative_pose_token`: Learnable prompt tokens (16 tokens)
- `RelativePoseDecoder`: MLP head for SE(3) output

### Relative Pose Convention
- 4x4 SE(3) matrix, camera-to-world (c2w)
- Accumulation: `T_c2w_curr = T_c2w_prev @ T_rel_inv`

### Loss Switches
- `use_pts_loss`: Point cloud loss
- `use_pose_loss`: Absolute pose loss
- `use_relative_pose_loss`: Relative pose loss

## Summary Instructions

When using compact mode, focus on test output and code changes.
