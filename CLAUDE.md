# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMU3R is a research project extending CUT3R (Continuous 3D Perception Model with Persistent State) with IMU-enhanced capabilities for improved relative pose estimation. The codebase builds on DUSt3R, MonST3R, and Spann3R.

**Key Innovation**: This project adds an IMU encoder and relative pose decoder to CUT3R's recurrent architecture, using IMU data (accelerometer + gyroscope) to improve pose estimation accuracy in video sequences.

## Core Architecture

### Model Components

1. **Base Model (CUT3R)**: ARCroco3DStereo - encoder-decoder architecture with recurrent state
   - Located in: `src/dust3r/model.py`
   - Encoder processes image patches into features
   - Decoder maintains persistent state across frames for continuous perception

2. **IMU Enhancement (CUT3RIMU)**: Wrapper that adds IMU capabilities
   - Located in: `src/dust3r/imu_encoder.py`
   - `IMUEncoder`: Processes 6-axis IMU sequences (accelerometer + gyroscope) into feature vectors
   - `RelativePoseDecoder`: Predicts relative pose between consecutive frames
   - `PoseTransformer`: Fuses visual features with IMU features using cross-attention

3. **Key Architectural Pattern**: State Precomputation
   - The model can precompute recurrent states for all scenes before training to reduce memory and speed up training
   - Enable with `precompute_states: True` in config
   - Dataset receives `precompute_model` and `precompute_device` to generate states

### Training Modes

Training is controlled via `training_mode` in config files:

```yaml
training_mode:
  enable_lora: false     # LoRA fine-tuning of base model
  enable_imu: true       # Train IMU components
  freeze_cut3r: true     # Freeze base CUT3R weights (IMU-only training)
```

**IMU-Only Training** (current focus):
- Base CUT3R model is frozen (`freeze_cut3r: true`)
- Only IMU encoder, pose transformer, and relative pose decoder are trained
- Uses separate learning rates: `imu_lr` for IMU components, `cut3r_lr` for base model
- Checkpoints saved as two files:
  - `checkpoint_*.pth`: Training state (optimizer, scaler, epoch)
  - `imu_weights_*.pth`: IMU model weights (encoder, decoder, transformer)

### Relative Pose Attention Mask
1. **State Gate Computation**: After processing each frame, compute which state tokens are associated with the current frame using the last decoder layer's cross-attention weights
2. **Soft Gating**: Use sigmoid-based soft gates (not hard masks) to create attention bias for the next frame
3. **Application**: Apply the mask during cross-attention in the decoder's image branch

## Common Commands

### Environment Setup

```bash
# Create environment
conda create -n cut3r python=3.11 cmake=3.14.0
conda activate cut3r

# Install dependencies
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
conda install 'llvm-openmp<16'
pip install git+https://github.com/nerfstudio-project/gsplat.git
pip install evo open3d

# Compile CUDA kernels for RoPE
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Training

**IMU-Enhanced Training** (current approach):
```bash
cd src/

# Multi-GPU training (4 GPUs) - PRODUCTION COMMAND
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 \
  accelerate launch --multi_gpu --num_processes=4 train_imu.py \
  --config-name train_imu

# Alternative: Full debugging with detailed logs
CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 \
  accelerate launch --multi_gpu train_imu.py

# Single GPU training
python train_imu.py
```

**Standard CUT3R Training** (original):
```bash
cd src/

# Multi-GPU training
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 \
  accelerate launch --multi_gpu train.py --config-name <config_name>

# Available configs: stage1, stage2, stage3, stage4, dpt_512_vary_4_64, linear_224_fixed_16
```

### Inference

**IMU-Enhanced Inference** (with trained IMU weights):
```bash
# Run IMU-enhanced demo with streaming mode - PRODUCTION COMMAND
CUDA_VISIBLE_DEVICES=0 python demo_imu.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --imu_path src/checkpoints/cut3r_relative_decoder/imu_weights_last.pth \
    --seq_path /project2/larg3r/dataset/dust3r_data/processed_kitti/kitti_00/rgb \
    --imu_data_path /project2/larg3r/dataset/dust3r_data/processed_kitti/kitti_00/imu \
    --gt_pose_path /project2/larg3r/dataset/dust3r_data/processed_kitti/kitti_00/cam \
    --device cuda --size 512 --streaming

# Without ground truth poses (inference only, no evaluation)
CUDA_VISIBLE_DEVICES=0 python demo_imu.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --imu_path src/checkpoints/cut3r_relative_decoder/imu_weights_last.pth \
    --seq_path /path/to/images \
    --imu_data_path /path/to/imu \
    --device cuda --size 512 --streaming
```

**Standard CUT3R Inference** (without IMU):
```bash
# Run demo inference
python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/001 --size 512 \
    --vis_threshold 1.5 --output_dir tmp

# With global alignment
python demo_ga.py --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/001 --size 512
```

### Evaluation

```bash
# Relative pose evaluation
cd eval/relpose
bash run.sh  # Evaluates on scannet, tum, sintel

# Other evaluations
cd eval/monodepth && bash run.sh
cd eval/mv_recon && bash run.sh
cd eval/video_depth && bash run.sh
```

## Dataset Configuration

Training datasets are specified in YAML configs using eval() syntax:
```yaml
dataset1: KITTI_Multi(
  ROOT='../../dataset/dust3r_data/processed_kitti',
  scenes=['kitti_00'],
  load_imu=true,  # Enable IMU data loading
  resolution=[(512, 384)],
  num_views=4
)

train_dataset: 200 @ ${dataset1}  # 200 samples from dataset1
```

**Supported Datasets**: 32 datasets total including ARKitScenes, BlendedMVS, CO3Dv2, KITTI, ScanNet, WayMo, TartanAir, etc. (see README.md)

**IMU Data Format**: When `load_imu=true`, datasets should provide IMU sequences with shape `(seq_len, 6)` where the 6 channels are `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`

## File Organization

```
src/
├── dust3r/              # Core model code
│   ├── model.py         # ARCroco3DStereo base model
│   ├── imu_encoder.py   # CUT3RIMU wrapper, IMU encoder, relative pose decoder
│   ├── blocks.py        # Attention blocks with mask support
│   ├── datasets/        # Dataset loaders (32+ datasets)
│   └── heads/           # Output heads (DPT, linear)
├── croco/               # CroCo foundation
├── train.py             # Standard training script
├── train_imu.py         # IMU-enhanced training script
└── demo.py              # Inference and visualization

config/                  # Hydra YAML configs
├── train_imu.yaml      # IMU training config
└── [stage configs]     # Training stage configs

eval/                   # Evaluation scripts
├── relpose/           # Relative pose evaluation
├── monodepth/         # Monocular depth evaluation
├── mv_recon/          # Multi-view reconstruction
└── video_depth/       # Video depth evaluation
```

## Important Implementation Details

### Parameter Management for IMU Training

The codebase uses a custom parameter grouping system (`get_imu_parameter_groups` in `train_imu.py`) that:
- Separates IMU and CUT3R parameters into different optimizer groups
- Only adds trainable parameters (`requires_grad=True`) to the optimizer
- Supports different learning rates for IMU vs CUT3R components
- Separates bias/norm parameters (no weight decay) from regular parameters

### Checkpoint Loading for IMU Training

IMU-only training uses a dual-checkpoint system:
1. **Training checkpoint** (`checkpoint_*.pth`): Contains optimizer state, loss scaler, epoch info
2. **IMU weights** (`imu_weights_*.pth`): Contains IMU encoder, pose decoder, pose transformer, relative pose token

When resuming, the system:
- Loads training checkpoint for optimizer/scaler state
- Loads IMU weights separately
- Does NOT load CUT3R weights (they remain frozen)

### Distributed Training

Uses HuggingFace Accelerate for multi-GPU training:
- `accelerate.prepare()` must be called BEFORE loading checkpoints
- Custom timeout: 6000 seconds (100 minutes)
- Mixed precision: bf16
- `find_unused_parameters=True` for DDP

### Slack Notifications

Training supports Slack notifications (configured in `slack` section of YAML):
```yaml
slack:
  enabled: true
  webhook_url: <webhook_url>
  send_on_completion: true
  send_on_error: true
```

## Development Patterns

### Adding New Components to IMU Pipeline

1. Define the component in `src/dust3r/imu_encoder.py` (e.g., new encoder architecture)
2. Add it to `CUT3RIMU.__init__()` initialization
3. Include it in `get_imu_parameter_groups()` parameter grouping (use appropriate keyword in parameter name)
4. Add to checkpoint saving/loading in `save_model()` and resume logic
5. Update `imu_config` in YAML if new hyperparameters are needed

### Debugging Training

Common issues and solutions:
- **"Parameter count mismatch"**: Check that frozen parameters aren't being added to optimizer
- **"Loss is NaN"**: Check for empty tensors in visualization, verify input data normalization
- **"Missing keys in state dict"**: Ensure checkpoint format matches (IMU training uses different format than standard training)
- **State precomputation fails**: Verify `precompute_states=True` and check dataset string injection in `train_imu.py:440-500`

### Modifying Attention Mechanisms

When changing attention in blocks:
1. Update `CrossAttention.forward()` signature in `src/dust3r/blocks.py`
2. Update `DecoderBlock.forward()` to pass masks through
3. Update `_decoder()` in `imu_encoder.py` to generate and apply masks
4. Test with `return_attn=True` to verify attention patterns

## Testing and Validation

The codebase includes defensive checks for empty tensors in visualization (`get_vis_imgs_new`, `vis_and_cat`) to prevent crashes during training/evaluation.

When adding new features:
- Use `safe_quantile()` helper for operations on potentially empty tensors
- Add bounds checking when indexing into tensors
- Verify behavior with both metric and non-metric datasets
