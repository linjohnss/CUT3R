# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
)  # noqa: F401, needed when loading the model
from dust3r.imu_encoder import CUT3RIMU  # IMU-enhanced model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch, loss_of_one_batch_tbptt  # noqa
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

# Import Slack notification utility
from slack_notification import create_slack_notifier


def get_imu_parameter_groups(model, weight_decay, args):
    """
    分離 IMU encoder 和 CUT3R 的參數組，為它們設置不同的學習率
    只將 requires_grad=True 的參數加入優化器，凍結的參數完全排除
    
    Args:
        model: CUT3RIMU model
        weight_decay: weight decay value
        args: training arguments
    
    Returns:
        list of parameter groups with different learning rates
    """
    # 獲取 IMU 相關的學習率設置
    imu_lr = getattr(args, 'imu_lr', args.lr)  # 默認使用主學習率
    cut3r_lr = getattr(args, 'cut3r_lr', args.lr)  # 默認使用主學習率
    
    parameter_groups = []
    
    # 分離 IMU encoder 參數和 CUT3R 參數
    imu_params = []
    cut3r_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # All frozen parameters (should be CUT3R parameters)
            frozen_params.append((name, param.numel()))
            continue  # 跳過凍結的參數，不加入優化器
            
        # 判斷是否為 IMU 相關參數
        # CRITICAL FIX: Only include parameters that are actually trainable (requires_grad=True)
        # and belong to IMU components
        # IMPORTANT: Include relative_pose_token which is a learnable parameter for relative pose estimation
        # IMPORTANT: Include reset_gate and update_gate for state update control
        if any(imu_keyword in name for imu_keyword in [
            'imu_encoder', 'pose_transformer', 'relative_pose_decoder', 
            'relative_pose_token', 'reset_gate', 'update_gate'
        ]):
            imu_params.append((name, param))
        else:
            # This should not happen if CUT3R parameters are properly frozen
            print(f"WARNING: Found trainable parameter that is not IMU-related: {name}")
            cut3r_params.append((name, param))
    
    # 統計信息
    total_frozen_params = sum(count for _, count in frozen_params)
    total_imu_params = sum(p.numel() for _, p in imu_params)
    total_cut3r_params = sum(p.numel() for _, p in cut3r_params)
    
    print(f"Parameter Statistics:")
    print(f"  Frozen parameters: {total_frozen_params:,} (not included in optimizer)")
    print(f"  IMU parameters: {total_imu_params:,}")
    print(f"  CUT3R parameters: {total_cut3r_params:,}")
    print(f"  Total trainable: {total_imu_params + total_cut3r_params:,}")
    
    # 只為 IMU 參數創建參數組（CUT3R 參數如果被凍結則不加入優化器）
    if imu_params:
        # 分離 bias 和 norm 層（設置 weight_decay=0）
        imu_decay_params = []
        imu_no_decay_params = []
        
        for name, param in imu_params:
            if len(param.shape) == 1 or param.dim() == 1:  # bias 和 norm 層
                imu_no_decay_params.append(param)
            else:
                imu_decay_params.append(param)
        
        if imu_decay_params:
            parameter_groups.append({
                'params': imu_decay_params,
                'lr': imu_lr,
                'weight_decay': weight_decay,
                'name': 'imu_decay'
            })
        
        if imu_no_decay_params:
            parameter_groups.append({
                'params': imu_no_decay_params,
                'lr': imu_lr,
                'weight_decay': 0.0,
                'name': 'imu_no_decay'
            })
    
    # 只為未凍結的 CUT3R 參數創建參數組
    if cut3r_params:
        # 分離 bias 和 norm 層（設置 weight_decay=0）
        cut3r_decay_params = []
        cut3r_no_decay_params = []
        
        for name, param in cut3r_params:
            if len(param.shape) == 1 or param.dim() == 1:  # bias 和 norm 層
                cut3r_no_decay_params.append(param)
            else:
                cut3r_decay_params.append(param)
        
        if cut3r_decay_params:
            parameter_groups.append({
                'params': cut3r_decay_params,
                'lr': cut3r_lr,
                'weight_decay': weight_decay,
                'name': 'cut3r_decay'
            })
        
        if cut3r_no_decay_params:
            parameter_groups.append({
                'params': cut3r_no_decay_params,
                'lr': cut3r_lr,
                'weight_decay': 0.0,
                'name': 'cut3r_no_decay'
            })
    
    # 打印參數組信息
    print("\nOptimizer parameter groups:")
    for i, group in enumerate(parameter_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {group['name']} - LR: {group['lr']:.2e}, WD: {group['weight_decay']}, Params: {param_count:,}")
    
    # 驗證沒有凍結的參數被意外加入優化器
    optimizer_param_count = sum(sum(p.numel() for p in group['params']) for group in parameter_groups)
    expected_trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if optimizer_param_count != expected_trainable_count:
        print(f"⚠️  Warning: Parameter count mismatch!")
        print(f"   Optimizer params: {optimizer_param_count:,}")
        print(f"   Expected trainable: {expected_trainable_count:,}")
    else:
        print(f"✅ All trainable parameters ({optimizer_param_count:,}) correctly included in optimizer")
    
    return parameter_groups


def adjust_imu_learning_rate(optimizer, epoch, args):
    """
    為分離的參數組調整學習率
    支持 IMU encoder 和 CUT3R 的不同學習率調度
    只調整優化器中實際存在的參數組
    """
    # 獲取分離的學習率設置
    imu_lr = getattr(args, 'imu_lr', args.lr)
    cut3r_lr = getattr(args, 'cut3r_lr', args.lr)
    imu_min_lr = getattr(args, 'imu_min_lr', args.min_lr)
    cut3r_min_lr = getattr(args, 'cut3r_min_lr', args.min_lr)
    
    # 計算 warmup 和 decay 後的學習率
    if epoch < args.warmup_epochs:
        # Warmup 階段
        imu_current_lr = imu_lr * epoch / args.warmup_epochs
        cut3r_current_lr = cut3r_lr * epoch / args.warmup_epochs
    else:
        # Cosine decay 階段
        imu_current_lr = imu_min_lr + (imu_lr - imu_min_lr) * 0.5 * (
            1.0 + math.cos(
                math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            )
        )
        cut3r_current_lr = cut3r_min_lr + (cut3r_lr - cut3r_min_lr) * 0.5 * (
            1.0 + math.cos(
                math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            )
        )
    
    # 統計實際調整的參數組
    imu_groups_updated = 0
    cut3r_groups_updated = 0
    
    # 為每個參數組設置對應的學習率
    for param_group in optimizer.param_groups:
        group_name = param_group.get('name', '')
        if 'imu' in group_name:
            param_group['lr'] = imu_current_lr
            imu_groups_updated += 1
        elif 'cut3r' in group_name:
            param_group['lr'] = cut3r_current_lr
            cut3r_groups_updated += 1
        else:
            # 如果沒有指定名稱，使用默認學習率（通常是IMU學習率）
            param_group['lr'] = imu_current_lr
            print(f"⚠️  Warning: Parameter group without name found, using IMU learning rate: {imu_current_lr:.2e}")
    return imu_current_lr, cut3r_current_lr


from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")

# Import LoRA utilities
from lora_utils import (
    get_lora_state_dict,
    load_lora_state_dict,
    freeze_non_lora_parameters,
)


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def create_imu_enhanced_model(base_model, imu_config):
    """Create IMU-enhanced CUT3R model if IMU config is provided"""
    if imu_config:
        # Convert OmegaConf to dict if needed
        if hasattr(imu_config, '_content'):
            imu_config = OmegaConf.to_container(imu_config, resolve=True)
        return CUT3RIMU(base_model, imu_config)
    else:
        # Return original model if no IMU config
        return base_model


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device

    setup_for_distributed(accelerator)

    # Initialize Slack notifier
    slack_config = getattr(args, 'slack', {})
    slack_enabled = slack_config.get('enabled', False)
    
    if slack_enabled:
        slack_notifier = create_slack_notifier(args)
        if slack_notifier.enabled:
            printer.info("Slack notifications enabled")
        else:
            printer.info("Slack notifications disabled (no webhook URL provided)")
            slack_notifier = None
    else:
        printer.info("Slack notifications disabled (disabled in config)")
        slack_notifier = None

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # # auto resume
    # if not args.resume:
    #     last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
    #     args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # model - Load model BEFORE dataset to enable state precomputation
    printer.info("Loading model: %s", args.model)
    base_model: PreTrainedModel = eval(args.model)
    
    # Create IMU-enhanced model if IMU config is provided
    imu_config = getattr(args, 'imu_config', None)
    model = create_imu_enhanced_model(base_model, imu_config)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Print gating status if model has use_gating attribute
    if hasattr(model, 'use_gating'):
        printer.info(f"Gating mechanism: {'Enabled' if model.use_gating else 'Disabled'}")
    
    # Load pretrained weights if available (before state precomputation)
    if args.pretrained and not args.resume:
        from dust3r.model import strip_module
        
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        load_only_encoder = getattr(args, "load_only_encoder", False)
        
        # Get base model (unwrap if it's CUT3RIMU)
        base_m = model.cut3r_model if hasattr(model, 'cut3r_model') else model
        
        if load_only_encoder:
            printer.info("Loading only encoder weights...")
            # Filter to only encoder keys
            if "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            
            # Strip module prefix if present
            state_dict = strip_module(state_dict)
            
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if "enc_blocks" in k or "patch_embed" in k
            }
            msg = base_m.load_state_dict(filtered_state_dict, strict=False)
            printer.info(f"Loaded encoder weights: {msg}")
        else:
            # Load full model
            if "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            msg = base_m.load_state_dict(strip_module(state_dict), strict=False)
            printer.info(f"Loaded pretrained weights: {msg}")
    
    # Move model to device for state precomputation
    model.to(device)
    
    # training dataset and loader (with state precomputation enabled)
    printer.info("Building train dataset %s", args.train_dataset)
    precompute_states = getattr(args, 'precompute_states', False)
    printer.info(f"precompute_states = {precompute_states}")
    printer.info(f"args.train_dataset = {args.train_dataset}")
    printer.info(f"args.train_dataset type = {type(args.train_dataset)}")
    printer.info(f"args.train_dataset repr = {repr(args.train_dataset)}")
    
    # Debug: Check if it's a list or string
    if isinstance(args.train_dataset, list):
        printer.info(f"args.train_dataset is a list with {len(args.train_dataset)} items")
        for i, item in enumerate(args.train_dataset):
            printer.info(f"  Item {i}: {item} (type: {type(item)})")
    elif isinstance(args.train_dataset, str):
        printer.info(f"args.train_dataset is a string with length {len(args.train_dataset)}")
    else:
        printer.info(f"args.train_dataset is {type(args.train_dataset)}: {args.train_dataset}")
    
    # Prepare extra context for dataset creation
    extra_context = {}
    train_dataset_str = args.train_dataset
    
    if precompute_states:
        printer.info("State precomputation enabled - injecting model into dataset")
        printer.info(f"Original train_dataset_str: {train_dataset_str}")
        printer.info(f"train_dataset_str type: {type(train_dataset_str)}")
        printer.info(f"train_dataset_str repr: {repr(train_dataset_str)}")
        # Add FULL MODEL (CUT3RIMU) to extra context for eval()
        # CRITICAL: Pass the full CUT3RIMU model instead of base model to ensure relative_pose_token is included
        extra_context['model'] = model
        
        # Inject precompute parameters into dataset string
        # Handle the case where train_dataset is "200 @ KITTI_Multi(...)"
        if 'precompute_model=' not in train_dataset_str:
            # Check if it's a repeated dataset format (look for " @ KITTI_Multi")
            if ' @ KITTI_Multi(' in train_dataset_str:
                printer.info("Found repeated dataset format")
                # Extract the base dataset string (split only on first @)
                parts = train_dataset_str.split(' @ ', 1)
                printer.info(f"Split parts: {parts}")
                if len(parts) == 2:
                    count, base_dataset = parts
                    printer.info(f"Count: {count}, Base dataset: {base_dataset}")
                    # Inject precompute parameters into the base dataset
                    insert_pos = base_dataset.rfind(')')
                    printer.info(f"Insert position: {insert_pos}")
                    if insert_pos > 0:
                        modified_base = (
                            base_dataset[:insert_pos] + 
                            f", precompute_model=model, precompute_device='{device}'" + 
                            base_dataset[insert_pos:]
                        )
                        train_dataset_str = f"{count} @ {modified_base}"
                        printer.info(f"✅ Modified dataset string for state precomputation")
                        printer.info(f"  Original: {args.train_dataset[:100]}...")
                        printer.info(f"  Modified: {train_dataset_str[:100]}...")
                    else:
                        printer.warning(f"⚠️ Could not find ')' in base dataset: {base_dataset}")
                else:
                    printer.warning(f"⚠️ Could not split dataset string into 2 parts: {parts}")
            else:
                printer.info("Not a repeated dataset format, checking for direct KITTI_Multi")
                # Direct dataset string (look for KITTI_Multi)
                if 'KITTI_Multi(' in train_dataset_str:
                    printer.info("Found direct KITTI_Multi format")
                    insert_pos = train_dataset_str.rfind(')')
                    printer.info(f"Insert position: {insert_pos}")
                    if insert_pos > 0:
                        train_dataset_str = (
                            train_dataset_str[:insert_pos] + 
                            f", precompute_model=model, precompute_device='{device}'" + 
                            train_dataset_str[insert_pos:]
                        )
                        printer.info(f"✅ Modified dataset string for state precomputation")
                        printer.info(f"  Original: {args.train_dataset[:100]}...")
                        printer.info(f"  Modified: {train_dataset_str[:100]}...")
                    else:
                        printer.warning(f"⚠️ Could not find ')' in dataset string: {train_dataset_str}")
                else:
                    printer.warning(f"⚠️ Could not find KITTI_Multi in dataset string: {train_dataset_str[:100]}...")
        else:
            printer.info("precompute_model already present in dataset string")
    
    # Final check
    printer.info(f"Final train_dataset_str: {train_dataset_str}")
    printer.info(f"Contains precompute_model: {'precompute_model=' in train_dataset_str}")
    
    data_loader_train = build_dataset(
        train_dataset_str,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length,
        extra_context=extra_context
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True
        )
        for dataset in args.test_dataset.split("+")
    }
    
    # Handle both wrapped (CUT3RIMU) and unwrapped models
    base_model = model.cut3r_model if hasattr(model, 'cut3r_model') else model
    printer.info(
        f"Encoder parameters: {sum(p.numel() for p in base_model.enc_blocks.parameters())}"
    )
    printer.info(
        f"Decoder parameters: {sum(p.numel() for p in base_model.dec_blocks.parameters())}"
    )

    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    # Model already moved to device earlier for state precomputation
    # model.to(device)

    if args.gradient_checkpointing:
        # Handle both wrapped (CUT3RIMU) and unwrapped models
        base_model = model.cut3r_model if hasattr(model, 'cut3r_model') else model
        base_model.gradient_checkpointing_enable()
    if args.long_context:
        # Handle both wrapped (CUT3RIMU) and unwrapped models
        base_model = model.cut3r_model if hasattr(model, 'cut3r_model') else model
        base_model.fixed_input_length = False

    # Pretrained weights already loaded before state precomputation
    # if args.pretrained and not args.resume:
    #     ...already done above...

    # # following timm: set wd as 0 for bias and norm layers
    # 分離 IMU encoder 和 CUT3R 的參數組，設置不同的學習率
    param_groups = get_imu_parameter_groups(model, args.weight_decay, args)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler(accelerator=accelerator)

    accelerator.even_batches = False
    
    # CRITICAL: Store reference to unwrapped dataset for state recomputation
    # After accelerator.prepare(), the dataset will be wrapped and hard to access
    train_dataset_unwrapped = None
    if hasattr(data_loader_train, 'dataset'):
        train_dataset_unwrapped = data_loader_train.dataset
    
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        actual_model = accelerator.unwrap_model(model)
        base_model = actual_model.cut3r_model if hasattr(actual_model, 'cut3r_model') else actual_model
        enable_imu = getattr(args.training_mode, 'enable_imu', False)
        # Save IMU weights if IMU training is enabled
        if enable_imu and hasattr(actual_model, 'relative_pose_decoder') and accelerator.is_main_process:
            imu_path = os.path.join(args.output_dir, f"imu_weights_{fname}.pth")
            imu_state_dict = {}

            # Save IMU encoder
            if hasattr(actual_model, 'imu_encoder'):
                imu_state_dict['imu_encoder'] = actual_model.imu_encoder.state_dict()
            
            # Save relative pose decoder (main IMU component)
            if hasattr(actual_model, 'relative_pose_decoder'):
                imu_state_dict['relative_pose_decoder'] = actual_model.relative_pose_decoder.state_dict()
            
            # Save new IMU-aware pose retriever if it exists
            if hasattr(actual_model, 'pose_transformer'):
                imu_state_dict['pose_transformer'] = actual_model.pose_transformer.state_dict()
            
            # Save relative pose token if it exists (it's a Parameter, not a module)
            if hasattr(actual_model, 'relative_pose_token'):
                imu_state_dict['relative_pose_token'] = actual_model.relative_pose_token

            # Save reset gate weights (only if gating is enabled)
            if hasattr(actual_model, 'reset_gate') and actual_model.reset_gate is not None:
                imu_state_dict['reset_gate'] = actual_model.reset_gate.state_dict()
            
            # Save update gate weights (only if gating is enabled)
            if hasattr(actual_model, 'update_gate') and actual_model.update_gate is not None:
                imu_state_dict['update_gate'] = actual_model.update_gate.state_dict()

            torch.save(imu_state_dict, imu_path)
            # Count parameters correctly by flattening nested state dicts
            imu_params = 0
            for key, module_dict in imu_state_dict.items():
                if isinstance(module_dict, dict):
                    imu_params += sum(w.numel() for w in module_dict.values())
                else:
                    # For scalar parameters like imu_weight, pose_weight
                    imu_params += module_dict.numel()
            printer.info(f"Saved IMU weights to {imu_path} ({imu_params:,} parameters)")


    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")
    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if accelerator.is_main_process else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    accelerator,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

                # Save best of all
                if stats["loss_med"] < best_so_far:
                    best_so_far = stats["loss_med"]
                    new_best = True
        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            slack_notifier=slack_notifier,
            slack_config=slack_config,
            train_dataset_unwrapped=train_dataset_unwrapped,
        )
        
        # Periodic state recomputation to align with updated model (epoch-based)
        precompute_recompute_interval = getattr(args, 'precompute_recompute_interval', None)
        if (precompute_recompute_interval is not None and 
            precompute_recompute_interval > 0 and
            epoch > 0 and
            epoch % precompute_recompute_interval == 0 and
            accelerator.is_main_process):
            
            printer.info(f"\n{'='*80}")
            printer.info(f"🔄 Triggering state recomputation after epoch {epoch}")
            printer.info(f"{'='*80}\n")
            
            try:
                # Get the unwrapped model (actual CUT3RIMU instance)
                actual_model = accelerator.unwrap_model(model)
                
                # Unwrap dataset layers to find the actual dataset with recompute_states
                def find_recomputable_dataset(ds):
                    """Recursively unwrap dataset to find one with recompute_states method"""
                    if ds is None:
                        return None
                    if hasattr(ds, 'recompute_states'):
                        return ds
                    # Check if wrapped by ResizedDataset, MulDataset, etc.
                    if hasattr(ds, 'dataset'):
                        return find_recomputable_dataset(ds.dataset)
                    # Check if CatDataset (has multiple datasets)
                    if hasattr(ds, 'datasets'):
                        for inner_ds in ds.datasets:
                            result = find_recomputable_dataset(inner_ds)
                            if result is not None:
                                return result
                    return None
                
                recomputable_dataset = find_recomputable_dataset(train_dataset_unwrapped)
                
                if recomputable_dataset is not None:
                    printer.info(f"Found recomputable dataset: {type(recomputable_dataset).__name__}")
                    recomputable_dataset.recompute_states(actual_model)
                    printer.info("✅ State recomputation completed successfully")
                else:
                    printer.warning(f"⚠️ Could not find dataset with recompute_states method")
            except Exception as e:
                printer.error(f"❌ Error during state recomputation: {e}")
                import traceback
                traceback.print_exc()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)
    
    # Send Slack notification for training completion
    if (accelerator.is_main_process and slack_notifier and 
        slack_config.get('send_on_completion', True)):
        try:
            slack_notifier.send_training_complete(
                args=args,
                total_time=total_time_str,
                best_loss=best_so_far if best_so_far != float("inf") else None,
                output_dir=args.output_dir
            )
            printer.info("Slack notification sent successfully")
        except Exception as e:
            printer.error(f"Failed to send Slack notification: {e}")


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    actual_model = accelerator.unwrap_model(model_without_ddp)
    base_model = actual_model.cut3r_model if hasattr(actual_model, 'cut3r_model') else actual_model
    
    # Check training mode configuration
    enable_lora = getattr(args.training_mode, 'enable_lora', False)
    enable_imu = getattr(args.training_mode, 'enable_imu', False)
    
    # Check if this is LoRA training
    is_lora_training = enable_lora and getattr(getattr(base_model, 'config', None), 'enable_lora', False)
    lora_param_count = sum(1 for name, _ in base_model.named_parameters() if 'lora_A' in name or 'lora_B' in name)
    lora_state_dict = get_lora_state_dict(base_model) if enable_lora else None
    
    # Check if this is IMU training
    is_imu_training = enable_imu and hasattr(actual_model, 'imu_encoder')
    
    printer.info(f"=== SAVE FINAL MODEL ===")
    printer.info(f"LoRA training: {is_lora_training}, LoRA params: {lora_param_count}, LoRA weights: {len(lora_state_dict) if lora_state_dict else 0}")
    printer.info(f"IMU training: {is_imu_training}")
    
    # checkpoint_path = output_dir / "checkpoint-final.pth"
    
    if is_lora_training or lora_param_count > 0 or is_imu_training:
        # # Save enhanced checkpoint (LoRA and/or IMU)
        # to_save = {
        #     "args": args,
        #     "epoch": epoch,
        #     "model_config": getattr(base_model, 'config', None),
        # }
        # if best_so_far is not None:
        #     to_save["best_so_far"] = best_so_far
        
        # # Add IMU config if available
        # if is_imu_training:
        #     to_save["imu_config"] = getattr(args, 'imu_config', {})
        
        # printer.info(f">> Saving enhanced checkpoint to {checkpoint_path} ...")
        # misc.save_on_master(accelerator, to_save, checkpoint_path)
        
        # Save standalone IMU weights if IMU training is enabled
        if enable_imu and is_imu_training and accelerator.is_main_process:
            imu_path = output_dir / "imu_weights_final.pth"
            imu_state_dict = {}
            
            # Save relative pose decoder (main IMU component)
            if hasattr(actual_model, 'relative_pose_decoder'):
                imu_state_dict['relative_pose_decoder'] = actual_model.relative_pose_decoder.state_dict()
            
            # Save IMU encoder
            if hasattr(actual_model, 'imu_encoder'):
                imu_state_dict['imu_encoder'] = actual_model.imu_encoder.state_dict()

            # Save new IMU-aware pose retriever if it exists
            if hasattr(actual_model, 'pose_transformer'):
                imu_state_dict['pose_transformer'] = actual_model.pose_transformer.state_dict()

            # Save relative pose token if it exists (it's a Parameter, not a module)
            if hasattr(actual_model, 'relative_pose_token'):
                imu_state_dict['relative_pose_token'] = actual_model.relative_pose_token
            
            # Save reset gate weights (only if gating is enabled)
            if hasattr(actual_model, 'reset_gate') and actual_model.reset_gate is not None:
                imu_state_dict['reset_gate'] = actual_model.reset_gate.state_dict()
            
            # Save update gate weights (only if gating is enabled)
            if hasattr(actual_model, 'update_gate') and actual_model.update_gate is not None:
                imu_state_dict['update_gate'] = actual_model.update_gate.state_dict()
            
            torch.save(imu_state_dict, imu_path)
            # Count parameters correctly by flattening nested state dicts
            imu_params = 0
            for key, module_dict in imu_state_dict.items():
                if isinstance(module_dict, dict):
                    imu_params += sum(w.numel() for w in module_dict.values())
                else:
                    # For scalar parameters like imu_weight, pose_weight
                    imu_params += module_dict.numel()
            printer.info(f"✅ Saved IMU weights: {imu_params:,} params")
    # else:
    #     # Save standard model
    #     to_save = {
    #         "args": args,
    #         "model": actual_model.cpu().state_dict() if not isinstance(actual_model, dict) else actual_model,
    #         "epoch": epoch,
    #     }
    #     if best_so_far is not None:
    #         to_save["best_so_far"] = best_so_far
    #     printer.info(f">> Saving model to {checkpoint_path} ...")
    #     misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False, extra_context=None):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length,
        extra_context=extra_context
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    slack_notifier=None,
    slack_config=None,
    train_dataset_unwrapped=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)

    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, 'cut3r_model'):
        unwrapped_model.cut3r_model.eval()  # 强制保持 eval 模式
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("imu_lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("cut3r_lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far):        
        # Save LoRA and IMU weights separately based on training mode
        actual_model = accelerator.unwrap_model(model)
        base_model = actual_model.cut3r_model if hasattr(actual_model, 'cut3r_model') else actual_model
        enable_lora = getattr(args.training_mode, 'enable_lora', False)
        enable_imu = getattr(args.training_mode, 'enable_imu', False)
        
        
        # Save IMU weights if IMU training is enabled
        if enable_imu and hasattr(actual_model, 'relative_pose_decoder') and accelerator.is_main_process:
            imu_path = os.path.join(args.output_dir, f"imu_weights_{fname}.pth")
            imu_state_dict = {}
            
            # Save relative pose decoder (main IMU component)
            if hasattr(actual_model, 'relative_pose_decoder'):
                imu_state_dict['relative_pose_decoder'] = actual_model.relative_pose_decoder.state_dict()
            
            # Save IMU encoder
            if hasattr(actual_model, 'imu_encoder'):
                imu_state_dict['imu_encoder'] = actual_model.imu_encoder.state_dict()

            # Save new IMU-aware pose retriever if it exists
            if hasattr(actual_model, 'pose_transformer'):
                imu_state_dict['pose_transformer'] = actual_model.pose_transformer.state_dict()

            # Save relative pose token if it exists (it's a Parameter, not a module)
            if hasattr(actual_model, 'relative_pose_token'):
                imu_state_dict['relative_pose_token'] = actual_model.relative_pose_token
            
            # Save reset gate weights (only if gating is enabled)
            if hasattr(actual_model, 'reset_gate') and actual_model.reset_gate is not None:
                imu_state_dict['reset_gate'] = actual_model.reset_gate.state_dict()
            
            # Save update gate weights (only if gating is enabled)
            if hasattr(actual_model, 'update_gate') and actual_model.update_gate is not None:
                imu_state_dict['update_gate'] = actual_model.update_gate.state_dict()
            
            torch.save(imu_state_dict, imu_path)
            # Count parameters correctly by flattening nested state dicts
            imu_params = 0
            for key, module_dict in imu_state_dict.items():
                if isinstance(module_dict, dict):
                    imu_params += sum(w.numel() for w in module_dict.values())
                else:
                    # For scalar parameters like imu_weight, pose_weight
                    imu_params += module_dict.numel()
            printer.info(f"Saved IMU weights to {imu_path} ({imu_params:,} parameters)")

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        with accelerator.accumulate(model):
            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                # 使用分離的學習率調度器
                imu_lr, cut3r_lr = adjust_imu_learning_rate(optimizer, epoch_f, args)
            else:
                # 如果沒有更新學習率，使用當前優化器中的學習率
                imu_lr = optimizer.param_groups[0]["lr"]  # 默認值
                cut3r_lr = optimizer.param_groups[0]["lr"]  # 默認值
            if not args.long_context:
                result = loss_of_one_batch(
                    batch,
                    model,
                    criterion,
                    accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            else:
                result = loss_of_one_batch_tbptt(
                    batch,
                    model,
                    criterion,
                    chunk_size=8,
                    loss_scaler=loss_scaler,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                error_msg = f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                print(error_msg)
                
                # Send Slack notification for training error
                if (slack_notifier and slack_notifier.enabled and accelerator.is_main_process and
                    slack_config.get('send_on_error', True)):
                    try:
                        slack_notifier.send_training_error(error_msg, args)
                    except Exception as e:
                        print(f"Failed to send Slack error notification: {e}")
                
                sys.exit(1)
            if not result.get("already_backprop", False):
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            # 獲取分離的學習率
            lr = optimizer.param_groups[0]["lr"]  # 保持向後兼容
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(imu_lr=imu_lr)
            metric_logger.update(cut3r_lr=cut3r_lr)
            metric_logger.update(step=step)

            metric_logger.update(loss=loss_value, **loss_details)

            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is None:
                    continue
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

            if tb_vis_img:
                if log_writer is None:
                    continue
                with torch.no_grad():
                    depths_self, gt_depths_self = get_render_results(
                        batch, result["pred"], self_view=True
                    )
                    depths_cross, gt_depths_cross = get_render_results(
                        batch, result["pred"], self_view=False
                    )
                    for k in range(len(batch)):
                        loss_details[f"self_pred_depth_{k+1}"] = (
                            depths_self[k].detach().cpu()
                        )
                        loss_details[f"self_gt_depth_{k+1}"] = (
                            gt_depths_self[k].detach().cpu()
                        )
                        loss_details[f"pred_depth_{k+1}"] = (
                            depths_cross[k].detach().cpu()
                        )
                        loss_details[f"gt_depth_{k+1}"] = (
                            gt_depths_cross[k].detach().cpu()
                        )

                imgs_stacked_dict = get_vis_imgs_new(
                    loss_details, args.num_imgs_vis, curr_num_view, is_metric=is_metric
                )
                for name, imgs_stacked in imgs_stacked_dict.items():
                    log_writer.add_images(
                        "train" + "/" + name, imgs_stacked, step, dataformats="HWC"
                    )
                del batch

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator,
            symmetrize_batch=False,
            use_amp=bool(args.amp),
        )

        loss_value, loss_details = result["loss"]  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    printer.info("Averaged stats: %s", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
            if isinstance(val, dict):
                continue
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

        depths_self, gt_depths_self = get_render_results(
            batch, result["pred"], self_view=True
        )
        depths_cross, gt_depths_cross = get_render_results(
            batch, result["pred"], self_view=False
        )
        for k in range(len(batch)):
            loss_details[f"self_pred_depth_{k+1}"] = depths_self[k].detach().cpu()
            loss_details[f"self_gt_depth_{k+1}"] = gt_depths_self[k].detach().cpu()
            loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
            loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()

        # Check if we have the required visualization data before attempting to create images
        # During test phase, some visualization keys might be missing
        required_keys_exist = True
        missing_keys = []
        for i in range(args.num_test_views):
            required_keys = [f"gt_img{i+1}", f"img_mask_{i+1}", f"ray_mask_{i+1}"]
            for key in required_keys:
                if key not in loss_details:
                    required_keys_exist = False
                    missing_keys.append(key)
        
        if not required_keys_exist:
            printer.info(f"Skipping test visualization for {prefix} - missing keys: {missing_keys}")
        
        if required_keys_exist:
            imgs_stacked_dict = get_vis_imgs_new(
                loss_details,
                args.num_imgs_vis,
                args.num_test_views,
                is_metric=batch[0]["is_metric"],
            )
            for name, imgs_stacked in imgs_stacked_dict.items():
                log_writer.add_images(
                    prefix + "/" + name, imgs_stacked, 1000 * epoch, dataformats="HWC"
                )
        else:
            printer.info(f"Skipping test visualization for {prefix} - missing required keys in loss_details")

    del loss_details, loss_value, batch
    torch.cuda.empty_cache()

    return results


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


def gen_mask_indicator(img_mask_list, ray_mask_list, num_views, h, w):
    output = []
    for img_mask, ray_mask in zip(img_mask_list, ray_mask_list):
        out = torch.zeros((h, w * num_views, 3))
        
        # Check if masks have sufficient elements
        if img_mask.numel() == 0 or ray_mask.numel() == 0:
            # If masks are empty, fill with default offset
            out += 0.5
        else:
            # Ensure we don't access beyond tensor bounds
            actual_views = min(num_views, len(img_mask), len(ray_mask))
            for i in range(actual_views):
                if img_mask[i] and not ray_mask[i]:
                    offset = 0
                elif not img_mask[i] and ray_mask[i]:
                    offset = 1
                else:
                    offset = 0.5
                out[:, i * w : (i + 1) * w] += offset
            
            # Fill remaining views with default offset if needed
            for i in range(actual_views, num_views):
                out[:, i * w : (i + 1) * w] += 0.5
        
        output.append(out)
    return output


def safe_quantile(tensor, quantile_val, fallback=0.0):
    """Safely compute quantile, returning fallback if tensor is empty"""
    if tensor.numel() == 0:
        return fallback
    return torch.quantile(tensor, quantile_val).item()

def vis_and_cat(
    gt_imgs,
    pred_imgs,
    cross_gt_depths,
    cross_pred_depths,
    self_gt_depths,
    self_pred_depths,
    cross_conf,
    self_conf,
    ray_indicator,
    is_metric,
):
    # Safe quantile operations for cross depths
    cross_depth_gt_min = safe_quantile(cross_gt_depths, 0.01, 0.0)
    cross_depth_gt_max = safe_quantile(cross_gt_depths, 0.99, 1.0)
    cross_depth_pred_min = safe_quantile(cross_pred_depths, 0.01, 0.0)
    cross_depth_pred_max = safe_quantile(cross_pred_depths, 0.99, 1.0)
    cross_depth_min = min(cross_depth_gt_min, cross_depth_pred_min)
    cross_depth_max = max(cross_depth_gt_max, cross_depth_pred_max)

    # Handle empty tensors for cross depths
    if cross_gt_depths.numel() == 0:
        # Create a dummy tensor for visualization
        cross_gt_depths_vis = torch.zeros((256, 512, 3))  # Default size
    else:
        cross_gt_depths_vis = colorize(
            cross_gt_depths,
            range=(
                (cross_depth_min, cross_depth_max)
                if is_metric
                else (cross_depth_gt_min, cross_depth_gt_max)
            ),
            append_cbar=True,
        )
    
    if cross_pred_depths.numel() == 0:
        # Create a dummy tensor for visualization
        cross_pred_depths_vis = torch.zeros((256, 512, 3))  # Default size
    else:
        cross_pred_depths_vis = colorize(
            cross_pred_depths,
            range=(
                (cross_depth_min, cross_depth_max)
                if is_metric
                else (cross_depth_pred_min, cross_depth_pred_max)
            ),
            append_cbar=True,
        )

    # Safe quantile operations for self depths
    self_depth_gt_min = safe_quantile(self_gt_depths, 0.01, 0.0)
    self_depth_gt_max = safe_quantile(self_gt_depths, 0.99, 1.0)
    self_depth_pred_min = safe_quantile(self_pred_depths, 0.01, 0.0)
    self_depth_pred_max = safe_quantile(self_pred_depths, 0.99, 1.0)
    self_depth_min = min(self_depth_gt_min, self_depth_pred_min)
    self_depth_max = max(self_depth_gt_max, self_depth_pred_max)

    # Handle empty tensors for self depths
    if self_gt_depths.numel() == 0:
        # Create a dummy tensor for visualization with same size as cross depths
        self_gt_depths_vis = torch.zeros_like(cross_gt_depths_vis)
    else:
        self_gt_depths_vis = colorize(
            self_gt_depths,
            range=(
                (self_depth_min, self_depth_max)
                if is_metric
                else (self_depth_gt_min, self_depth_gt_max)
            ),
            append_cbar=True,
        )
    
    if self_pred_depths.numel() == 0:
        # Create a dummy tensor for visualization with same size as cross depths
        self_pred_depths_vis = torch.zeros_like(cross_pred_depths_vis)
    else:
        self_pred_depths_vis = colorize(
            self_pred_depths,
            range=(
                (self_depth_min, self_depth_max)
                if is_metric
                else (self_depth_pred_min, self_depth_pred_max)
            ),
            append_cbar=True,
        )
    if len(cross_conf) > 0 and cross_conf.numel() > 0:
        cross_conf_vis = colorize(cross_conf, append_cbar=True)
    else:
        # Create empty tensor with same shape as cross_gt_depths_vis for consistency
        cross_conf_vis = torch.zeros_like(cross_gt_depths_vis)
    
    if len(self_conf) > 0 and self_conf.numel() > 0:
        self_conf_vis = colorize(self_conf, append_cbar=True)
    else:
        # Create empty tensor with same shape as self_gt_depths_vis for consistency
        self_conf_vis = torch.zeros_like(self_gt_depths_vis)
    gt_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    if gt_imgs.numel() > 0 and len(gt_imgs.shape) >= 2:
        # Safely assign gt_imgs with bounds checking
        h_min = min(gt_imgs.shape[0], gt_imgs_vis.shape[0])
        w_min = min(gt_imgs.shape[1], gt_imgs_vis.shape[1])
        gt_imgs_vis[:h_min, :w_min] = gt_imgs[:h_min, :w_min]
    
    pred_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    if pred_imgs.numel() > 0 and len(pred_imgs.shape) >= 2:
        # Safely assign pred_imgs with bounds checking
        h_min = min(pred_imgs.shape[0], pred_imgs_vis.shape[0])
        w_min = min(pred_imgs.shape[1], pred_imgs_vis.shape[1])
        pred_imgs_vis[:h_min, :w_min] = pred_imgs[:h_min, :w_min]
    # Handle ray_indicator safely
    if ray_indicator.numel() > 0 and len(ray_indicator.shape) >= 2:
        width_diff = cross_pred_depths_vis.shape[1] - ray_indicator.shape[1]
        if width_diff > 0:
            ray_indicator_vis = torch.cat(
                [
                    ray_indicator,
                    torch.zeros(
                        ray_indicator.shape[0],
                        width_diff,
                        ray_indicator.shape[2] if len(ray_indicator.shape) > 2 else 3,
                    ),
                ],
                dim=1,
            )
        else:
            # If ray_indicator is wider than or equal to target, just use it as is
            ray_indicator_vis = ray_indicator[:, :cross_pred_depths_vis.shape[1]]
    else:
        # Create a fallback ray_indicator if the original is empty
        ray_indicator_vis = torch.zeros_like(cross_pred_depths_vis)
    out = torch.cat(
        [
            ray_indicator_vis,
            gt_imgs_vis,
            pred_imgs_vis,
            self_gt_depths_vis,
            self_pred_depths_vis,
            self_conf_vis,
            cross_gt_depths_vis,
            cross_pred_depths_vis,
            cross_conf_vis,
        ],
        dim=0,
    )
    return out


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric):
    ret_dict = {}
    gt_img_list = [[] for _ in range(num_imgs_vis)]
    pred_img_list = [[] for _ in range(num_imgs_vis)]

    cross_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    cross_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    self_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    self_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    cross_view_conf_list = [[] for _ in range(num_imgs_vis)]
    self_view_conf_list = [[] for _ in range(num_imgs_vis)]
    cross_view_conf_exits = False
    self_view_conf_exits = False

    img_mask_list = [[] for _ in range(num_imgs_vis)]
    ray_mask_list = [[] for _ in range(num_imgs_vis)]

    if num_views > 30:
        stride = 5
    elif num_views > 20:
        stride = 3
    elif num_views > 10:
        stride = 2
    else:
        stride = 1
    
    # Safety check: ensure we have at least one view with required keys
    valid_views = []
    for i in range(0, num_views, stride):
        required_keys = [f"gt_img{i+1}", f"gt_depth_{i+1}", f"self_gt_depth_{i+1}", 
                        f"pred_depth_{i+1}", f"self_pred_depth_{i+1}", 
                        f"img_mask_{i+1}", f"ray_mask_{i+1}"]
        if all(key in loss_details for key in required_keys):
            valid_views.append(i)
        else:
            missing_keys = [key for key in required_keys if key not in loss_details]
            printer.debug(f"View {i+1} missing keys: {missing_keys}")
    
    printer.info(f"Found {len(valid_views)} valid views out of {len(range(0, num_views, stride))} total views")
    
    if not valid_views:
        printer.warning("No valid views found for visualization - returning empty dict")
        return ret_dict
    
    # Process each valid view with error handling
    for i in valid_views:
        try:
            gt_imgs = 0.5 * (loss_details[f"gt_img{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
            width = gt_imgs.shape[2]
            # Check if RGB predictions exist, otherwise use GT images or zeros
            if f"pred_rgb_{i+1}" in loss_details:
                pred_imgs = (
                    0.5 * (loss_details[f"pred_rgb_{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
                )
            else:
                # Use GT images as fallback when RGB predictions are not available
                pred_imgs = gt_imgs.clone()
            gt_img_list = batch_append(gt_img_list, gt_imgs.unbind(dim=0))
            pred_img_list = batch_append(pred_img_list, pred_imgs.unbind(dim=0))

            cross_pred_depths = (
                loss_details[f"pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            cross_gt_depths = (
                loss_details[f"gt_depth_{i+1}"]
                .to(gt_imgs.device)[:num_imgs_vis]
                .detach()
                .cpu()
            )
            cross_pred_depth_list = batch_append(
                cross_pred_depth_list, cross_pred_depths.unbind(dim=0)
            )
            cross_gt_depth_list = batch_append(
                cross_gt_depth_list, cross_gt_depths.unbind(dim=0)
            )

            self_gt_depths = (
                loss_details[f"self_gt_depth_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            self_pred_depths = (
                loss_details[f"self_pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            self_gt_depth_list = batch_append(
                self_gt_depth_list, self_gt_depths.unbind(dim=0)
            )
            self_pred_depth_list = batch_append(
                self_pred_depth_list, self_pred_depths.unbind(dim=0)
            )

            if f"conf_{i+1}" in loss_details:
                cross_view_conf = loss_details[f"conf_{i+1}"][:num_imgs_vis].detach().cpu()
                cross_view_conf_list = batch_append(
                    cross_view_conf_list, cross_view_conf.unbind(dim=0)
                )
                cross_view_conf_exits = True

            if f"self_conf_{i+1}" in loss_details:
                self_view_conf = (
                    loss_details[f"self_conf_{i+1}"][:num_imgs_vis].detach().cpu()
                )
                self_view_conf_list = batch_append(
                    self_view_conf_list, self_view_conf.unbind(dim=0)
                )
                self_view_conf_exits = True

            img_mask_list = batch_append(
                img_mask_list,
                loss_details[f"img_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
            )
            ray_mask_list = batch_append(
                ray_mask_list,
                loss_details[f"ray_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
            )
        except Exception as e:
            printer.warning(f"Error processing view {i+1}: {e}")
            continue

    # Check if we have any data to concatenate
    if not gt_img_list or not any(sublist for sublist in gt_img_list):
        printer.warning("No valid image data collected - returning empty dict")
        return ret_dict

    # each element in the list is [H, num_views * W, (3)], the size of the list is num_imgs_vis
    # Safe concatenation with empty list handling
    gt_img_list = [torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in gt_img_list]
    pred_img_list = [torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in pred_img_list]
    cross_pred_depth_list = [
        torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in cross_pred_depth_list
    ]
    cross_gt_depth_list = [torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in cross_gt_depth_list]
    self_gt_depth_list = [torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in self_gt_depth_list]
    self_pred_depth_list = [
        torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in self_pred_depth_list
    ]
    cross_view_conf_list = (
        [torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in cross_view_conf_list]
        if cross_view_conf_exits
        else []
    )
    self_view_conf_list = (
        [torch.cat(sublist, dim=1) if sublist else torch.empty(0) for sublist in self_view_conf_list]
        if self_view_conf_exits
        else []
    )
    # each elment in the list is [num_views,], the size of the list is num_imgs_vis
    img_mask_list = [torch.stack(sublist, dim=0) if sublist else torch.empty(0) for sublist in img_mask_list]
    ray_mask_list = [torch.stack(sublist, dim=0) if sublist else torch.empty(0) for sublist in ray_mask_list]

    # Only generate visualizations if we have actual data
    if not gt_img_list or len(gt_img_list) == 0:
        printer.warning("No valid image data found for visualization")
        return ret_dict
    
    if gt_img_list[0].numel() == 0:
        printer.warning("No valid image data found for visualization")
        return ret_dict
    
    # Ensure we have a width value (from the last valid view)
    if not valid_views:
        printer.warning("No valid views processed - cannot generate visualizations")
        return ret_dict
    
    # Get width from the first valid image
    try:
        width = gt_img_list[0].shape[2] if gt_img_list[0].numel() > 0 else 512  # fallback width
    except (IndexError, AttributeError):
        width = 512  # fallback width
        printer.warning("Could not determine image width, using fallback value 512")
    
    ray_indicator = gen_mask_indicator(
        img_mask_list, ray_mask_list, len(img_mask_list[0]) if img_mask_list and img_mask_list[0].numel() > 0 else 0, 30, width
    )

    for i in range(num_imgs_vis):
        if i < len(gt_img_list) and i < len(ray_indicator):
            # Safe indexing for is_metric - use first element if index is out of bounds
            metric_idx = min(i, len(is_metric) - 1) if len(is_metric) > 0 else 0
            metric_value = is_metric[metric_idx] if len(is_metric) > 0 else False
            
            out = vis_and_cat(
                gt_img_list[i],
                pred_img_list[i],
                cross_gt_depth_list[i],
                cross_pred_depth_list[i],
                self_gt_depth_list[i],
                self_pred_depth_list[i],
                cross_view_conf_list[i],
                self_view_conf_list[i],
                ray_indicator[i],
                metric_value,
            )
            ret_dict[f"imgs_{i}"] = out
    return ret_dict


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train_imu.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    # Handle deterministic mode
    deterministic = getattr(cfg, 'deterministic', False)
    if deterministic:
        print("Enabling deterministic mode...")
        # Set deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # Set seed if not already set (though train() sets it too, setting it early is good)
        seed = getattr(cfg, 'seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Deterministic mode enabled with seed {seed}")

    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
