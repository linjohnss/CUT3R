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
        return CUT3RIMU(base_model, imu_config)
    else:
        # Return original model if no IMU config
        return base_model


def freeze_parameters_for_imu_training(model):
    """
    Freeze parameters for new IMU training with IMUAwarePoseRetriever:
    - Keep CUT3R frozen (except LoRA if enabled)
    - Train IMU encoder
    - Train NEW IMUAwarePoseRetriever
    - FREEZE original pose_retriever
    - Train LoRA parameters if enabled
    """
    if not hasattr(model, 'imu_encoder'):
        # Not an IMU model, don't change anything
        printer.info("Not an IMU model, skipping parameter freezing")
        return
    
    printer.info("=== IMU PARAMETER FREEZING ===")
    
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze IMU encoder
    imu_encoder_params = 0
    for param in model.imu_encoder.parameters():
        param.requires_grad = True
        imu_encoder_params += 1
    printer.info(f"Unfroze {imu_encoder_params} IMU encoder parameters")
    
    # Unfreeze NEW IMU-aware pose retriever
    imu_pose_retriever_params = 0
    if hasattr(model, 'imu_pose_retriever'):
        for param in model.imu_pose_retriever.parameters():
            param.requires_grad = True
            imu_pose_retriever_params += 1
        printer.info(f"Unfroze {imu_pose_retriever_params} IMU pose retriever parameters")
    
    # Unfreeze relative pose decoder
    relative_pose_decoder_params = 0
    if hasattr(model, 'relative_pose_decoder'):
        for param in model.relative_pose_decoder.parameters():
            param.requires_grad = True
            relative_pose_decoder_params += 1
        printer.info(f"Unfroze {relative_pose_decoder_params} relative pose decoder parameters")
    
    # Unfreeze pose encoder (encode 7D camera pose back to latent token)
    pose_encoder_params = 0
    if hasattr(model, 'pose_encoder'):
        for param in model.pose_encoder.parameters():
            param.requires_grad = True
            pose_encoder_params += 1
        printer.info(f"Unfroze {pose_encoder_params} pose encoder parameters")
    
    # Unfreeze pose token transformer
    pose_token_transformer_params = 0
    if hasattr(model, 'pose_token_transformer'):
        for param in model.pose_token_transformer.parameters():
            param.requires_grad = True
            pose_token_transformer_params += 1
        printer.info(f"Unfroze {pose_token_transformer_params} pose token transformer parameters")
    
    # Unfreeze pose token fusion MLP
    pose_token_fusion_mlp_params = 0
    if hasattr(model, 'pose_token_fusion_mlp'):
        for param in model.pose_token_fusion_mlp.parameters():
            param.requires_grad = True
            pose_token_fusion_mlp_params += 1
        printer.info(f"Unfroze {pose_token_fusion_mlp_params} pose token fusion MLP parameters")
    
    # EXPLICITLY freeze original pose retriever
    original_pose_retriever_params = 0
    if hasattr(model, 'cut3r_model') and hasattr(model.cut3r_model, 'pose_retriever'):
        for param in model.cut3r_model.pose_retriever.parameters():
            param.requires_grad = False
            original_pose_retriever_params += 1
        printer.info(f"Explicitly froze {original_pose_retriever_params} original pose retriever parameters")

    # Unfreeze LoRA parameters ONLY if LoRA training is enabled
    # Note: This should be controlled by the training configuration
    lora_params = 0
    if hasattr(model, 'cut3r_model'):
        for name, param in model.cut3r_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                # Only unfreeze LoRA if it's explicitly enabled in training mode
                # This will be controlled by the calling function based on args.training_mode.enable_lora
                param.requires_grad = False  # Default to frozen
                lora_params += 1
    if lora_params > 0:
        printer.info(f"Found {lora_params} LoRA parameters (currently frozen - will be controlled by training mode)")
    
    # Count total trainable parameters
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    printer.info(f"Total parameters: {total_params:,}")
    printer.info(f"Trainable parameters: {total_trainable:,}")
    printer.info(f"Trainable ratio: {total_trainable/total_params*100:.3f}%")
    
    # Breakdown of trainable parameters
    if hasattr(model, 'imu_encoder'):
        imu_enc_trainable = sum(p.numel() for p in model.imu_encoder.parameters() if p.requires_grad)
        printer.info(f"  - IMU Encoder: {imu_enc_trainable:,} parameters")
    
    if hasattr(model, 'imu_pose_retriever'):
        imu_ret_trainable = sum(p.numel() for p in model.imu_pose_retriever.parameters() if p.requires_grad)
        printer.info(f"  - IMU Pose Retriever: {imu_ret_trainable:,} parameters")
    
    if hasattr(model, 'relative_pose_decoder'):
        rel_dec_trainable = sum(p.numel() for p in model.relative_pose_decoder.parameters() if p.requires_grad)
        printer.info(f"  - Relative Pose Decoder: {rel_dec_trainable:,} parameters")
    if hasattr(model, 'pose_encoder'):
        pose_enc_trainable = sum(p.numel() for p in model.pose_encoder.parameters() if p.requires_grad)
        printer.info(f"  - Pose Encoder: {pose_enc_trainable:,} parameters")
    
    if hasattr(model, 'pose_token_transformer'):
        pose_trans_trainable = sum(p.numel() for p in model.pose_token_transformer.parameters() if p.requires_grad)
        printer.info(f"  - Pose Token Transformer: {pose_trans_trainable:,} parameters")
    
    if hasattr(model, 'pose_token_fusion_mlp'):
        pose_mlp_trainable = sum(p.numel() for p in model.pose_token_fusion_mlp.parameters() if p.requires_grad)
        printer.info(f"  - Pose Token Fusion MLP: {pose_mlp_trainable:,} parameters")
    
    if total_trainable == 0:
        printer.error("❌ NO TRAINABLE PARAMETERS FOUND!")
        raise RuntimeError("No trainable parameters found after IMU parameter freezing")
    
    printer.info("=== END IMU PARAMETER FREEZING ===")


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

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

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

    # training dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
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

    # model
    printer.info("Loading model: %s", args.model)
    base_model: PreTrainedModel = eval(args.model)
    
    # Create IMU-enhanced model if IMU config is provided
    imu_config = getattr(args, 'imu_config', None)
    model = create_imu_enhanced_model(base_model, imu_config)
    
    # Check LoRA configuration (access through base_model for IMU models)
    actual_model = model.cut3r_model if hasattr(model, 'cut3r_model') else model
    enable_lora = getattr(args.training_mode, 'enable_lora', False)
    
    if enable_lora and getattr(actual_model, 'config', None) and getattr(actual_model.config, 'enable_lora', False):
        printer.info("=== LoRA MODEL ANALYSIS ===")
        cfg = actual_model.config
        printer.info(f"LoRA enabled: {cfg.enable_lora}, rank: {cfg.lora_rank}, alpha: {cfg.lora_alpha}, dropout: {cfg.lora_dropout}")

        lora_layers = [name for name, module in actual_model.named_modules() if 'LoRA' in str(type(module))]
        if lora_layers:
            printer.info(f"✅ LoRA successfully applied to {len(lora_layers)} layers")
            for name in lora_layers:
                printer.info(f"Found LoRA layer: {name}")
        else:
            printer.warning("❌ NO LoRA LAYERS FOUND! This means LoRA was not applied correctly.")
            printer.warning("The model will train normally but won't generate LoRA weights.")
            # Show up to 10 Linear layer names for debugging
            linear_names = [name for name, module in actual_model.named_modules() if isinstance(module, torch.nn.Linear)]
            printer.info("Sample linear layer names in model: " + ", ".join(linear_names[:10]))
        printer.info("=== END LoRA ANALYSIS ===")
    elif enable_lora:
        printer.warning("LoRA training enabled but model does not support LoRA!")
    else:
        printer.info("LoRA training disabled")
    
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")
    printer.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Handle parameter counting for both regular and IMU models
    if hasattr(model, 'cut3r_model'):
        # IMU-enhanced model
        printer.info(f"CUT3R encoder parameters: {sum(p.numel() for p in model.cut3r_model.enc_blocks.parameters())}")
        printer.info(f"CUT3R decoder parameters: {sum(p.numel() for p in model.cut3r_model.dec_blocks.parameters())}")
        if hasattr(model, 'imu_encoder'):
            printer.info(f"IMU encoder parameters: {sum(p.numel() for p in model.imu_encoder.parameters())}")
        if hasattr(model, 'imu_weight') and hasattr(model, 'pose_weight'):
            weight_params = model.imu_weight.numel() + model.pose_weight.numel()
            printer.info(f"IMU fusion weight parameters: {weight_params:,}")
    else:
        # Regular model
        printer.info(f"Encoder parameters: {sum(p.numel() for p in model.enc_blocks.parameters())}")
        printer.info(f"Decoder parameters: {sum(p.numel() for p in model.dec_blocks.parameters())}")

    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        # Check if this is LoRA or frozen training
        enable_lora = getattr(args.training_mode, 'enable_lora', False)
        freeze_cut3r = getattr(args.training_mode, 'freeze_cut3r', True)
        
        if (enable_lora or freeze_cut3r) and getattr(args.training_mode, 'freeze_cut3r', True):
            # FORCE disable gradient checkpointing for frozen/LoRA training
            # Gradient checkpointing is incompatible with frozen parameters
            printer.warning("⚠️  FORCE disabling gradient checkpointing for frozen parameter training")
            printer.warning("   Gradient checkpointing conflicts with frozen parameters in LoRA/selective training mode")
            args.gradient_checkpointing = False
            # Also disable on model
            if hasattr(model, 'gradient_checkpointing'):
                model.gradient_checkpointing = False
        else:
            printer.info("Enabling gradient checkpointing for full model training")
            model.gradient_checkpointing_enable()
    else:
        printer.info("Gradient checkpointing disabled")
        # Ensure it's disabled on model too
        if hasattr(model, 'gradient_checkpointing'):
            model.gradient_checkpointing = False
    
    if args.long_context:
        model.fixed_input_length = False

    # Handle LoRA and pretrained loading - work with the base model
    target_model = model.cut3r_model if hasattr(model, 'cut3r_model') else model
    enable_lora = getattr(args.training_mode, 'enable_lora', False)
    
    # Load pretrained weights FIRST (before applying LoRA if enabled)
    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        load_only_encoder = getattr(args, "load_only_encoder", False)
        state_dict = ckpt["model"]
        if load_only_encoder:
            state_dict = {k: v for k, v in state_dict.items() if "enc_blocks" in k or "patch_embed" in k}
        
        # Actually load the state dict to the base model
        printer.info(
            target_model.load_state_dict(strip_module(state_dict), strict=False)
        )
        del ckpt
    
    # Apply LoRA if enabled and configured
    if enable_lora and getattr(target_model, '_lora_config', None):
        lora_config = target_model._lora_config
        printer.info(f"=== APPLYING LORA TO MODEL ===\n"
                     f"LoRA config: rank={lora_config['lora_rank']}, alpha={lora_config['lora_alpha']}, "
                     f"dropout={lora_config['lora_dropout']}, targets={lora_config['lora_target_modules']}")
        
        # Apply LoRA AFTER loading pretrained weights
        target_model.apply_lora(
            rank=lora_config['lora_rank'],
            alpha=lora_config['lora_alpha'],
            dropout=lora_config['lora_dropout'],
            target_modules=lora_config['lora_target_modules'],
        )
        
        # Freeze non-LoRA parameters on the base model
        freeze_non_lora_parameters(target_model)
        printer.info("=== END LORA APPLICATION ===")
    elif enable_lora:
        printer.warning("LoRA training enabled but no LoRA config found in model!")
    else:
        printer.info("LoRA training disabled")
    
    # Apply IMU-specific parameter freezing if this is an IMU model
    freeze_parameters_for_imu_training(model)
    
    # Handle LoRA parameter training based on training mode
    enable_lora = getattr(args.training_mode, 'enable_lora', False)
    if enable_lora and hasattr(model, 'cut3r_model'):
        printer.info("=== ENABLING LoRA PARAMETER TRAINING ===")
        lora_trainable_count = 0
        for name, param in model.cut3r_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                lora_trainable_count += 1
        printer.info(f"Enabled training for {lora_trainable_count} LoRA parameters")
        
        # Recalculate trainable parameters
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        printer.info(f"Updated - Trainable parameters: {total_trainable:,} / {total_params:,} ({total_trainable/total_params*100:.3f}%)")
        printer.info("=== END LoRA PARAMETER ENABLEMENT ===")
    elif hasattr(model, 'cut3r_model'):
        # Ensure LoRA parameters are frozen when LoRA training is disabled
        lora_frozen_count = 0
        for name, param in model.cut3r_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False
                lora_frozen_count += 1
        if lora_frozen_count > 0:
            printer.info(f"Ensured {lora_frozen_count} LoRA parameters remain frozen (LoRA training disabled)")

    # # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler(accelerator=accelerator)

    accelerator.even_batches = False
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
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )
        
        # Save LoRA and IMU weights separately based on training mode
        actual_model = accelerator.unwrap_model(model)
        base_model = actual_model.cut3r_model if hasattr(actual_model, 'cut3r_model') else actual_model
        enable_lora = getattr(args.training_mode, 'enable_lora', False)
        enable_imu = getattr(args.training_mode, 'enable_imu', False)
        
        # Save LoRA weights if LoRA training is enabled
        if enable_lora and hasattr(base_model, 'config') and hasattr(base_model.config, 'enable_lora') and base_model.config.enable_lora:
            lora_path = os.path.join(args.output_dir, f"lora_weights_{fname}.pth")
            lora_state_dict = get_lora_state_dict(base_model)
            if lora_state_dict and accelerator.is_main_process:
                torch.save(lora_state_dict, lora_path)
                total_params = sum(w.numel() for w in lora_state_dict.values())
                printer.info(f"Saved LoRA weights to {lora_path} ({total_params:,} parameters)")
            elif accelerator.is_main_process:
                printer.warning(f"⚠️  No LoRA weights to save for checkpoint {fname}!")
        
        # Save IMU weights if IMU training is enabled
        if enable_imu and hasattr(actual_model, 'imu_encoder') and accelerator.is_main_process:
            imu_path = os.path.join(args.output_dir, f"imu_weights_{fname}.pth")
            imu_state_dict = {
                'imu_encoder': actual_model.imu_encoder.state_dict(),
            }
            
            # Save new IMU-aware pose retriever if it exists
            if hasattr(actual_model, 'imu_pose_retriever'):
                imu_state_dict['imu_pose_retriever'] = actual_model.imu_pose_retriever.state_dict()
            
            # Save relative pose decoder if it exists
            if hasattr(actual_model, 'relative_pose_decoder'):
                imu_state_dict['relative_pose_decoder'] = actual_model.relative_pose_decoder.state_dict()
            
            # Save pose encoder if it exists
            if hasattr(actual_model, 'pose_encoder'):
                imu_state_dict['pose_encoder'] = actual_model.pose_encoder.state_dict()
            
            # Save pose token transformer if it exists
            if hasattr(actual_model, 'pose_token_transformer'):
                imu_state_dict['pose_token_transformer'] = actual_model.pose_token_transformer.state_dict()
            
            # Save pose token fusion MLP if it exists
            if hasattr(actual_model, 'pose_token_fusion_mlp'):
                imu_state_dict['pose_token_fusion_mlp'] = actual_model.pose_token_fusion_mlp.state_dict()
            
            # Keep old fusion weights for backward compatibility (but they're not used in new architecture)
            if hasattr(actual_model, 'imu_weight'):
                imu_state_dict['imu_weight'] = actual_model.imu_weight.data
            if hasattr(actual_model, 'pose_weight'):
                imu_state_dict['pose_weight'] = actual_model.pose_weight.data
            
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

    # Load checkpoints (handles both regular and LoRA training)
    if args.resume:
        # For LoRA training, we need to load LoRA weights from checkpoint
        actual_model = accelerator.unwrap_model(model)
        if hasattr(actual_model, '_lora_config') and actual_model._lora_config is not None:
            printer.info("=== LOADING LORA CHECKPOINT ===")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load LoRA weights if they exist in checkpoint
            if 'lora_weights' in checkpoint:
                from lora_utils import load_lora_state_dict
                load_lora_state_dict(actual_model, checkpoint['lora_weights'])
                printer.info("Successfully loaded LoRA weights from checkpoint")
            else:
                printer.warning("No LoRA weights found in checkpoint!")
            
            # Load optimizer state (skip if parameter groups don't match)
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    printer.info("Loaded optimizer state")
                except Exception as e:
                    printer.warning(f"Failed to load optimizer state: {e}")
                    printer.warning("Continuing with fresh optimizer state...")
            
            # Load other training state
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                printer.info(f"Resuming from epoch {args.start_epoch}")
            
            # Load best_so_far
            best_so_far = checkpoint.get('best_so_far', float('inf'))
            printer.info(f"Best loss so far: {best_so_far}")
            
            printer.info("=== END LORA CHECKPOINT LOADING ===")
        else:
            # Standard checkpoint loading with optimizer state handling
            printer.info("=== LOADING STANDARD CHECKPOINT ===")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model weights
            model.load_state_dict(checkpoint["model"], strict=False)
            printer.info("Loaded model weights")
            
            # Load optimizer state (skip if parameter groups don't match)
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    printer.info("Loaded optimizer state")
                except Exception as e:
                    printer.warning(f"Failed to load optimizer state: {e}")
                    printer.warning("Continuing with fresh optimizer state...")
            
            # Load other training state
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                printer.info(f"Resuming from epoch {args.start_epoch}")
            else:
                args.start_epoch = 0
            
            # Load scaler state
            if 'scaler' in checkpoint and loss_scaler is not None:
                try:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                    printer.info("Loaded loss scaler state")
                except Exception as e:
                    printer.warning(f"Failed to load loss scaler state: {e}")
            
            # Load best_so_far
            best_so_far = checkpoint.get('best_so_far', float('inf'))
            printer.info(f"Best loss so far: {best_so_far}")
            
            printer.info("=== END STANDARD CHECKPOINT LOADING ===")
    else:
        # No checkpoint to resume from
        args.start_epoch = 0
        best_so_far = float('inf')
        printer.info("Starting training from scratch")
    
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
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)


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
    
    checkpoint_path = output_dir / "checkpoint-final.pth"
    
    if is_lora_training or lora_param_count > 0 or is_imu_training:
        # Save enhanced checkpoint (LoRA and/or IMU)
        to_save = {
            "args": args,
            "epoch": epoch,
            "model_config": getattr(base_model, 'config', None),
        }
        if best_so_far is not None:
            to_save["best_so_far"] = best_so_far
        
        # Add LoRA weights if available
        if lora_state_dict:
            to_save["lora_weights"] = lora_state_dict
        
        # Add IMU config if available
        if is_imu_training:
            to_save["imu_config"] = getattr(args, 'imu_config', {})
        
        printer.info(f">> Saving enhanced checkpoint to {checkpoint_path} ...")
        misc.save_on_master(accelerator, to_save, checkpoint_path)
        
        # Save standalone LoRA weights if LoRA training is enabled
        if enable_lora and lora_state_dict and accelerator.is_main_process:
            lora_path = output_dir / "lora_weights_final.pth"
            torch.save(lora_state_dict, lora_path)
            total_params = sum(w.numel() for w in lora_state_dict.values())
            printer.info(f"✅ Saved LoRA weights: {len(lora_state_dict)} tensors, {total_params:,} params")
        
        # Save standalone IMU weights if IMU training is enabled
        if enable_imu and is_imu_training and accelerator.is_main_process:
            imu_path = output_dir / "imu_weights_final.pth"
            imu_state_dict = {
                'imu_encoder': actual_model.imu_encoder.state_dict(),
            }
            
            # Save new IMU-aware pose retriever if it exists
            if hasattr(actual_model, 'imu_pose_retriever'):
                imu_state_dict['imu_pose_retriever'] = actual_model.imu_pose_retriever.state_dict()
            
            # Save relative pose decoder if it exists
            if hasattr(actual_model, 'relative_pose_decoder'):
                imu_state_dict['relative_pose_decoder'] = actual_model.relative_pose_decoder.state_dict()
            
            # Save pose encoder if it exists
            if hasattr(actual_model, 'pose_encoder'):
                imu_state_dict['pose_encoder'] = actual_model.pose_encoder.state_dict()
            
            # Save pose token transformer if it exists
            if hasattr(actual_model, 'pose_token_transformer'):
                imu_state_dict['pose_token_transformer'] = actual_model.pose_token_transformer.state_dict()
            
            # Save pose token fusion MLP if it exists
            if hasattr(actual_model, 'pose_token_fusion_mlp'):
                imu_state_dict['pose_token_fusion_mlp'] = actual_model.pose_token_fusion_mlp.state_dict()
            
            # Keep old fusion weights for backward compatibility (but they're not used in new architecture)
            if hasattr(actual_model, 'imu_weight'):
                imu_state_dict['imu_weight'] = actual_model.imu_weight.data
            if hasattr(actual_model, 'pose_weight'):
                imu_state_dict['pose_weight'] = actual_model.pose_weight.data
            
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
    else:
        # Save standard model
        to_save = {
            "args": args,
            "model": actual_model.cpu().state_dict() if not isinstance(actual_model, dict) else actual_model,
            "epoch": epoch,
        }
        if best_so_far is not None:
            to_save["best_so_far"] = best_so_far
        printer.info(f">> Saving model to {checkpoint_path} ...")
        misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
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
        fixed_length=fixed_length
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
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )
        
        # Save LoRA and IMU weights separately based on training mode
        actual_model = accelerator.unwrap_model(model)
        base_model = actual_model.cut3r_model if hasattr(actual_model, 'cut3r_model') else actual_model
        enable_lora = getattr(args.training_mode, 'enable_lora', False)
        enable_imu = getattr(args.training_mode, 'enable_imu', False)
        
        # Save LoRA weights if LoRA training is enabled
        if enable_lora and hasattr(base_model, 'config') and hasattr(base_model.config, 'enable_lora') and base_model.config.enable_lora:
            lora_path = os.path.join(args.output_dir, f"lora_weights_{fname}.pth")
            lora_state_dict = get_lora_state_dict(base_model)
            if lora_state_dict and accelerator.is_main_process:
                torch.save(lora_state_dict, lora_path)
                total_params = sum(w.numel() for w in lora_state_dict.values())
                printer.info(f"Saved LoRA weights to {lora_path} ({total_params:,} parameters)")
            elif accelerator.is_main_process:
                printer.warning(f"⚠️  No LoRA weights to save for checkpoint {fname}!")
        
        # Save IMU weights if IMU training is enabled
        if enable_imu and hasattr(actual_model, 'imu_encoder') and accelerator.is_main_process:
            imu_path = os.path.join(args.output_dir, f"imu_weights_{fname}.pth")
            imu_state_dict = {
                'imu_encoder': actual_model.imu_encoder.state_dict(),
            }
            
            # Save new IMU-aware pose retriever if it exists
            if hasattr(actual_model, 'imu_pose_retriever'):
                imu_state_dict['imu_pose_retriever'] = actual_model.imu_pose_retriever.state_dict()
            
            # Save relative pose decoder if it exists
            if hasattr(actual_model, 'relative_pose_decoder'):
                imu_state_dict['relative_pose_decoder'] = actual_model.relative_pose_decoder.state_dict()
            
            # Save pose token transformer if it exists
            if hasattr(actual_model, 'pose_token_transformer'):
                imu_state_dict['pose_token_transformer'] = actual_model.pose_token_transformer.state_dict()
            
            # Save pose token fusion MLP if it exists
            if hasattr(actual_model, 'pose_token_fusion_mlp'):
                imu_state_dict['pose_token_fusion_mlp'] = actual_model.pose_token_fusion_mlp.state_dict()
            
            # Keep old fusion weights for backward compatibility (but they're not used in new architecture)
            if hasattr(actual_model, 'imu_weight'):
                imu_state_dict['imu_weight'] = actual_model.imu_weight.data
            if hasattr(actual_model, 'pose_weight'):
                imu_state_dict['pose_weight'] = actual_model.pose_weight.data
            
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
                misc.adjust_learning_rate(optimizer, epoch_f, args)
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
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
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

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
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
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
