"""
LoRA (Low-Rank Adaptation) utilities for fine-tuning large models.
Efficient adaptation of pre-trained models using low-rank decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import re
import math
import warnings
from collections import OrderedDict


class LoRALinear(nn.Module):
    """
    LoRA Linear layer that adapts a pre-trained linear layer with low-rank matrices.
    
    This layer wraps around an existing linear layer and adds low-rank adaptation
    matrices A and B such that the adapted weight is: W + B @ A * scaling
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = True,
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA matrices
        if rank > 0:
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.scaling = alpha / rank
            self.merged = False
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0
            self.merged = True
        
        # Dropout
        if dropout > 0.0:
            self.lora_dropout = nn.Dropout(dropout)
        else:
            self.lora_dropout = nn.Identity()
    
    def reset_parameters(self):
        """Reset LoRA parameters to initial values"""
        if self.rank > 0:
            # Initialize A with random values and B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype
        
        if self.rank > 0 and not self.merged:
            # LoRA forward pass
            result = self.linear(x)
            
            # Apply LoRA adaptation
            if self.rank > 0:
                x_lora = self.lora_dropout(x)
                if self.fan_in_fan_out:
                    result += (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling
                else:
                    result += (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling
            
            result = result.to(previous_dtype)
            return result
        else:
            return self.linear(x)
    
    def merge(self):
        """Merge LoRA weights into the original layer"""
        if self.rank > 0 and not self.merged:
            # Merge LoRA weights
            delta_w = self.lora_B @ self.lora_A * self.scaling
            if self.fan_in_fan_out:
                self.linear.weight.data += delta_w.T
            else:
                self.linear.weight.data += delta_w
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from the original layer"""
        if self.rank > 0 and self.merged:
            # Unmerge LoRA weights
            delta_w = self.lora_B @ self.lora_A * self.scaling
            if self.fan_in_fan_out:
                self.linear.weight.data -= delta_w.T
            else:
                self.linear.weight.data -= delta_w
            self.merged = False
    
    def train(self, mode: bool = True):
        """Set training mode"""
        super().train(mode)
        if self.merge_weights and self.merged:
            warnings.warn("Merging weights in training mode is not recommended")
    
    def eval(self):
        """Set evaluation mode"""
        super().eval()
        if self.merge_weights and not self.merged:
            self.merge()
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


def replace_linear_with_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    merge_weights: bool = True,
    fan_in_fan_out: bool = False,
    **kwargs
) -> nn.Module:
    """
    Replace linear layers in a model with LoRA layers.
    
    Args:
        model: The model to modify
        target_modules: List of module names to replace (supports regex)
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
        merge_weights: Whether to merge weights during evaluation
        fan_in_fan_out: Whether to use fan_in_fan_out mode
        
    Returns:
        Modified model with LoRA layers
    """
    replaced_count = 0
    all_linear_layers = []
    
    # First, collect all linear layers for debugging
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_linear_layers.append(name)
    
    print(f"Found {len(all_linear_layers)} linear layers in model")
    if len(all_linear_layers) <= 20:  # Show all if not too many
        print("Linear layers found:")
        for name in all_linear_layers:
            print(f"  {name}")
    else:
        print("Sample linear layers:")
        for name in all_linear_layers[:10]:
            print(f"  {name}")
        print(f"  ... and {len(all_linear_layers) - 10} more")
    
    def _replace_module(parent_module, child_name, target_modules, parent_path=""):
        nonlocal replaced_count
        child_module = getattr(parent_module, child_name)
        
        # Build full module path
        full_path = f"{parent_path}.{child_name}" if parent_path else child_name
        
        # Check if this module should be replaced
        should_replace = False
        matched_pattern = None
        
        if isinstance(child_module, nn.Linear):
            for target_pattern in target_modules:
                try:
                    if re.search(target_pattern, full_path):
                        should_replace = True
                        matched_pattern = target_pattern
                        break
                except Exception as e:
                    print(f"Warning: regex pattern '{target_pattern}' failed: {e}")
                    continue
        
        if should_replace:
            # Create LoRA layer
            lora_layer = LoRALinear(
                in_features=child_module.in_features,
                out_features=child_module.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=child_module.bias is not None,
                merge_weights=merge_weights,
                fan_in_fan_out=fan_in_fan_out,
                **kwargs
            )
            
            # Copy weights and bias
            lora_layer.linear.weight.data = child_module.weight.data.clone()
            if child_module.bias is not None:
                lora_layer.linear.bias.data = child_module.bias.data.clone()
            
            # Replace the module
            setattr(parent_module, child_name, lora_layer)
            replaced_count += 1
            print(f"Replaced {full_path} with LoRA layer (rank={rank}, alpha={alpha}, matched: {matched_pattern})")
    
    # Recursively replace modules
    def _recursive_replace(module, prefix=""):
        for name, child in module.named_children():
            full_path = f"{prefix}.{name}" if prefix else name
            _replace_module(module, name, target_modules, prefix)
            _recursive_replace(child, full_path)
    
    print(f"Applying LoRA with target patterns: {target_modules}")
    _recursive_replace(model)
    
    if replaced_count == 0:
        print("WARNING: No modules were replaced! Check your target patterns.")
        print("Consider using more general patterns like:")
        print("  - r'.*linear.*' for any module with 'linear' in the name")
        print("  - r'.*\\.fc\\d+$' for fully connected layers")
        print("  - r'.*\\.weight$' (this won't work as it targets parameters, not modules)")
        
        # Try to suggest patterns based on actual module names
        suggested_patterns = []
        for name in all_linear_layers[:5]:  # Check first few layers
            parts = name.split('.')
            if len(parts) >= 2:
                # Create pattern for the last two parts
                pattern = f".*\\.{re.escape(parts[-2])}\\.{re.escape(parts[-1])}$"
                suggested_patterns.append(pattern)
        
        if suggested_patterns:
            print("Suggested patterns based on your model:")
            for pattern in suggested_patterns[:3]:
                print(f"  {pattern}")
    else:
        print(f"Successfully replaced {replaced_count} linear layers with LoRA")
    
    return model


def get_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters from a model.
    
    Args:
        model: Model containing LoRA layers
        
    Returns:
        Dictionary of LoRA parameters
    """
    lora_params = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            lora_params[f"{name}.lora_A"] = module.lora_A
            lora_params[f"{name}.lora_B"] = module.lora_B
    
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, Any]:
    """
    Get state dict containing only LoRA parameters.
    
    Args:
        model: Model containing LoRA layers
        
    Returns:
        State dict with LoRA parameters
    """
    lora_state_dict = {}
    
    # Method 1: Check for LoRALinear instances
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    
    # Method 2: If no LoRALinear instances found, extract from state_dict directly
    if not lora_state_dict:
        full_state_dict = model.state_dict()
        for key, value in full_state_dict.items():
            if 'lora_A' in key or 'lora_B' in key:
                lora_state_dict[key] = value.data if hasattr(value, 'data') else value
    
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, Any], strict: bool = True):
    """
    Load LoRA parameters from state dict.
    
    Args:
        model: Model containing LoRA layers
        state_dict: State dict with LoRA parameters
        strict: Whether to strictly enforce matching keys
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            lora_a_key = f"{name}.lora_A"
            lora_b_key = f"{name}.lora_B"
            
            if lora_a_key in state_dict:
                module.lora_A.data = state_dict[lora_a_key]
            elif strict:
                raise KeyError(f"Missing key {lora_a_key} in state dict")
            
            if lora_b_key in state_dict:
                module.lora_B.data = state_dict[lora_b_key]
            elif strict:
                raise KeyError(f"Missing key {lora_b_key} in state dict")


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights in the model"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights in the model"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def freeze_non_lora_parameters(model: nn.Module):
    """
    Freeze all non-LoRA parameters in the model.
    Only LoRA A/B matrices should remain trainable - be very strict!
    
    Args:
        model: Model containing LoRA layers
    """
    lora_param_count = 0
    frozen_param_count = 0
    other_trainable = 0
    
    for name, param in model.named_parameters():
        # ONLY LoRA A/B parameters should be trainable - be very strict
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            lora_param_count += 1
        else:
            # Freeze EVERYTHING else - no exceptions
            param.requires_grad = False
            frozen_param_count += 1
            
    # Double check - count actual trainable parameters
    actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    actual_total = sum(p.numel() for p in model.parameters())
    
    print(f"LoRA strict freeze: {lora_param_count} LoRA params trainable, {frozen_param_count} others frozen")
    print(f"Total trainable parameters: {actual_trainable:,} / {actual_total:,} ({actual_trainable/actual_total*100:.3f}%)")
    
    # Safety check - adjust threshold based on LoRA rank
    # For rank=16, expect ~2M parameters for decoder attention only
    expected_max = lora_param_count * 50000  # Rough estimate: each LoRA param can have up to 50k weights
    
    if actual_trainable > expected_max:
        print(f"❌ WARNING: {actual_trainable:,} trainable parameters exceeds expected max {expected_max:,}")
        print("Check if non-LoRA parameters are accidentally trainable.")
    else:
        print(f"✅ LoRA parameters: {actual_trainable:,} weights for {lora_param_count} LoRA modules")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in the model.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_lora_target_modules_for_cut3r() -> List[str]:
    """
    Get recommended target modules for LoRA adaptation in CUT3R models.
    
    Only targets decoder Self-Attention and Cross-Attention layers:
    - dec_blocks.*.attn.qkv (decoder self-attention QKV)
    - dec_blocks.*.attn.proj (decoder self-attention output)
    - dec_blocks.*.cross_attn.projq (decoder cross-attention query)
    - dec_blocks.*.cross_attn.projk (decoder cross-attention key)
    - dec_blocks.*.cross_attn.projv (decoder cross-attention value)
    - dec_blocks.*.cross_attn.proj (decoder cross-attention output)
    
    Returns:
        List of target module patterns
    """
    return [
        # Decoder Self-Attention layers only
        r"^dec_blocks\.\d+\.attn\.qkv$",         # Decoder self-attention QKV projections
        r"^dec_blocks\.\d+\.attn\.proj$",        # Decoder self-attention output projections
        
        # Decoder Cross-Attention layers only
        r"^dec_blocks\.\d+\.cross_attn\.projq$", # Decoder cross-attention query projections
        r"^dec_blocks\.\d+\.cross_attn\.projk$", # Decoder cross-attention key projections
        r"^dec_blocks\.\d+\.cross_attn\.projv$", # Decoder cross-attention value projections
        r"^dec_blocks\.\d+\.cross_attn\.proj$",  # Decoder cross-attention output projections
    ]


def print_lora_info(model: nn.Module):
    """
    Print information about LoRA layers in the model.
    
    Args:
        model: Model containing LoRA layers
    """
    lora_layers = []
    total_lora_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            lora_params = module.lora_A.numel() + module.lora_B.numel()
            total_lora_params += lora_params
            lora_layers.append({
                'name': name,
                'rank': module.rank,
                'alpha': module.alpha,
                'params': lora_params,
                'original_params': module.linear.weight.numel()
            })
    
    print(f"\n=== LoRA Information ===")
    print(f"Total LoRA layers: {len(lora_layers)}")
    print(f"Total LoRA parameters: {total_lora_params:,}")
    
    if lora_layers:
        print(f"\nLoRA layers:")
        for layer in lora_layers:
            reduction_ratio = layer['params'] / layer['original_params']
            print(f"  {layer['name']}: rank={layer['rank']}, alpha={layer['alpha']}, "
                  f"params={layer['params']:,} (reduction: {reduction_ratio:.4f})")
    
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params:.4f}")
    print("=" * 25)


def apply_lora_to_cut3r_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    freeze_base_model: bool = True,
    **kwargs
) -> nn.Module:
    """
    Apply LoRA to a CUT3R model with recommended settings.
    
    Args:
        model: CUT3R model to adapt
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
        target_modules: Custom target modules (uses default if None)
        freeze_base_model: Whether to freeze non-LoRA parameters
        
    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = get_lora_target_modules_for_cut3r()
    
    print(f"Applying LoRA with rank={rank}, alpha={alpha}, dropout={dropout}")
    print(f"Target modules: {target_modules}")
    
    # Apply LoRA with default patterns
    model = replace_linear_with_lora(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        **kwargs
    )
    
    # Check if any LoRA layers were created
    lora_count = sum(1 for _, module in model.named_modules() if 'LoRA' in str(type(module)))
    
    if lora_count == 0:
        print("⚠️  No LoRA layers were created with default patterns!")
        print("Trying with universal pattern to match all linear layers...")
        
        # Use universal pattern as fallback
        universal_patterns = [r".*"]  # Match all linear layers
        model = replace_linear_with_lora(
            model=model,
            target_modules=universal_patterns,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            **kwargs
        )
        
        # Check again
        lora_count = sum(1 for _, module in model.named_modules() if 'LoRA' in str(type(module)))
        if lora_count > 0:
            print(f"✅ Successfully applied LoRA to {lora_count} layers using universal pattern")
        else:
            print("❌ Failed to apply LoRA even with universal pattern")
    
    # Freeze base model parameters if requested
    if freeze_base_model:
        freeze_non_lora_parameters(model)
    
    # Print information
    print_lora_info(model)
    
    return model 