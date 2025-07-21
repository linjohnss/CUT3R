#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script with LoRA + IMU Support

This script performs inference using the ARCroco3DStereo model with LoRA fine-tuning
and IMU encoder enhancement, visualizing the resulting 3D point clouds with the PointCloudViewer.

Usage:
    python demo_imu.py [--model_path MODEL_PATH] [--lora_path LORA_PATH] [--imu_path IMU_PATH] 
                        [--seq_path SEQ_PATH] [--size IMG_SIZE] [--device DEVICE] 
                        [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo_imu.py --model_path src/cut3r_512_dpt_4_64.pth \
        --lora_path src/checkpoints/cut3r_imu_lora/lora_weights_final.pth \
        --imu_path src/checkpoints/cut3r_imu_lora/imu_weights_final.pth \
        --seq_path /project2/larg3r/dataset/dust3r_data/processed_kitti/kitti_00/rgb \
        --imu_data_path /project2/larg3r/dataset/dust3r_data/processed_kitti/kitti_00/imu \
        --device cuda --size 512
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo with LoRA + IMU."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="",
        help="Path to the LoRA weights file (.pth). If not provided, will run without LoRA.",
    )
    parser.add_argument(
        "--imu_path",
        type=str,
        default="",
        help="Path to the IMU weights file (.pth). If not provided, will run without IMU enhancement.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--imu_data_path",
        type=str,
        default="",
        help="Path to the directory containing the IMU data sequence. If not provided, will use seq_path/../imu",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )

    return parser.parse_args()


def load_imu_data(imu_dir, img_basenames):
    """
    Load IMU data for corresponding image files.
    
    Args:
        imu_dir (str): Directory containing IMU data files
        img_basenames (list): List of image basenames (without extension)
        
    Returns:
        list: List of IMU data arrays, one for each image
    """
    imu_data_list = []
    
    for basename in img_basenames:
        imu_file_path = os.path.join(imu_dir, basename + ".npz")
        
        if os.path.exists(imu_file_path):
            try:
                imu_file = np.load(imu_file_path)
                imu_data = imu_file["imu_data"]  # Shape: (IMU_FREQ, 6)
                
                # Ensure IMU data has the expected shape (10, 6)
                if imu_data.shape[0] > 10:
                    # Take the first 10 frames
                    imu_data = imu_data[:10]
                elif imu_data.shape[0] < 10:
                    # Pad with zeros if less than 10 frames
                    padding = np.zeros((10 - imu_data.shape[0], 6), dtype=imu_data.dtype)
                    imu_data = np.vstack([imu_data, padding])
                
                imu_data_list.append(imu_data.astype(np.float32))
                
            except Exception as e:
                print(f"Warning: Failed to load IMU data for {basename}: {e}")
                # Create zero IMU data as fallback
                imu_data_list.append(np.zeros((10, 6), dtype=np.float32))
        else:
            print(f"Warning: IMU file not found: {imu_file_path}")
            # Create zero IMU data as fallback
            imu_data_list.append(np.zeros((10, 6), dtype=np.float32))
    
    return imu_data_list


def prepare_input(
    img_paths, img_mask, size, imu_data_list=None, revisit=1, update=True
):
    """
    Prepare input views for inference from a list of image paths with IMU data.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        imu_data_list (list, optional): List of IMU data arrays.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    for i in range(len(images)):
        view = {
            "img": images[i]["img"],
            "ray_map": torch.full(
                (
                    images[i]["img"].shape[0],
                    6,
                    images[i]["img"].shape[-2],
                    images[i]["img"].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                0
            ),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
    
        # Add IMU data if available
        if imu_data_list is not None and i < len(imu_data_list):
            view["imu"] = torch.from_numpy(imu_data_list[i])
    
        views.append(view)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )

    return pts3ds_other, colors, conf_other, cam_dict


def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def load_lora_weights(model, lora_path, device):
    """
    Load LoRA weights into the model.
    
    Args:
        model: The model to load LoRA weights into
        lora_path: Path to the LoRA weights file
        device: Device to load weights on
    """
    from src.lora_utils import (
        load_lora_state_dict, print_lora_info, 
        apply_lora_to_cut3r_model, get_lora_target_modules_for_cut3r
    )
    
    print(f"Loading LoRA weights from {lora_path}...")
    
    # First, check if the checkpoint contains LoRA configuration
    checkpoint = torch.load(lora_path, map_location=device)
    
    # Check if this is a full checkpoint with LoRA config or just LoRA weights
    if isinstance(checkpoint, dict) and 'lora_weights' in checkpoint:
        # This is a full checkpoint with LoRA config
        lora_state_dict = checkpoint['lora_weights']
        model_config = checkpoint.get('model_config', None)
        
        # Extract LoRA config from model config if available
        if model_config and hasattr(model_config, 'lora_rank'):
            rank = model_config.lora_rank
            alpha = model_config.lora_alpha
            dropout = model_config.lora_dropout
            print(f"Using LoRA config from checkpoint: rank={rank}, alpha={alpha}, dropout={dropout}")
        else:
            # Use default values
            rank = 16
            alpha = 16.0
            dropout = 0.0
            print(f"Using default LoRA config: rank={rank}, alpha={alpha}, dropout={dropout}")
    else:
        # This is just LoRA weights
        lora_state_dict = checkpoint
        # Use default LoRA config
        rank = 16
        alpha = 16.0
        dropout = 0.0
        print(f"Using default LoRA config: rank={rank}, alpha={alpha}, dropout={dropout}")
    
    # Apply LoRA to the model first
    print("Applying LoRA to model...")
    target_modules = get_lora_target_modules_for_cut3r()
    apply_lora_to_cut3r_model(
        model=model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        freeze_base_model=True  # Freeze non-LoRA parameters
    )
    
    # Now load the LoRA weights
    print("Loading LoRA weights into model...")
    load_lora_state_dict(model, lora_state_dict, strict=False)
    
    # Print LoRA information
    print_lora_info(model)
    
    # Ensure model is in eval mode
    model.eval()
    
    print("LoRA weights loaded successfully!")


def load_imu_weights(model, imu_path, device):
    """
    Load IMU weights into the model.
    
    Args:
        model: The model to load IMU weights into
        imu_path: Path to the IMU weights file
        device: Device to load weights on
    """
    from src.dust3r.models.imu_encoder import CUT3RIMU
    
    print(f"Loading IMU weights from {imu_path}...")
    
    # Load IMU checkpoint
    imu_checkpoint = torch.load(imu_path, map_location=device)
    
    # Check if this is a CUT3RIMU model
    if not hasattr(model, 'imu_encoder'):
        print("Converting model to IMU-enhanced model...")
        # Create IMU config (use default values)
        imu_config = {
            'input_dim': 6,
            'seq_len': 10,
            'dropout': 0.1,
            'fusion_method': 'add',
            'num_heads': 8
        }
        model = CUT3RIMU(model, imu_config)
        # Move the entire model to the correct device
        model = model.to(device)
        print("Model converted to IMU-enhanced model")
    
    # Load IMU encoder weights
    if 'imu_encoder' in imu_checkpoint:
        model.imu_encoder.load_state_dict(imu_checkpoint['imu_encoder'])
        print("IMU encoder weights loaded successfully!")
    else:
        print("Warning: No IMU encoder weights found in checkpoint")
    
    # Load IMU projection weights (new architecture)
    if 'imu_proj' in imu_checkpoint:
        print("Warning: Old imu_proj weights found but not used in simplified architecture")
    
    # Load IMU fusion weights (for backward compatibility with old checkpoints)
    if 'imu_fusion' in imu_checkpoint:
        print("Warning: Old imu_fusion weights found but not used in new direct-addition architecture")
    
    # Load learnable weights for pose token fusion
    if 'imu_weight' in imu_checkpoint:
        model.imu_weight.data = imu_checkpoint['imu_weight']
        print("IMU weight parameter loaded successfully!")
    if 'pose_weight' in imu_checkpoint:
        model.pose_weight.data = imu_checkpoint['pose_weight']
        print("Pose weight parameter loaded successfully!")
    
    # Ensure model is in eval mode
    model.eval()
    
    print("IMU weights loaded successfully!")
    return model


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    # Limit images to 200 maximum
    MAX_IMAGES = 200
    if len(img_paths) > MAX_IMAGES:
        print(f"Found {len(img_paths)} images, limiting to {MAX_IMAGES} images.")
        img_paths = img_paths[:MAX_IMAGES]
    else:
        print(f"Found {len(img_paths)} images in {args.seq_path}.")
    
    img_mask = [True] * len(img_paths)

    # Prepare IMU data if IMU path is provided
    imu_data_list = None
    if args.imu_path:  # Only load IMU data if IMU weights are provided
        # Determine IMU data directory
        if args.imu_data_path:
            imu_dir = args.imu_data_path
        else:
            # Use default path: seq_path/../imu
            imu_dir = os.path.join(os.path.dirname(args.seq_path), "..", "imu")
        
        if os.path.exists(imu_dir):
            print(f"Loading IMU data from {imu_dir}...")
            # Extract basenames from image paths
            img_basenames = [os.path.splitext(os.path.basename(path))[0] for path in img_paths]
            imu_data_list = load_imu_data(imu_dir, img_basenames)
            print(f"Loaded IMU data for {len(imu_data_list)} images")
        else:
            print(f"Warning: IMU data directory {imu_dir} not found. Running without IMU data.")
            imu_data_list = [np.zeros((10, 6), dtype=np.float32) for _ in img_paths]

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        imu_data_list=imu_data_list,
        revisit=1,
        update=True,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    
    # Load LoRA weights if provided
    if args.lora_path and os.path.exists(args.lora_path):
        load_lora_weights(model, args.lora_path, device)
    elif args.lora_path:
        print(f"Warning: LoRA path {args.lora_path} does not exist. Running without LoRA.")
    else:
        print("No LoRA path provided. Running with base model.")
    
    # Load IMU weights if provided
    if args.imu_path and os.path.exists(args.imu_path):
        model = load_imu_weights(model, args.imu_path, device)
    elif args.imu_path:
        print(f"Warning: IMU path {args.imu_path} does not exist. Running without IMU enhancement.")
    else:
        print("No IMU path provided. Running without IMU enhancement.")
    
    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, 1, True
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size = args.size
    )
    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main() 