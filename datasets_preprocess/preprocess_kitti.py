#!/usr/bin/env python3
"""
Preprocess the KITTI dataset for CUT3R training.

This script processes KITTI sequences by:
  - Loading camera intrinsics from calib.txt
  - Loading camera poses from poses/*.txt
  - Copying RGB images from image_2/
  - Saving the processed images and camera metadata in the format expected by CUT3R

Usage:
  python preprocess_kitti.py --data_dir /project2/larg3r/VIFT/data/kitti_data \
                             --output_dir /path/to/processed_kitti
"""

import os
import os.path as osp
import argparse
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from glob import glob


def parse_calib_file(calib_path):
    """
    Parse KITTI calibration file to extract camera intrinsics.
    
    Args:
        calib_path: Path to calib.txt file
        
    Returns:
        dict: Dictionary containing camera intrinsics for each camera
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    calib_data = {}
    for line in lines:
        if line.startswith('P'):
            camera_id = line.split(':')[0]
            values = np.array([float(x) for x in line.split(':')[1].strip().split()])
            
            # P matrix is 3x4, extract 3x3 intrinsics matrix
            P = values.reshape(3, 4)
            K = P[:3, :3]  # Extract 3x3 intrinsics matrix
            
            calib_data[camera_id] = K.astype(np.float32)
    
    return calib_data


def parse_poses_file(poses_path):
    """
    Parse KITTI poses file to extract camera poses.
    
    Args:
        poses_path: Path to poses/*.txt file
        
    Returns:
        list: List of 4x4 pose matrices (camera to world)
    """
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = np.array([float(x) for x in line.strip().split()])
            pose = values.reshape(3, 4)
                        
            # Convert to 4x4 homogeneous matrix
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :3] = pose[:3, :3]  # 旋转部分 (3x3)
            pose_4x4[:3, 3] = pose[:3, 3]    # 平移部分 (3x1)
            # 最后一行保持 [0,0,0,1]
            
            # Convert from world-to-camera to camera-to-world
            # pose_4x4 = np.linalg.inv(pose_4x4)
            
            poses.append(pose_4x4.astype(np.float32))
    
    return poses


def process_sequence(seq_id, data_dir, output_dir):
    """
    Process a single KITTI sequence.
    
    Args:
        seq_id: Sequence ID (e.g., '00', '01', etc.)
        data_dir: Root directory of KITTI dataset
        output_dir: Output directory for processed data
        
    Returns:
        int: Number of processed frames
    """
    # Define paths
    seq_dir = osp.join(data_dir, 'sequences', seq_id)
    poses_path = osp.join(data_dir, 'poses', f'{seq_id}.txt')
    calib_path = osp.join(seq_dir, 'calib.txt')
    image_dir = osp.join(seq_dir, 'image_2')  # Left camera
    
    # Create output directories
    output_seq_dir = osp.join(output_dir, f'kitti_{seq_id}')
    output_rgb_dir = osp.join(output_seq_dir, 'rgb')
    output_cam_dir = osp.join(output_seq_dir, 'cam')
    
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_cam_dir, exist_ok=True)
    
    # Check if files exist
    if not osp.exists(poses_path):
        print(f"Warning: Poses file {poses_path} not found, skipping sequence {seq_id}")
        return 0
    
    if not osp.exists(calib_path):
        print(f"Warning: Calib file {calib_path} not found, skipping sequence {seq_id}")
        return 0
    
    if not osp.exists(image_dir):
        print(f"Warning: Image directory {image_dir} not found, skipping sequence {seq_id}")
        return 0
    
    # Load calibration and poses
    calib_data = parse_calib_file(calib_path)
    poses = parse_poses_file(poses_path)
    
    # Use P2 (left camera) intrinsics
    if 'P2' not in calib_data:
        print(f"Warning: P2 calibration not found in {calib_path}, skipping sequence {seq_id}")
        return 0
    
    intrinsics = calib_data['P2']
    
    # Get list of image files
    image_files = sorted(glob(osp.join(image_dir, '*.png')))
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {image_dir}, skipping sequence {seq_id}")
        return 0
    
    # Check if we have enough poses
    if len(poses) < len(image_files):
        print(f"Warning: Number of poses ({len(poses)}) < number of images ({len(image_files)}) for sequence {seq_id}")
        # Truncate to match poses
        image_files = image_files[:len(poses)]
    elif len(poses) > len(image_files):
        print(f"Warning: Number of poses ({len(poses)}) > number of images ({len(image_files)}) for sequence {seq_id}")
        # Truncate poses to match images
        poses = poses[:len(image_files)]
    
    processed_count = 0
    
    # Process each frame
    for i, (image_path, pose) in enumerate(zip(image_files, poses)):
        try:
            # Get basename for output files
            basename = osp.splitext(osp.basename(image_path))[0]
            
            # Output paths
            out_img_path = osp.join(output_rgb_dir, f'{basename}.png')
            out_cam_path = osp.join(output_cam_dir, f'{basename}.npz')
            
            # Skip if already processed
            if osp.exists(out_cam_path):
                processed_count += 1
                continue
            
            # Copy image
            shutil.copy2(image_path, out_img_path)
            
            # Save camera parameters
            np.savez(out_cam_path, 
                    intrinsics=intrinsics, 
                    pose=pose)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing frame {i} in sequence {seq_id}: {e}")
            continue
    
    print(f"Processed {processed_count} frames for sequence {seq_id}")
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess KITTI dataset for CUT3R training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/project2/larg3r/VIFT/data/kitti_data",
        help="Directory containing KITTI dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_kitti",
        help="Directory where processed data will be saved"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21",
        help="Comma-separated list of sequence IDs to process"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse sequence IDs
    seq_ids = [seq.strip() for seq in args.sequences.split(',')]
    
    total_processed = 0
    
    # Process each sequence
    for seq_id in tqdm(seq_ids, desc="Processing sequences"):
        try:
            processed_count = process_sequence(seq_id, args.data_dir, args.output_dir)
            total_processed += processed_count
        except Exception as e:
            print(f"Error processing sequence {seq_id}: {e}")
            continue
    
    print(f"Total processed frames: {total_processed}")
    print(f"Processed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 