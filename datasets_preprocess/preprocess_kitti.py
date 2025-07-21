#!/usr/bin/env python3
"""
Preprocess the KITTI dataset for CUT3R training.

This script processes KITTI sequences by:
  - Loading camera intrinsics from calib.txt
  - Loading camera poses from poses/*.txt
  - Loading IMU data from imus/*.mat files
  - Copying RGB images from image_2/
  - Saving the processed images, camera metadata, and IMU data in the format expected by CUT3R

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
import scipy.io as sio

# IMU frequency for KITTI dataset (10Hz)
IMU_FREQ = 10


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


def load_imu_data(imu_path):
    """
    Load IMU data from .mat file.
    
    Args:
        imu_path: Path to IMU .mat file
        
    Returns:
        numpy.ndarray: IMU data array with shape (N, 6) [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        None: If file doesn't exist or can't be loaded
    """
    if not osp.exists(imu_path):
        print(f"Warning: IMU file {imu_path} not found")
        return None
        
    try:
        mat_data = sio.loadmat(imu_path)
        # Extract IMU data - assuming the key is 'imu_data_interp' as in VIFT
        if 'imu_data_interp' in mat_data:
            imu_data = mat_data['imu_data_interp']
        elif 'imu_data' in mat_data:
            imu_data = mat_data['imu_data']
        else:
            # Try to find the first non-metadata key
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if keys:
                imu_data = mat_data[keys[0]]
                print(f"Using IMU data key: {keys[0]}")
            else:
                print(f"Warning: No valid IMU data found in {imu_path}")
                return None
        
        return imu_data.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading IMU data from {imu_path}: {e}")
        return None


def align_imu_with_images(imu_data, num_images):
    """
    Align IMU data with image timestamps based on VIFT's approach.
    
    Args:
        imu_data: IMU data array with shape (N, 6)
        num_images: Number of images in the sequence
        
    Returns:
        list: List of IMU segments, each corresponding to one image
    """
    if imu_data is None:
        return None
        
    imu_segments = []
    
    for i in range(num_images):
        # Calculate IMU segment for this image based on VIFT's method
        # Each image gets IMU_FREQ samples (10Hz IMU for 1Hz images)
        start_idx = i * IMU_FREQ
        end_idx = start_idx + IMU_FREQ
        
        # Handle boundary cases
        if end_idx > len(imu_data):
            # If we don't have enough IMU data, pad with last available sample
            segment = imu_data[start_idx:]
            if len(segment) < IMU_FREQ:
                # Pad with the last sample
                last_sample = segment[-1] if len(segment) > 0 else np.zeros(imu_data.shape[1])
                padding = np.tile(last_sample, (IMU_FREQ - len(segment), 1))
                segment = np.vstack([segment, padding])
        else:
            segment = imu_data[start_idx:end_idx]
            
        imu_segments.append(segment)
    
    return imu_segments


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
    imu_path = osp.join(data_dir, 'imus', f'{seq_id}.mat')  # IMU data
    
    # Create output directories
    output_seq_dir = osp.join(output_dir, f'kitti_{seq_id}')
    output_rgb_dir = osp.join(output_seq_dir, 'rgb')
    output_cam_dir = osp.join(output_seq_dir, 'cam')
    output_imu_dir = osp.join(output_seq_dir, 'imu')  # New IMU directory
    
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_cam_dir, exist_ok=True)
    os.makedirs(output_imu_dir, exist_ok=True)
    
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
    
    # Load calibration, poses, and IMU data
    calib_data = parse_calib_file(calib_path)
    poses = parse_poses_file(poses_path)
    imu_data = load_imu_data(imu_path)  # Load IMU data
    
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
    
    # Align IMU data with images
    imu_segments = None
    if imu_data is not None:
        imu_segments = align_imu_with_images(imu_data, len(image_files))
        print(f"Loaded IMU data with shape {imu_data.shape}, aligned to {len(image_files)} images")
    else:
        print(f"Warning: No IMU data available for sequence {seq_id}")
    
    processed_count = 0
    
    # Process each frame
    for i, (image_path, pose) in enumerate(zip(image_files, poses)):
        try:
            # Get basename for output files
            basename = osp.splitext(osp.basename(image_path))[0]
            
            # Output paths
            out_img_path = osp.join(output_rgb_dir, f'{basename}.png')
            out_cam_path = osp.join(output_cam_dir, f'{basename}.npz')
            out_imu_path = osp.join(output_imu_dir, f'{basename}.npz')
            
            # Skip if already processed
            if osp.exists(out_cam_path) and (imu_segments is None or osp.exists(out_imu_path)):
                processed_count += 1
                continue
            
            # Copy image
            shutil.copy2(image_path, out_img_path)
            
            # Save camera parameters
            np.savez(out_cam_path, 
                    intrinsics=intrinsics, 
                    pose=pose)
            
            # Save IMU data if available
            if imu_segments is not None and i < len(imu_segments):
                np.savez(out_imu_path, 
                        imu_data=imu_segments[i],
                        timestamp=i)  # Simple timestamp based on frame index
            
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
        default="00,01,02,04,06,08,09",
        help="Comma-separated list of sequence IDs to process"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Only allow available data
    availdata = ['00', '01', '02', '04', '06', '08', '09']
    seq_ids = [seq.strip() for seq in args.sequences.split(',') if seq.strip() in availdata]
    if not seq_ids:
        print(f"No valid sequences to process. Available: {availdata}")
        return
    print(f"Processing sequences: {seq_ids}")
    
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