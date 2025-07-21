import os.path as osp
import os
import sys
import numpy as np
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class KITTI_Multi(BaseMultiViewDataset):
    """
    KITTI dataset loader for CUT3R training.
    
    This dataset supports both camera pose supervision and IMU data loading.
    No depth maps are available, so depthmap is set to all ones.
    """
    
    def __init__(self, *args, ROOT, load_imu=True, scenes=None, **kwargs):
        self.ROOT = ROOT
        self.load_imu = load_imu  # Flag to control IMU loading
        self.scenes_filter = scenes  # Filter for specific scenes
        self.video = True  # KITTI is a video dataset
        self.is_metric = False  # Changed to False to match RE10K
        self.max_interval = 128  # Changed to match RE10K
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        """
        Load KITTI dataset structure.
        
        Returns:
            None: Data is stored in instance variables
        """
        # Get all sequence directories
        all_scenes = sorted([
            d for d in os.listdir(self.ROOT) 
            if osp.isdir(osp.join(self.ROOT, d)) and d.startswith('kitti_')
        ])
        
        # Filter scenes if specified
        if self.scenes_filter is not None:
            self.scenes = [scene for scene in all_scenes if scene in self.scenes_filter]
            if len(self.scenes) == 0:
                raise ValueError(f"None of the specified scenes {self.scenes_filter} found in {self.ROOT}. Available scenes: {all_scenes}")
            print(f"Filtered to scenes: {self.scenes} (from {len(all_scenes)} available)")
        else:
            self.scenes = all_scenes
        
        if len(self.scenes) == 0:
            raise ValueError(f"No KITTI sequences found in {self.ROOT}")
        
        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        
        j = 0
        for scene in self.scenes:
            scene_dir = osp.join(self.ROOT, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            cam_dir = osp.join(scene_dir, "cam")
            imu_dir = osp.join(scene_dir, "imu")  # IMU directory
            
            # Check required directories
            required_dirs = [rgb_dir, cam_dir]
            if self.load_imu:
                required_dirs.append(imu_dir)
                
            if not all(osp.exists(d) for d in required_dirs):
                missing_dirs = [d for d in required_dirs if not osp.exists(d)]
                print(f"Warning: Missing directories for scene {scene}: {missing_dirs}")
                continue
            
            # Get all image files in this sequence
            img_files = sorted([
                osp.splitext(f)[0] for f in os.listdir(rgb_dir)
                if f.endswith('.png')
            ])
            
            if len(img_files) == 0:
                continue
            
            # If loading IMU, check that we have corresponding IMU files
            if self.load_imu:
                imu_files = [f for f in img_files if osp.exists(osp.join(imu_dir, f + ".npz"))]
                if len(imu_files) != len(img_files):
                    print(f"Warning: Scene {scene} has {len(img_files)} images but {len(imu_files)} IMU files")
                    # Use only files that have both image and IMU data
                    img_files = imu_files
                    
            if len(img_files) == 0:
                continue
            
            # Store sequence info
            scenes.extend([j] * len(img_files))
            sceneids.append(j)
            images.extend(img_files)
            start_img_ids.append(offset)
            scene_img_list.append(img_files)
            offset += len(img_files)
            j += 1
        
        self.scenes = [self.scenes[i] for i in sceneids]
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        
        imu_status = "with IMU" if self.load_imu else "without IMU"
        print(f"Loaded {len(self.scenes)} KITTI sequences {imu_status} with {len(self.start_img_ids)} total samples")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        """
        Get multiple views for a given index.
        
        Args:
            idx: Index into the dataset
            resolution: Target resolution (height, width)
            rng: Random number generator
            num_views: Number of views to return
            
        Returns:
            list: List of view dictionaries
        """
        invalid_seq = True
        max_attempts = 10  # 防止无限循环
        
        while invalid_seq and max_attempts > 0:
            max_attempts -= 1
            views = []
            
            # Get sequence info
            scene_id = self.sceneids[idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            cam_dir = osp.join(scene_dir, "cam")
            imu_dir = osp.join(scene_dir, "imu")  # IMU directory
            
            # Get image indices for this sequence
            img_list = self.scene_img_list[scene_id]
            if len(img_list) < num_views:
                # 如果当前序列图像不足，尝试下一个序列
                idx = rng.integers(0, len(self.start_img_ids))
                continue
            
            # Select frames with interval
            if self.video:
                start_idx = rng.integers(0, len(img_list) - num_views + 1)
                image_idxs = list(range(start_idx, start_idx + num_views))
                ordered_video = True
            else:
                image_idxs = rng.choice(len(img_list), size=num_views, replace=False)
                ordered_video = False
            
            # 尝试加载所有视图
            for v, view_idx in enumerate(image_idxs):
                try:
                    basename = img_list[view_idx]
                    
                    # Load RGB image
                    rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
                    
                    # Load camera parameters
                    cam = np.load(osp.join(cam_dir, basename + ".npz"))
                    camera_pose = cam["pose"]  # 4x4 matrix
                    intrinsics = cam["intrinsics"]
                    
                    # Load IMU data if requested
                    imu_data = None
                    if self.load_imu:
                        imu_path = osp.join(imu_dir, basename + ".npz")
                        if osp.exists(imu_path):
                            imu_file = np.load(imu_path)
                            imu_data = imu_file["imu_data"]  # Shape: (IMU_FREQ, 6)
                        else:
                            print(f"Warning: IMU file not found: {imu_path}")
                            # Skip this view if IMU is required but not available
                            views = []
                            idx = rng.integers(0, len(self.start_img_ids))
                            break
                    
                    # Create dummy depthmap (all ones since we only have camera pose supervision)
                    depthmap = np.ones_like(rgb_image[..., 0], dtype=np.float32)
                    
                    # Crop and resize if necessary
                    rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                        rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                    )
                    
                    # Note: Keep rgb_image as PIL Image - it will be converted later by transform
                    # Only check/fix depthmap dimensions if needed
                    if hasattr(depthmap, 'ndim') and depthmap.ndim == 3 and depthmap.shape[0] == 1:
                        depthmap = depthmap.squeeze(0)
                    
                    view_dict = dict(
                        img=rgb_image,
                        depthmap=depthmap,
                        camera_pose=camera_pose.astype(np.float32),  # Keep as 4x4 matrix
                        camera_intrinsics=intrinsics,
                        dataset="KITTI",
                        label=self.scenes[scene_id] + "_" + basename,
                        instance=osp.join(rgb_dir, basename + ".png"),
                        is_metric=self.is_metric,
                        is_video=ordered_video,
                        quantile=np.array(0.98, dtype=np.float32),  # Changed to match RE10K
                        img_mask=True,  # Changed to match RE10K
                        ray_mask=False,  # Changed to match RE10K
                        camera_only=True,  # Only camera pose supervision
                        depth_only=False,
                        single_view=False,
                        reset=False,
                    )
                    
                    # Add IMU data to view dictionary if available
                    if self.load_imu and imu_data is not None:
                        # Remove batch dimension if present
                        if hasattr(imu_data, 'ndim') and imu_data.ndim == 3 and imu_data.shape[0] == 1:
                            imu_data = imu_data.squeeze(0)
                        view_dict['imu'] = imu_data.astype(np.float32)
                    
                    views.append(view_dict)
                    
                except Exception as e:
                    print(f"Error loading view {v} for scene {self.scenes[scene_id]}: {e}")
                    # 如果加载失败，清空views并尝试下一个序列
                    views = []
                    idx = rng.integers(0, len(self.start_img_ids))
                    break
            
            # 检查是否成功加载了所有视图
            if len(views) == num_views:
                invalid_seq = False
        return views
