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
    
    This dataset only has camera pose supervision (camera_only=True).
    No depth maps are available, so depthmap is set to all ones.
    """
    
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
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
        self.scenes = sorted([
            d for d in os.listdir(self.ROOT) 
            if osp.isdir(osp.join(self.ROOT, d)) and d.startswith('kitti_')
        ])
        
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
            
            if not osp.exists(rgb_dir) or not osp.exists(cam_dir):
                continue
            
            # Get all image files in this sequence
            img_files = sorted([
                osp.splitext(f)[0] for f in os.listdir(rgb_dir)
                if f.endswith('.png')
            ])
            
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
        
        print(f"Loaded {len(self.scenes)} KITTI sequences with {len(self.start_img_ids)} total samples")

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
                    
                    # Create dummy depthmap (all ones since we only have camera pose supervision)
                    depthmap = np.ones_like(rgb_image[..., 0], dtype=np.float32)
                    
                    # Crop and resize if necessary
                    rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                        rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                    )
                    
                    views.append(
                        dict(
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
                    )
                    
                except Exception as e:
                    print(f"Error loading view {v} for scene {self.scenes[scene_id]}: {e}")
                    # 如果加载失败，清空views并尝试下一个序列
                    views = []
                    idx = rng.integers(0, len(self.start_img_ids))
                    break
            
            # 检查是否成功加载了所有视图
            if len(views) == num_views:
                invalid_seq = False
            else:
                # 如果当前序列失败，尝试下一个序列
                idx = rng.integers(0, len(self.start_img_ids))
        
        # 如果所有尝试都失败，返回一个默认的视图列表
        if len(views) != num_views:
            print(f"Warning: Failed to load {num_views} views after multiple attempts, returning empty list")
            return []
        
        return views 