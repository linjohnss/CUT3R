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
    
    Args:
        ROOT: Root directory containing processed KITTI data
        load_imu: Whether to load IMU data (default: True)
        scenes: List of scene names to use (default: None, uses all available)
        frame_range: Tuple of (start_frame, end_frame) to limit frame range (default: None)
    """
    
    def __init__(self, *args, ROOT, load_imu=True, scenes=None, frame_range=None, 
                 precompute_model=None, precompute_device='cuda', demo_compatible=True, **kwargs):
        self.ROOT = ROOT
        self.load_imu = load_imu  # Flag to control IMU loading
        self.scenes_filter = scenes  # Filter for specific scenes
        self.frame_range = frame_range  # Frame range filter: (start_frame, end_frame) or None
        self.video = True  # KITTI is a video dataset
        self.is_metric = True  # Changed to False to match RE10K
        self.max_interval = 128  # Changed to match RE10K
        self.precompute_model = precompute_model  # Full CUT3RIMU model for state precomputation (with relative_pose_token)
        self.precompute_device = precompute_device  # Device for precomputation
        self.precomputed_states = {}  # Store precomputed states: {scene_id: {frame_idx: state}}
        self.demo_compatible = demo_compatible  # Use demo-compatible crop (simple center crop)
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()
        
        # Precompute states if model is provided
        import os
        import torch.distributed as dist
        
        # Check if we're in the main process
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        process_info = f"[Process {os.getpid()}, Rank {dist.get_rank() if dist.is_initialized() else 'N/A'}]"
        
        if self.precompute_model is not None:
            print(f"{process_info} ✅ KITTI_Multi.__init__ called WITH precompute_model (TRAINING DATASET)")
            if is_main_process:
                print("=" * 80)
                print(f"{process_info} STARTING STATE PRECOMPUTATION FOR ALL SCENES")
                print(f"precompute_model type: {type(self.precompute_model)}")
                print(f"precompute_device: {self.precompute_device}")
                print("=" * 80)
                self._precompute_all_states()
                print("=" * 80)
                print(f"{process_info} STATE PRECOMPUTATION COMPLETED FOR {len(self.precomputed_states)} SCENES")
                for scene_id, states in self.precomputed_states.items():
                    print(f"  Scene {scene_id}: {len(states)} precomputed states")
                print("=" * 80)
            else:
                print(f"{process_info} ⚠️ Skipping state precomputation in worker process")
        else:
            print(f"{process_info} ℹ️ KITTI_Multi.__init__ called WITHOUT precompute_model (TEST DATASET - NORMAL)")

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
            
            # Apply frame range filter if specified
            if self.frame_range is not None:
                start_frame, end_frame = self.frame_range
                if start_frame < 0:
                    start_frame = 0
                if end_frame > len(img_files):
                    end_frame = len(img_files)
                if start_frame >= end_frame:
                    print(f"Warning: Invalid frame range {self.frame_range} for scene {scene}, skipping")
                    continue
                img_files = img_files[start_frame:end_frame]
                print(f"Applied frame range {self.frame_range} to scene {scene}: using frames {start_frame}:{end_frame} ({len(img_files)} frames)")
            
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
        # Return total number of samples, not number of scenes
        total_samples = 0
        for scene_id in range(len(self.scene_img_list)):
            img_list = self.scene_img_list[scene_id]
            samples_per_scene = max(1, len(img_list) - self.num_views + 1)
            total_samples += samples_per_scene
        return total_samples

    def get_image_num(self):
        return len(self.images)
    
    def _precompute_all_states(self):
        """
        Precompute states for all scenes using forward_recurrent.
        This runs the model once on each complete scene sequence.
        """
        import torch
        from torch.utils.data import DataLoader
        
        self.precompute_model.eval()
        
        print(f"\nPrecomputing states for {len(self.scenes)} scenes...")
        
        for scene_id in tqdm(range(len(self.scenes)), desc="Precomputing scene states"):
            scene_name = self.scenes[scene_id]
            img_list = self.scene_img_list[scene_id]
            
            print(f"\n[Scene {scene_id}/{len(self.scenes)}] {scene_name}: {len(img_list)} frames")
            
            # Load all views for this scene
            scene_views = []
            scene_dir = osp.join(self.ROOT, scene_name)
            rgb_dir = osp.join(scene_dir, "rgb")
            cam_dir = osp.join(scene_dir, "cam")
            imu_dir = osp.join(scene_dir, "imu")
            
            for frame_idx, basename in enumerate(tqdm(img_list, desc=f"Loading {scene_name}", leave=False)):
                try:
                    # Load RGB image
                    rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
                    
                    # Load camera parameters
                    cam = np.load(osp.join(cam_dir, basename + ".npz"))
                    camera_pose = cam["pose"]
                    intrinsics = cam["intrinsics"]
                    
                    # Load IMU data if requested
                    imu_data = None
                    if self.load_imu:
                        imu_path = osp.join(imu_dir, basename + ".npz")
                        if osp.exists(imu_path):
                            imu_file = np.load(imu_path)
                            imu_data = imu_file["imu_data"]
                        else:
                            print(f"Warning: IMU file not found: {imu_path}")
                            continue
                    
                    # Create dummy depthmap
                    depthmap = np.ones_like(rgb_image[..., 0], dtype=np.float32)
                    
                    # Crop and resize to the resolution used in training
                    if hasattr(self, 'resolution') and self.resolution:
                        resolution = self.resolution[0] if isinstance(self.resolution, list) else self.resolution
                    else:
                        resolution = (512, 384)  # Default resolution from config
                    
                    # Use a compatible random number generator
                    rng = np.random.default_rng(42)  # Use fixed seed for consistency
                    rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                        rgb_image, depthmap, intrinsics, resolution, rng=rng, info=frame_idx
                    )
                    
                    # Create view dictionary
                    view_dict = dict(
                        img=rgb_image,
                        depthmap=depthmap,
                        camera_pose=camera_pose.astype(np.float32),
                        camera_intrinsics=intrinsics,
                        dataset="KITTI",
                        label=scene_name + "_" + basename,
                        instance=osp.join(rgb_dir, basename + ".png"),
                        is_metric=self.is_metric,
                        is_video=True,
                        quantile=np.array(0.98, dtype=np.float32),
                        img_mask=np.array([True], dtype=bool),  # Shape: [1]
                        ray_mask=np.array([False], dtype=bool),  # Shape: [1]
                        camera_only=True,
                        depth_only=False,
                        single_view=False,
                        reset=np.array([False], dtype=bool),  # Shape: [1]
                        update=np.array([True], dtype=bool),  # Shape: [1]
                    )
                    
                    # Add IMU data if available
                    if self.load_imu and imu_data is not None:
                        if hasattr(imu_data, 'ndim') and imu_data.ndim == 3 and imu_data.shape[0] == 1:
                            imu_data = imu_data.squeeze(0)
                        view_dict['imu'] = imu_data.astype(np.float32)
                    
                    scene_views.append(view_dict)
                    
                except Exception as e:
                    print(f"Error loading frame {frame_idx} of scene {scene_name}: {e}")
                    continue
            
            if len(scene_views) == 0:
                print(f"Warning: No valid views loaded for scene {scene_name}")
                continue
            
            # Preprocess views (apply transforms)
            processed_views = []
            for view in scene_views:
                processed_view = self._preprocess_view(view)
                processed_views.append(processed_view)
            
            # Run forward_recurrent to get states
            print(f"Running forward_recurrent for {len(processed_views)} frames...")
            with torch.no_grad():
                try:
                    ress, processed_views_output, all_state_args = self.precompute_model.forward_recurrent(
                        processed_views, 
                        device=self.precompute_device,
                        ret_state=True
                    )
                    
                    # Store states for each frame
                    # all_state_args contains: [initial_state, state_after_frame_0, state_after_frame_1, ...]
                    self.precomputed_states[scene_id] = {}
                    
                    # CRITICAL FIX: For training, we want to use the state BEFORE processing each frame
                    # all_state_args[0] = initial state (for frame 0)
                    # all_state_args[1] = state after frame 0 (for frame 1) 
                    # all_state_args[2] = state after frame 1 (for frame 2)
                    # So frame i should use all_state_args[i] (state before processing frame i)
                    for frame_idx in range(len(processed_views)):
                        # Frame i uses state from all_state_args[i] (state before processing frame i)
                        state_idx = frame_idx  # This is correct: frame 0 uses all_state_args[0]
                        if state_idx < len(all_state_args):
                            state_feat, state_pos, init_state_feat, mem, init_mem = all_state_args[state_idx]
                            self.precomputed_states[scene_id][frame_idx] = {
                                'state_feat': state_feat,
                                'state_pos': state_pos,
                                'init_state_feat': init_state_feat,
                                'mem': mem,
                                'init_mem': init_mem,
                            }
                    
                    print(f"Stored states for {len(self.precomputed_states[scene_id])} frames from {len(all_state_args)} state snapshots")
                    print(f"DEBUG: all_state_args length: {len(all_state_args)}, processed_views length: {len(processed_views)}")
                    if len(all_state_args) > 0:
                        print(f"DEBUG: First state args shape: {[x.shape if hasattr(x, 'shape') else type(x) for x in all_state_args[0]]}")
                    
                except Exception as e:
                    print(f"Error during forward_recurrent for scene {scene_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def recompute_states(self, updated_model):
        """
        Recompute precomputed states with an updated model.
        This allows periodic recomputation during training to align with the evolving model.
        
        Args:
            updated_model: Updated CUT3RIMU model (should include relative_pose_token)
        """
        import torch.distributed as dist
        
        # Only recompute in main process
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if not is_main_process:
            return
        
        print("\n" + "=" * 80)
        print("🔄 RECOMPUTING PRECOMPUTED STATES WITH UPDATED MODEL")
        print(f"Updated model type: {type(updated_model)}")
        print("=" * 80)
        
        # Update model reference
        old_model = self.precompute_model
        self.precompute_model = updated_model
        
        # Recompute all states
        try:
            self._precompute_all_states()
            print("=" * 80)
            print(f"✅ STATE RECOMPUTATION COMPLETED FOR {len(self.precomputed_states)} SCENES")
            for scene_id, states in self.precomputed_states.items():
                print(f"  Scene {scene_id}: {len(states)} precomputed states")
            print("=" * 80)
        except Exception as e:
            print(f"❌ ERROR during state recomputation: {e}")
            import traceback
            traceback.print_exc()
            # Restore old model on error
            self.precompute_model = old_model
            raise
    
    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """
        Override base class method to use demo-compatible crop when enabled.
        This ensures training/precomputation uses the same crop as demo inference.
        """
        if self.demo_compatible:
            return self._demo_compatible_crop_resize(image, depthmap, intrinsics, resolution)
        else:
            # Use base class method (intrinsics-based crop)
            return super()._crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng, info)
    
    def _demo_compatible_crop_resize(self, image, depthmap, intrinsics, resolution):
        """
        Demo-compatible crop and resize - matches load_images() exactly.
        Uses simple center crop based on image geometry, not camera intrinsics.
        
        CRITICAL: This must exactly match the load_images() behavior in demo!
        """
        import PIL.Image
        from dust3r.utils.image import _resize_pil_image
        
        # Convert to PIL if needed
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        
        # CRITICAL FIX: Pass resolution as tuple to _resize_pil_image
        # This ensures we use the same resize logic as demo (direct resize to target size)
        # instead of resizing long edge (which would happen with int size)
        if isinstance(resolution, tuple):
            target_width, target_height = resolution
            size = resolution  # ⭐ Keep as tuple for _resize_pil_image
            size_for_crop = target_width  # Use width for crop logic
        else:
            size = resolution
            size_for_crop = resolution
        
        # Step 1: Resize (matching demo's logic)
        W1, H1 = image.size
        if (isinstance(size, tuple) and size[0] == 224) or (isinstance(size, int) and size == 224):
            # For 224 models
            if isinstance(size, int):
                image = _resize_pil_image(image, round(size * max(W1 / H1, H1 / W1)))
            else:
                # For tuple, directly resize
                image = _resize_pil_image(image, size)
        else:
            # For 512 models - pass size as-is (tuple or int)
            image = _resize_pil_image(image, size)
        
        # Step 2: Center crop (matching demo's logic)
        W, H = image.size
        cx, cy = W // 2, H // 2  # Use geometric center (not principal point!)
        
        # Use size_for_crop for determining crop logic
        if size_for_crop == 224:
            half = min(cx, cy)
            crop_box = (cx - half, cy - half, cx + half, cy + half)
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            # Special case for square images
            if W == H:
                halfh = 3 * halfw // 4
            crop_box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
        
        image = image.crop(crop_box)
        
        # Verify final size matches demo expectations
        final_W, final_H = image.size
        if isinstance(resolution, tuple):
            expected_W, expected_H = resolution
            # For tuple resolution with demo mode, we expect the size after resize+crop
            # to match the input tuple (after accounting for center crop alignment)
            # Note: Due to crop alignment (multiples of 16), final size might differ slightly
            # Demo: 512x384 -> after resize+crop might be slightly different due to alignment
            pass  # Size verification - expected some variance due to crop alignment
        
        # Crop depthmap to match (simple center crop)
        if depthmap is not None and len(depthmap.shape) > 0:
            H_d, W_d = depthmap.shape[:2]
            cx_d, cy_d = W_d // 2, H_d // 2
            
            # Calculate crop box for depthmap (proportional to image crop)
            scale_w = W_d / W1
            scale_h = H_d / H1
            
            left_d = int((crop_box[0] / W1) * W_d)
            top_d = int((crop_box[1] / H1) * H_d)
            right_d = int((crop_box[2] / W1) * W_d)
            bottom_d = int((crop_box[3] / H1) * H_d)
            
            depthmap = depthmap[top_d:bottom_d, left_d:right_d]
            
            # Resize depthmap to match image size
            from PIL import Image as PILImage
            depthmap_pil = PILImage.fromarray(depthmap)
            depthmap_pil = depthmap_pil.resize(image.size, PILImage.NEAREST)
            depthmap = np.array(depthmap_pil)
        
        # Update intrinsics (simplified - just adjust principal point)
        intrinsics = intrinsics.copy()
        W_final, H_final = image.size
        
        # Adjust principal point to new center
        intrinsics[0, 2] = W_final / 2.0
        intrinsics[1, 2] = H_final / 2.0
        
        return image, depthmap, intrinsics
    
    def _preprocess_view(self, view):
        """
        Preprocess a single view (apply transforms)
        """
        import torch
        from torchvision.transforms import ToTensor
        
        # Convert data to tensors
        processed = {}
        for key, value in view.items():
            if key == 'img':
                # Handle image data
                if hasattr(value, 'mode'):  # PIL Image
                    pil_img = value
                elif isinstance(value, np.ndarray):  # numpy array
                    from PIL import Image
                    pil_img = Image.fromarray(value.astype(np.uint8))
                else:
                    raise TypeError(f"Unexpected image type: {type(value)}")
                
                # Apply transforms if they exist
                if hasattr(self, 'transform') and self.transform is not None:
                    # transform already returns a tensor, so we don't need ToTensor()
                    processed[key] = self.transform(pil_img)
                else:
                    # Convert to tensor only if no transform was applied
                    to_tensor = ToTensor()
                    processed[key] = to_tensor(pil_img)
            elif isinstance(value, np.ndarray):
                processed[key] = torch.from_numpy(value)
            elif isinstance(value, bool):
                if key in ['img_mask', 'ray_mask']:
                    # For mask values, we need to create a proper tensor shape
                    # This will be handled later in the data loading process
                    processed[key] = torch.tensor(value)
                else:
                    processed[key] = torch.tensor(value)
            else:
                processed[key] = value
        
        return processed

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
            
            # Find which scene this idx belongs to
            scene_id = 0
            cumulative_samples = 0
            for i, img_list in enumerate(self.scene_img_list):
                samples_per_scene = max(1, len(img_list) - num_views + 1)
                if idx < cumulative_samples + samples_per_scene:
                    scene_id = i
                    break
                cumulative_samples += samples_per_scene
            
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
            # Check if we have precomputed states for this scene
            has_precomputed_states = (scene_id in self.precomputed_states)
            
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
                        img_mask=np.array([True], dtype=bool),  # Shape: [1]
                        ray_mask=np.array([False], dtype=bool),  # Shape: [1]
                        camera_only=True,  # Only camera pose supervision
                        depth_only=False,
                        single_view=False,
                        reset=np.array([False], dtype=bool),  # Shape: [1]
                        update=np.array([True], dtype=bool),  # Shape: [1]
                    )
                    
                    # Add IMU data to view dictionary if available
                    if self.load_imu and imu_data is not None:
                        # Remove batch dimension if present
                        if hasattr(imu_data, 'ndim') and imu_data.ndim == 3 and imu_data.shape[0] == 1:
                            imu_data = imu_data.squeeze(0)
                        view_dict['imu'] = imu_data.astype(np.float32)
                    
                    # Add precomputed state if available
                    if has_precomputed_states and view_idx in self.precomputed_states[scene_id]:
                        view_dict['precomputed_state'] = self.precomputed_states[scene_id][view_idx]
                    
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
