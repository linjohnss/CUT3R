import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, zeros_


class ZeroConv1d(nn.Module):
    """
    Zero convolution layer that initializes with zero weights and bias.
    This allows the IMU encoder to start from zero contribution and gradually learn.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # Mark this conv layer as part of ZeroConv1d
        self.conv._is_zero_conv = True
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights and bias to zero"""
        zeros_(self.conv.weight)
        if self.conv.bias is not None:
            zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)


class IMUEncoder(nn.Module):
    """
    IMU Encoder based on VIFT design for CUT3R integration.
    
    Takes IMU sequences and outputs features compatible with CUT3R's pose token dimensions.
    """
    
    def __init__(self, 
                 input_dim=6,          # IMU 6軸 (acc_x,y,z + gyro_x,y,z)
                 hidden_dims=[64, 128, 256], 
                 output_dim=768,       # 與 CUT3R decoder 維度一致 (pose token dimension)
                 seq_len=10,           # 每個 frame 對應的 IMU 長度
                 dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # 基於 VIFT 的 1D Conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
        
        # 投影到與 pose token 一致的維度 (使用 zero conv 讓 IMU 從零開始學習)
        self.projection = ZeroConv1d(hidden_dims[2] * seq_len, output_dim, kernel_size=1)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following VIFT's initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                # Skip ZeroConv1d as it initializes itself to zero
                # Check if this module is part of ZeroConv1d by checking its parent
                if not hasattr(m, '_is_zero_conv'):
                    kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, imu_data):
        """
        Forward pass of IMU encoder
        
        Args:
            imu_data: (batch_size, seq_len, 10, 6) or (batch_size * seq_len, 10, 6)
            
        Returns:
            imu_features: (batch_size, seq_len, output_dim) or (batch_size * seq_len, output_dim)
        """
        original_shape = imu_data.shape
        
        if len(original_shape) == 4:
            # Input: (batch_size, seq_len, 10, 6)
            batch_size, seq_len = original_shape[:2]
            # Reshape for conv1d: (batch_size * seq_len, 6, 10)
            x = imu_data.view(batch_size * seq_len, original_shape[3], original_shape[2])
            reshape_output = True
        else:
            # Input: (batch_size * seq_len, 10, 6)
            # Reshape for conv1d: (batch_size * seq_len, 6, 10)
            x = imu_data.permute(0, 2, 1)  # (batch_size * seq_len, 6, 10)
            reshape_output = False
        
        # 通過 conv layers
        x = self.conv_layers(x)  # (batch_size * seq_len, 256, 10)
        
        # Flatten and project
        x = x.view(x.size(0), -1)  # (batch_size * seq_len, 256 * 10)
        # Reshape for conv1d: (batch_size * seq_len, 256 * 10, 1)
        x = x.unsqueeze(-1)  # Add channel dimension for conv1d
        x = self.projection(x)     # (batch_size * seq_len, output_dim, 1)
        x = x.squeeze(-1)    # Remove channel dimension: (batch_size * seq_len, output_dim)
        
        if reshape_output:
            # Reshape back: (batch_size, seq_len, output_dim)
            return x.view(batch_size, seq_len, -1)
        else:
            return x


class CUT3RIMU(nn.Module):
    """
    CUT3R model enhanced with IMU encoder
    IMU features are directly added to pose tokens
    """
    
    def __init__(self, cut3r_model, imu_config=None):
        super().__init__()
        
        self.cut3r_model = cut3r_model
        
        # IMU encoder configuration
        imu_config = imu_config or {}
        
        # Get correct embed_dim from CUT3R model config (use decoder dimension for pose token)
        if hasattr(cut3r_model, 'config'):
            # Use decoder embedding dimension since pose token is in decoder space
            embed_dim = getattr(cut3r_model.config, 'dec_embed_dim', 768)
        else:
            # Fallback: try to infer from model parameters
            embed_dim = 768
            print(f"Warning: Could not get dec_embed_dim from cut3r_model.config, using default {embed_dim}")
        
        print(f"Using embed_dim={embed_dim} for IMU encoder (CUT3R decoder/pose token dimension)")
        
        self.imu_encoder = IMUEncoder(
            input_dim=imu_config.get('input_dim', 6),
            output_dim=embed_dim,  # Use CUT3R's decoder dimension (pose token dimension)
            seq_len=imu_config.get('seq_len', 10),
            dropout=imu_config.get('dropout', 0.1)
        )
        
        # 學習權重用於 IMU 特徵與 pose token 的加權相加
        self.imu_weight = nn.Parameter(torch.tensor(0.1))  # IMU 特徵權重，初始化為較小值
        self.pose_weight = nn.Parameter(torch.tensor(0.9))  # pose token 權重
    
    def forward(self, images, imu_data=None, **kwargs):
        # Use our custom _forward_impl that handles IMU processing
        ret_state = kwargs.get('ret_state', False)
        
        if ret_state:
            ress, views, state_args = self._forward_impl(images, ret_state=ret_state)
            # Import the output class
            from dust3r.model import ARCroco3DStereoOutput
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(images, ret_state=ret_state)
            # Import the output class
            from dust3r.model import ARCroco3DStereoOutput
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def _forward_encoder(self, batch, *args, **kwargs):
        # Just call the underlying encoder - IMU processing will be done in decoder
        return self.cut3r_model._forward_encoder(batch, *args, **kwargs)

    def _forward_decoder_step(self, batch, idx, *args, **kwargs):
        # Extract and process IMU data for this specific view (skip first view)
        imu_feat = None
        if idx > 0 and isinstance(batch, (list, tuple)) and idx < len(batch) and isinstance(batch[idx], dict) and 'imu' in batch[idx]:
            imu_data = batch[idx]['imu']  # (seq_len, 6)
            if imu_data.numel() > 0:
                # Process IMU data for this view
                imu_feat = self.imu_encoder(imu_data.unsqueeze(0))  # (1, embed_dim)
                # Ensure we get the right shape - remove all singleton dimensions
                imu_feat = imu_feat.squeeze()  # (embed_dim,)
        
        if imu_feat is not None and hasattr(self.cut3r_model, 'pose_token') and self.cut3r_model.pose_head_flag:
            # Create a modified decoder step that adds IMU to pose token
            return self._modified_decoder_step_with_imu(batch, idx, imu_feat, *args, **kwargs)
        
        # Default behavior if no IMU data or first view
        return self.cut3r_model._forward_decoder_step(batch, idx, *args, **kwargs)
    
    def _modified_decoder_step_with_imu(self, views, i, imu_feat, feat_i, pos_i, shape_i, init_state_feat, init_mem, state_feat, state_pos, mem):
        """Modified decoder step that adds IMU features to pose token"""
        
        if self.cut3r_model.pose_head_flag:
            global_img_feat_i = self.cut3r_model._get_img_level_feat(feat_i)
            
            # Get original pose token behavior
            if i == 0:
                original_pose_feat_i = self.cut3r_model.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                original_pose_feat_i = self.cut3r_model.pose_retriever.inquire(global_img_feat_i, mem)
            
            # Add IMU features to pose token
            # imu_feat shape: (embed_dim,), need to match pose token shape: (batch_size, 1, embed_dim)
            imu_feat_expanded = imu_feat.unsqueeze(0).unsqueeze(0).expand(feat_i.shape[0], 1, -1)
            
            # Weighted addition of pose token and IMU features
            pose_feat_i = original_pose_feat_i + imu_feat_expanded
            
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        
        # Continue with normal decoder rollout
        new_state_feat, dec = self.cut3r_model._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,  # This now contains IMU + pose features
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
        )
        
        # Update memory as normal
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.cut3r_model.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        
        # Generate head input and output
        head_input = [
            dec[0].float(),
            dec[self.cut3r_model.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.cut3r_model.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.cut3r_model.dec_depth].float(),
        ]
        res = self.cut3r_model._downstream_head(head_input, shape_i, pos=pos_i)
        
        # Update state and memory
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        
        state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        
        return res, (state_feat, mem)
    
    def _forward_impl(self, views, ret_state=False):
        """Override _forward_impl to add IMU processing during inference"""
        shape, feat_ls, pos = self.cut3r_model._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self.cut3r_model._init_state(feat[0], pos[0])
        mem = self.cut3r_model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            
            # Process IMU data for this view (same logic as _forward_decoder_step)
            pose_feat_i = None
            if self.cut3r_model.pose_head_flag:
                global_img_feat_i = self.cut3r_model._get_img_level_feat(feat_i)
                if i == 0:
                    original_pose_feat_i = self.cut3r_model.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    original_pose_feat_i = self.cut3r_model.pose_retriever.inquire(global_img_feat_i, mem)
                
                # Check if we have IMU data for this view (skip first view)
                imu_feat = None
                if i > 0 and isinstance(views, (list, tuple)) and i < len(views) and isinstance(views[i], dict) and 'imu' in views[i]:
                    imu_data = views[i]['imu']  # (seq_len, 6)
                    if imu_data.numel() > 0:
                        # Process IMU data for this view
                        imu_feat = self.imu_encoder(imu_data.unsqueeze(0))  # (1, embed_dim)
                        # Ensure we get the right shape - remove all singleton dimensions
                        imu_feat = imu_feat.squeeze()  # (embed_dim,)
                
                if imu_feat is not None:
                    # Add IMU features to pose token (same logic as training)
                    imu_feat_expanded = imu_feat.unsqueeze(0).unsqueeze(0).expand(feat_i.shape[0], 1, -1)
                    pose_feat_i = original_pose_feat_i + imu_feat_expanded
                else:
                    pose_feat_i = original_pose_feat_i
                    
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_pos_i = None
                
            new_state_feat, dec = self.cut3r_model._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.cut3r_model.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.cut3r_model.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.cut3r_model.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.cut3r_model.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.cut3r_model.dec_depth].float(),
            ]
            res = self.cut3r_model._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views 