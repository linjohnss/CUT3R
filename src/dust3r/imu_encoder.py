import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, zeros_
from torch.utils.checkpoint import checkpoint
import math

# Import for proper output format
from dust3r.model import ARCroco3DStereoOutput
from dust3r.utils.camera import Mlp, quaternion_multiply, quaternion_conjugate, rotate_vector, standardize_quaternion
from dust3r.heads.postprocess import postprocess_pose

class IMUEncoder(nn.Module):
    """
    Enhanced IMU encoder that processes IMU sequences to extract motion features.
    Uses 1D convolutions with residual connections and zero-initialized output.
    """
    def __init__(self, input_dim=6, seq_len=10, output_dim=768, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Feature extraction layers with residual connections
        hidden_dims = [64, 128, 256]
        
        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(hidden_dims[i], hidden_dims[i+1] if i+1 < len(hidden_dims) else hidden_dims[i])
            for i in range(len(hidden_dims))
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 投影到與 pose token 一致的維度 (使用 Linear 層更簡潔)
        self.projection = nn.Linear(hidden_dims[2] * seq_len, output_dim)
        
        self._init_weights()
        
    def _make_res_block(self, in_dim, out_dim):
        """Create a residual block with 1D convolutions"""
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_dim), num_channels=out_dim),  # 改用 GroupNorm
            nn.ReLU(inplace=True),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_dim), num_channels=out_dim),  # 改用 GroupNorm
        )
        
    def _init_weights(self):
        """Initialize weights following VIFT's initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                # Skip ZeroConv1d as it initializes itself to zero
                # Check if this module is marked as zero conv
                if hasattr(m, '_is_zero_conv') and m._is_zero_conv:
                    continue
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, imu_data):
        """
        Forward pass through IMU encoder
        
        Args:
            imu_data: (batch_size, seq_len, 6) or (seq_len, 6)
            
        Returns:
            imu_features: (batch_size, output_dim)
        """
        if imu_data.dim() == 2:
            imu_data = imu_data.unsqueeze(0)  # Add batch dimension
        
        # Transpose for conv1d: (batch_size, 6, seq_len)
        x = imu_data.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Residual blocks with skip connections
        for i, res_block in enumerate(self.res_blocks):
            identity = x
            x = res_block(x)
            
            # Add residual connection if dimensions match
            if x.shape == identity.shape:
                x = x + identity
            
            x = F.relu(x)
        
        x = self.dropout(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)  # (batch_size, 256 * seq_len)
        x = self.projection(x)     # (batch_size, output_dim)
        
        return x


class RelativePoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        mlp_ratio=4,
        pose_encoding_type="absT_rot9d",
    ):
        super().__init__()

        self.pose_encoding_type = pose_encoding_type
        # Directly predict 3 translation + 9 rotation matrix (flattened 3x3)
        self.target_dim = 12

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=self.target_dim,
            drop=0,
        )
        
        # Initialize for better convergence
        self._init_rotation_output()

    def _init_rotation_output(self):
        """
        Initialize the last layer to output close to identity rotation.
        This helps the model start from a good initial state.
        """
        with torch.no_grad():
            # Scale down the last layer weights for small initial predictions
            self.mlp.fc2.weight.data *= 0.01
            
            if self.mlp.fc2.bias is not None:
                # Translation: initialize to zero
                self.mlp.fc2.bias.data[:3] = 0.0
                
                # Rotation: initialize to identity matrix (flattened)
                # [[1, 0, 0],
                #  [0, 1, 0],
                #  [0, 0, 1]]
                identity_9d = torch.eye(3).reshape(9)
                self.mlp.fc2.bias.data[3:12] = identity_9d

    def orthogonalize_rotation(self, R):
        """
        Orthogonalize rotation matrix using SVD (differentiable).
        Ensures the output is a valid rotation matrix with det(R) = +1.
        
        Note: SVD and det operations don't support bfloat16 on CUDA,
        so we wrap the entire computation in float32 context.
        
        Args:
            R: (B, 3, 3) potentially non-orthogonal matrices
            
        Returns:
            R_ortho: (B, 3, 3) orthogonalized rotation matrices
        """
        # Save original dtype
        original_dtype = R.dtype
        
        # Force float32 computation for all operations
        with torch.cuda.amp.autocast(enabled=False):
            # Convert to float32
            R_float = R.float()
            
            # SVD decomposition
            U, _, Vh = torch.linalg.svd(R_float)
            
            # Reconstruct orthogonal matrix: R = U @ Vh
            R_ortho = torch.bmm(U, Vh)
            
            # Ensure det(R) = +1 (proper rotation, not reflection)
            det = torch.det(R_ortho)
            
            # If det is negative, flip the sign of the last column of Vh
            Vh_corrected = Vh.clone()
            Vh_corrected[:, -1, :] *= det.sign().view(-1, 1)
            R_ortho = torch.bmm(U, Vh_corrected)
        
        # Convert back to original dtype if needed
        if original_dtype == torch.bfloat16:
            R_ortho = R_ortho.to(original_dtype)
        
        return R_ortho

    def forward(
        self,
        pose_feat,
    ):
        """
        Forward pass to predict relative pose.
        
        Args:
            pose_feat: (B, hidden_size) pose features
            
        Returns:
            pred_pose: (B, 12) = (B, 3 translation + 9 rotation)
        """
        pred = self.mlp(pose_feat)  # Bx12
        
        # Extract translation and rotation
        rel_trans = pred[:, :3]  # (B, 3)
        rel_rot_9d = pred[:, 3:12]  # (B, 9)
        
        # Reshape to 3x3 matrix and orthogonalize
        rel_rot_matrix = rel_rot_9d.reshape(-1, 3, 3)  # (B, 3, 3)
        rel_rot_matrix = self.orthogonalize_rotation(rel_rot_matrix)  # (B, 3, 3)
        
        # Flatten back to 9D
        rel_rot_9d = rel_rot_matrix.reshape(-1, 9)  # (B, 9)
        
        # Concatenate translation and rotation
        pred_pose = torch.cat([rel_trans, rel_rot_9d], dim=-1)  # (B, 12)
        
        return pred_pose

class CrossAttentionPoseDecoder(nn.Module):
    def __init__(
            self,
            hidden_size=768,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1,
    ):
        super().__init__()
        self.target_dim = 7
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 交叉注意力層
        # 讓 current_decoder_feat (Query) 去關注 prev_decoder_feat (Key, Value)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True  # 讓輸入的維度是 [Batch, Sequence, Dim]
        )
        
        # 注意力後的 Layer Normalization 和殘差連接
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # MLP (Feed-Forward Network)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # MLP 後的 Layer Normalization 和殘差連接
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # 最終的姿態預測頭
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.target_dim),
        )
        
        # Dropout for the final feature before the pose head
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, current_decoder_feat, prev_decoder_feat):
        """
        Args:
            current_decoder_feat (torch.Tensor): 當前幀的完整 decoder feature [B, N, D] - 作為 Query
            prev_decoder_feat (torch.Tensor): 前一幀的完整 decoder feature [B, N, D] - 作為 Key & Value
        Returns:
            torch.Tensor: 預測出的相對姿態 [B, 7]
        """
        
        # 1. Cross-Attention 使用完整的 decoder features
        # Query: 當前幀的完整 decoder feature，代表「問題」
        # Context: 前一幀的完整 decoder feature，代表「上下文」或「參考答案」
        attn_output, _ = self.attention(
            query=current_decoder_feat,
            key=prev_decoder_feat,
            value=prev_decoder_feat,
            need_weights=False
        )
        
        # 2. 第一個殘差連接和正規化 (Add & Norm)
        # 將注意力提取出的「變化」資訊加回到原始的「問題」上
        x = self.norm1(current_decoder_feat + attn_output)
        
        # 3. MLP Block
        mlp_output = self.mlp(x)
        
        # 4. 第二個殘差連接和正規化 (Add & Norm)
        x = self.norm2(x + mlp_output)
        
        # 5. 提取 pose token 部分 (第一個 token): [B, N, D] -> [B, D]
        pose_token_feat = x[:, 0:1].squeeze(1)  # [B, D]
        
        # 6. 最終姿態預測
        pred_pose = self.pose_head(self.dropout_final(pose_token_feat))
        
        return pred_pose


class CUT3RIMU(nn.Module):
    """
    Enhanced CUT3R model with IMU encoder and new relative pose retriever.
    This version uses relative pose estimation and accumulates to global pose from frame 0.
    """
    def __init__(self, cut3r_model, imu_config):
        super().__init__()
        self.cut3r_model = cut3r_model
        self.cut3r_model.eval()
        
        # CRITICAL FIX: Freeze CUT3R model parameters to prevent gradient flow
        # This ensures only the relative_pose_decoder is trained
        for param in self.cut3r_model.parameters():
            param.requires_grad = False
        
        # Get device from the base model
        device = next(cut3r_model.parameters()).device

        self.relative_pose_token = nn.Parameter(
            torch.randn(1, 1, self.cut3r_model.dec_embed_dim) * 0.02, requires_grad=True
        )

        # IMU encoder
        self.imu_encoder = IMUEncoder(
            input_dim=imu_config.get('input_dim', 6),
            seq_len=imu_config.get('seq_len', 10),
            output_dim=self.cut3r_model.dec_embed_dim,
            dropout=imu_config.get('dropout', 0.0)
        ).to(device)

        # self.relative_pose_decoder = CrossAttentionPoseDecoder(
        #     hidden_size=self.relative_pose_dim,
        #     num_heads=8,
        #     mlp_ratio=4,
        #     dropout=0.0,
        # ).to(device)

        self.relative_pose_decoder = RelativePoseDecoder(
            hidden_size=self.cut3r_model.dec_embed_dim,
        ).to(device)


    
    def forward(self, views, ret_state=False):
        """
        Forward pass using relative pose estimation and accumulation
        """
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def _forward_encoder(self, views):
        """
        Forward pass through encoder only - stateless retriever: no mem needed
        """
        # Get encoding from base model
        return self.cut3r_model._forward_encoder(views)
    
    def _encode_views(self, views):
        """
        Encode views - delegated to base model
        """
        return self.cut3r_model._encode_views(views)
    
    def _decode_features(self, feats, masks, shape):
        """
        Decode features - delegated to base model
        """
        return self.cut3r_model._decode_features(feats, masks, shape)
    
    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        """
        IMU-enhanced decoder step with relative pose accumulation using 9D rotation matrix
        """
        
        if self.pose_head_flag:
            global_img_feat_i = self.cut3r_model._get_img_level_feat(feat_i)
            if i == 0:
                self.prev_pose_token = None
                # Initialize with identity rotation (as 9D vector)
                current_trans = torch.zeros(feat_i.shape[0], 3, device=feat_i.device, dtype=feat_i.dtype)
                current_rot_matrix = torch.eye(3, device=feat_i.device, dtype=feat_i.dtype).unsqueeze(0).expand(feat_i.shape[0], -1, -1)  # (B, 3, 3)
                current_rot_9d = current_rot_matrix.reshape(feat_i.shape[0], 9)  # (B, 9)
                self.prev_camera_pose = (current_trans, current_rot_9d)
            pose_feat_i = self.cut3r_model.pose_retriever.inquire(global_img_feat_i, mem)

            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
            use_imu_feat = False
            imu_data = views[i]['imu']  # (seq_len, 6)
            imu_feat_i = self.imu_encoder(imu_data)
        
            if use_imu_feat:
                rel_pose_feat_i = imu_feat_i.unsqueeze(1)  # [B, dec_embed_dim] -> [B, 1, dec_embed_dim]
            else:
                rel_pose_feat_i = self.relative_pose_token.expand(feat_i.shape[0], -1, -1)  # [B, 1, dec_embed_dim]
            rel_pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )

        else:
            pose_feat_i = None
            pose_pos_i = None
            rel_pose_feat_i = None
            rel_pose_pos_i = None
        
        # Decoder rollout (傳入準備好的 relative_pose_token)
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=None,
            reset_mask=None,
            update=None,
            rel_pose_feat=rel_pose_feat_i,
            rel_pose_pos=rel_pose_pos_i,
        )

        # 從 decoder 輸出中提取更新後的 pose token
        out_pose_feat_i = dec[-1][:, 0:1]  # 使用 rollout 後的 pose token
        new_mem = self.cut3r_model.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )

        # 統一使用 out_pose_feat_i 來獲取當前幀的 pose token
        pose_token_curr = out_pose_feat_i.squeeze(1)  # [B, D] - 當前幀融合了所有歷史的 pose token
        
        # 初始化 relative_pose 變量
        relative_pose = None
        
        if self.cut3r_model.pose_head_flag and i > 0:
            # CRITICAL FIX: Use precomputed decoder features for consistency with full sequence
            # Instead of using self.dec_feat_prev which accumulates errors during batch training
            current_decoder_feat = dec[-1]  # [B, N, D] - 當前幀的完整 decoder feature
            prev_decoder_feat = self.dec_feat_prev  # [B, N, D]
            
            # 提取 decode 完的 relative pose token（在 _decoder 中 concat 在最後）
            # dec[-1] 的格式：[pose_token, img_tokens..., relative_pose_token]
            # relative_pose_token 是最後一個 token
            decoded_relative_pose_token = current_decoder_feat[:, -1]  # [B, D]

            # 將 decode 完的 relative pose token 作為 decoder 的輸入
            relative_pose = self.relative_pose_decoder(
                pose_feat=decoded_relative_pose_token  # [B, D]
            )  # (B, 12) = (B, 3 + 9) - 已經過 SVD 正交化
                        
            # 位姿累加邏輯 (使用 9D rotation matrix)
            # Loss 中定義: R_rel = R_curr^T @ R_prev (backward definition)
            # 因此累加時需要: R_curr = R_prev @ R_rel^T
            from dust3r.utils.camera import rotation_9d_to_matrix, rotation_matrix_to_9d
            
            rel_trans = relative_pose[:, :3]  # (B, 3) - 相對平移
            rel_rot_9d = relative_pose[:, 3:12]  # (B, 9) - 9D rotation (已正交化)
            
            # 將 9D rotation 轉換成 3x3 rotation matrix
            rel_rot_matrix = rotation_9d_to_matrix(rel_rot_9d)  # (B, 3, 3)
            
            # 從 prev_camera_pose 恢復上一幀的 rotation matrix
            prev_rot_9d = self.prev_camera_pose[1]  # (B, 9)
            prev_rot_matrix = rotation_9d_to_matrix(prev_rot_9d)  # (B, 3, 3)
            
            # 累加旋轉: R_curr = R_prev @ R_rel^T
            # 因為 loss 中定義 R_rel = R_curr^T @ R_prev
            current_rot_matrix = torch.bmm(prev_rot_matrix, rel_rot_matrix.transpose(-2, -1))  # (B, 3, 3)
            
            # 累加平移: t_curr = t_prev - R_curr @ t_rel = t_prev - (R_prev @ R_rel^T) @ t_rel
            # 因為 loss 中定義 t_rel = R_curr^T @ (t_prev - t_curr)
            # 先計算 R_rel^T @ t_rel
            rel_trans_rotated = torch.bmm(rel_rot_matrix.transpose(-2, -1), rel_trans.unsqueeze(-1)).squeeze(-1)  # (B, 3)
            # 再用 R_prev 旋轉
            world_trans_offset = torch.bmm(prev_rot_matrix, rel_trans_rotated.unsqueeze(-1)).squeeze(-1)  # (B, 3)
            current_trans = self.prev_camera_pose[0] - world_trans_offset
            
            # 將 rotation matrix 展平為 9D
            current_rot_9d = rotation_matrix_to_9d(current_rot_matrix)  # (B, 9)
            
            self.prev_camera_pose = (current_trans, current_rot_9d)
        
        # 更新 pose tokens
        if self.cut3r_model.pose_head_flag:
            # 更新 prev_pose_token 為 rollout 後的結果，使用統一的 pose_token_curr
            self.prev_pose_token = pose_token_curr.clone().detach()  # (B, pose_token_dim)


        if self.pose_head_flag:
            # # 移除 relative_pose_token（最後一個 token）
            def remove_last_token(x):
                return x[:, :-1]
            
            head_input = [
                dec[0].float(),  # dec[0] 没有额外 token，保持原样
                remove_last_token(dec[self.cut3r_model.dec_depth * 2 // 4])[:, 1:].float(),  # [pose, img, rel] -> [pose, img] -> [img]
                remove_last_token(dec[self.cut3r_model.dec_depth * 3 // 4])[:, 1:].float(),  # [pose, img, rel] -> [pose, img] -> [img]
                remove_last_token(dec[self.cut3r_model.dec_depth]).float(),  # [pose, img, rel] -> [pose, img]
            ]
            # head_input = [
            #     dec[0].float(),
            #     dec[self.cut3r_model.dec_depth * 2 // 4][:, 1:].float(),
            #     dec[self.cut3r_model.dec_depth * 3 // 4][:, 1:].float(),
            #     dec[self.cut3r_model.dec_depth].float(),
            # ]
        else:
            # 没有 pose_head，dec 中没有 pose_token 和 relative_pose_token
            head_input = [
                dec[0].float(),
                dec[self.cut3r_model.dec_depth * 2 // 4].float(),
                dec[self.cut3r_model.dec_depth * 3 // 4].float(),
                dec[self.cut3r_model.dec_depth].float(),
            ]
        res = self.cut3r_model._downstream_head(head_input, shape_i, pos=pos_i)
        
        # Always update state without mask filtering
        state_feat = new_state_feat
        mem = new_mem
        
        # CRITICAL FIX: Detach decoder features since they come from frozen CUT3R model
        self.dec_feat_prev = dec[-1].clone().detach()  # 使用最新的解碼輸出
        
        # 將 relative_pose 加入 res 字典，供 loss 使用
        if self.cut3r_model.pose_head_flag and i > 0:
            res["relative_pose"] = relative_pose  # (B, 12) = (B, 3 + 9)
        if self.prev_camera_pose is not None:
            current_trans, current_rot_9d = self.prev_camera_pose
            # camera_pose: (B, 12) = (B, 3 + 9) - translation + 9D rotation matrix
            current_pose = torch.cat([current_trans, current_rot_9d], dim=-1)
            res["camera_pose"] = current_pose

        return res, (state_feat, mem)
    
    def _init_state(self, feat, pos):
        """
        Initialize state - delegated to base model
        """
        return self.cut3r_model._init_state(feat, pos)
    
    def _get_img_level_feat(self, feat):
        """
        Get image level features - delegated to base model
        """
        return self.cut3r_model._get_img_level_feat(feat)
    
    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, f_rel_pose=None, pos_rel_pose=None):
        """
        自定義的 decoder，接收外部準備好的 relative_pose_token
        
        Args:
            f_rel_pose: [B, 1, dec_embed_dim] - 準備好的 relative pose token feature
            pos_rel_pose: [B, 1, 2] - relative pose token 的位置編碼
        """
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        
        # 1. 先投影 f_img 到 decoder 維度
        f_img = self.cut3r_model.decoder_embed(f_img)  # [B, N, enc_embed_dim] -> [B, N, dec_embed_dim]
        
        # 2. 如果提供了 relative_pose_token，concat 到 f_img 後面
        if f_rel_pose is not None and pos_rel_pose is not None:
            f_img = torch.cat([f_img, f_rel_pose], dim=1)  # [B, N+1, dec_embed_dim]
            pos_img = torch.cat([pos_img, pos_rel_pose], dim=1)  # [B, N+1, 2]
        
        # 3. 如果有 pose_head，concat pose_token 到最前面
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)  # [B, 1+..., dec_embed_dim]
            pos_img = torch.cat([pos_pose, pos_img], dim=1)  # [B, 1+..., 2]
        
        final_output.append((f_state, f_img))
        
        # 5. Decoder blocks
        for blk_state, blk_img in zip(self.cut3r_model.dec_blocks_state, self.cut3r_model.dec_blocks):
            if (
                self.cut3r_model.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _ = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_reentrant=not self.cut3r_model.fixed_input_length,
                )
                f_img, _ = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    use_reentrant=not self.cut3r_model.fixed_input_length,
                )
            else:
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)
                f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
            final_output.append((f_state, f_img))
        
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.cut3r_model.dec_norm_state(final_output[-1][0]),
            self.cut3r_model.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)
    
    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        rel_pose_feat=None,  # 新增：relative pose token feature
        rel_pose_pos=None,   # 新增：relative pose token position
    ):
        """
        使用自定義的 _decoder，傳遞 relative_pose_token
        """
        new_state_feat, dec = self._decoder(
            state_feat, state_pos, current_feat, current_pos, 
            pose_feat, pose_pos,
            f_rel_pose=rel_pose_feat,
            pos_rel_pose=rel_pose_pos
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec
    
    def _downstream_head(self, *args, **kwargs):
        """
        Downstream head - delegated to base model
        """
        return self.cut3r_model._downstream_head(*args, **kwargs)
    
    @property
    def dec_depth(self):
        """Delegate dec_depth to base model"""
        return self.cut3r_model.dec_depth
    
    @property
    def pose_retriever(self):
        """Use IMU-aware pose retriever for compatibility"""
        return self.cut3r_model.pose_retriever
    
    @property
    def config(self):
        """Delegate config to base model"""
        return self.cut3r_model.config
    
    @property
    def enc_embed_dim(self):
        """Delegate enc_embed_dim to base model"""
        return self.cut3r_model.enc_embed_dim
    
    @property
    def dec_embed_dim(self):
        """Delegate dec_embed_dim to base model"""
        return self.cut3r_model.dec_embed_dim
    
    @property
    def dec_num_heads(self):
        """Delegate dec_num_heads to base model"""
        return self.cut3r_model.dec_num_heads
    
    @property
    def pose_head_flag(self):
        """Delegate pose_head_flag to base model"""
        return self.cut3r_model.pose_head_flag
    
    @property
    def pose_token(self):
        """Delegate pose_token to base model"""
        return self.cut3r_model.pose_token
    
    def _forward_impl(self, views, ret_state=False):
        ress = []
        all_state_args = []
        
        # Initialize state variables
        state_feat = None
        state_pos = None
        init_state_feat = None
        mem = None
        init_mem = None
        
        # Reset IMU state for each batch to ensure consistency
        self.prev_pose_token = None
        self.prev_camera_pose = None
        
        # Check if any view has precomputed state
        use_precomputed_states = any('precomputed_state' in view for view in views) if len(views) > 0 else False
        
        # Process views one at a time (similar to forward_recurrent)
        for i, view in enumerate(views):
            # Get device from the current view
            device = view["img"].device
            
            # Handle image encoding - dynamic shape handling for training vs demo
            # view["img"] can have different shapes depending on context:
            # - Training: [C, H, W] (from dataset transforms)
            # - Demo: [1, C, H, W] (from load_images)
            if view["img"].dim() == 3:  # [C, H, W] - training case
                imgs = view["img"].unsqueeze(0)  # Add batch dimension: [1, C, H, W]
                batch_size = 1
            elif view["img"].dim() == 4:  # [1, C, H, W] - demo case
                imgs = view["img"]  # Already has batch dimension
                batch_size = view["img"].shape[0]
            else:
                raise ValueError(f"Unexpected image tensor shape: {view['img'].shape}, expected 3D [C,H,W] or 4D [1,C,H,W]")
            
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )
            
            # Ensure imgs has correct shape for processing
            if imgs.dim() == 4 and imgs.shape[0] == 1:
                # Already correct: [1, C, H, W]
                pass
            else:
                # Reshape to ensure batch dimension is first
                imgs = imgs.view(-1, *imgs.shape[1:])  # [B, C, H, W]
            shapes = shapes.view(-1, 2).to(imgs.device)
            
            # Encode image directly without mask filtering
            img_out, img_pos, _ = self.cut3r_model._encode_image(imgs, shapes)
            feat_i = img_out[-1]
            pos_i = img_pos
            shape = shapes

            # CRITICAL FIX: Use precomputed state for EVERY frame, not just frame 0
            # This ensures each frame sees the correct historical context from the full 200-frame sequence
            if use_precomputed_states and 'precomputed_state' in view:
                precomp_state = view['precomputed_state']
                state_feat = precomp_state['state_feat'].to(device)
                state_pos = precomp_state['state_pos'].to(device)
                init_state_feat = precomp_state['init_state_feat'].to(device)
                mem = precomp_state['mem'].to(device)
                init_mem = precomp_state['init_mem'].to(device)
                
                # CRITICAL FIX: Detach precomputed states to prevent gradient flow through them
                # This ensures that only the relative_pose_decoder receives gradients, not the precomputed states
                state_feat = state_feat.detach()
                state_pos = state_pos.detach()
                init_state_feat = init_state_feat.detach()
                mem = mem.detach()
                init_mem = init_mem.detach()
                # Fix: Remove extra dimensions if present
                if state_feat.dim() == 4 and state_feat.shape[1] == 1:
                    state_feat = state_feat.squeeze(1)
                    state_pos = state_pos.squeeze(1) if state_pos.dim() == 4 else state_pos
                    init_state_feat = init_state_feat.squeeze(1) if init_state_feat.dim() == 4 else init_state_feat
                
                # Fix mem dimensions as well
                # mem should be [batch_size, local_mem_size, 2 * dec_embed_dim]
                if mem.dim() == 4:
                    if mem.shape[1] == 1:
                        mem = mem.squeeze(1)
                    else:
                        # If mem.shape[1] != 1, we might have [B, 1, N, C] or [B, N, N, C]
                        # Try to squeeze the second dimension first
                        mem = mem.squeeze(1)
                    init_mem = init_mem.squeeze(1) if init_mem.dim() == 4 else init_mem
            else:
                # Initialize from scratch only if no precomputed state available
                if i == 0:
                    state_feat, state_pos = self.cut3r_model._init_state(feat_i, pos_i)
                    mem = self.cut3r_model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                    init_state_feat = state_feat.clone()
                    init_mem = mem.clone()
                # else: continue using state from previous frame (already updated)
                
            if ret_state:
                all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))

            # Create a temporary views list with only the current view
            temp_views = [None] * (i + 1)
            temp_views[i] = view

            res, (state_feat, mem) = self._forward_decoder_step(
                temp_views,
                i,
                feat_i,
                pos_i,
                shape,
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            )
            ress.append(res)

        if ret_state:
            return ress, views, all_state_args
        return ress, views
    
    def forward_recurrent(self, views, device, ret_state=False):
        """
        Forward pass using recurrent processing - processes views one by one to save GPU memory.
        Based on ARCroco3DStereo.forward_recurrent but adapted for IMU-enhanced processing.
        """
        ress = []
        all_state_args = []
        processed_views = []  # Store processed views with minimal data
        
        # Initialize state variables
        state_feat = None
        state_pos = None
        init_state_feat = None
        mem = None
        init_mem = None
        
        # Reset IMU state for each batch to ensure consistency
        self.prev_pose_token = None
        self.prev_camera_pose = None
               
        # Process views one at a time to save GPU memory
        for i, cpu_view in enumerate(views):
            print(f"Processing view {i + 1}/{len(views)} - GPU memory management active (IMU-enhanced)")
            
            # Move current view to GPU
            view = {}
            ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"])
            
            for name, value in cpu_view.items():
                if name in ignore_keys:
                    view[name] = value
                elif isinstance(value, tuple) or isinstance(value, list):
                    view[name] = [x.to(device, non_blocking=True) for x in value]
                else:
                    view[name] = value.to(device, non_blocking=True)
            
            # Set device from the current view
            current_device = view["img"].device
            
            # Handle image encoding - similar to original but without ray_maps
            # view["img"] can have different shapes depending on context:
            # - Training: [C, H, W] (from dataset transforms)
            # - Demo: [1, C, H, W] (from load_images)
            if view["img"].dim() == 3:  # [C, H, W] - training case
                imgs = view["img"].unsqueeze(0)  # Add batch dimension: [1, C, H, W]
            elif view["img"].dim() == 4:  # [1, C, H, W] - demo case
                imgs = view["img"]  # Already has batch dimension
            else:
                raise ValueError(f"Unexpected image tensor shape: {view['img'].shape}, expected 3D [C,H,W] or 4D [1,C,H,W]")
            
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=current_device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            
            # imgs is already [1, C, H, W] from load_images, no need to reshape
            shapes = shapes.view(-1, 2).to(imgs.device)
            
            # Encode image directly without mask filtering
            img_out, img_pos, _ = self.cut3r_model._encode_image(imgs, shapes)
            feat_i = img_out[-1]
            pos_i = img_pos
            shape = shapes

            if i == 0:
                state_feat, state_pos = self.cut3r_model._init_state(feat_i, pos_i)
                mem = self.cut3r_model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if ret_state:
                all_state_args.append(
                    (state_feat.cpu(), state_pos.cpu(), init_state_feat.cpu(), mem.cpu(), init_mem.cpu())
                )

            # CRITICAL FIX: Create a temporary views list with only the current view
            # but positioned at the correct index to maintain IMU state consistency
            temp_views = [None] * (i + 1)  # Create list of correct length
            temp_views[i] = view  # Place current view at correct index
            
            res, (state_feat, mem) = self._forward_decoder_step(
                temp_views,  # Pass views list with current view at correct index
                i,           # Use the correct frame index
                feat_i,
                pos_i,
                shape,
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            )
            
            # Move result to CPU immediately to save GPU memory
            res_cpu = {}
            for key, value in res.items():
                if isinstance(value, torch.Tensor):
                    res_cpu[key] = value.cpu()
                else:
                    res_cpu[key] = value
            ress.append(res_cpu)
            
            # Create minimal processed view (keep only essential data on CPU)
            processed_view = {
                "img": view["img"].cpu(),
                "idx": view.get("idx", i),
                "instance": view.get("instance", str(i)),
                "reset": view["reset"].cpu(),  # Need this for prepare_output
            }
            processed_views.append(processed_view)
            # Explicit cleanup
            del view, imgs, feat_i, pos_i, res
            if 'img_out' in locals():
                del img_out
            if 'img_pos' in locals():
                del img_pos
                
            # Force GPU memory cleanup
            torch.cuda.empty_cache()
            
            # Print memory usage every 50 frames
            if (i + 1) % 50 == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"  After {i+1} frames: {allocated:.2f}GB GPU memory allocated")
            
        if ret_state:
            return ress, processed_views, all_state_args
        return ress, processed_views 
