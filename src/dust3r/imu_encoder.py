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
        self.hidden_size = hidden_size
        
        # Separate MLPs for translation and rotation
        self.trans_mlp = Mlp(
            in_features=4 * hidden_size,  # 4 translation tokens concatenated
            hidden_features=int(4 * hidden_size * mlp_ratio),
            out_features=3,  # Predict 3D translation
            drop=0,
        )
        
        self.rot_mlp = Mlp(
            in_features=4 * hidden_size,  # 4 rotation tokens concatenated
            hidden_features=int(4 * hidden_size * mlp_ratio),
            out_features=9,  # Predict 9D rotation matrix
            drop=0,
        )
        
        # Initialize for better convergence
        self._init_output()

    def _init_output(self):
        """
        Initialize the last layers to output close to zero translation and identity rotation.
        This helps the model start from a good initial state.
        """
        with torch.no_grad():
            # Scale down the last layer weights for small initial predictions
            self.trans_mlp.fc2.weight.data *= 0.01
            self.rot_mlp.fc2.weight.data *= 0.01
            
            # Translation: initialize to zero
            if self.trans_mlp.fc2.bias is not None:
                self.trans_mlp.fc2.bias.data.zero_()
            
            # Rotation (9D): initialize to identity matrix
            if self.rot_mlp.fc2.bias is not None:
                identity_9d = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    device=self.rot_mlp.fc2.bias.data.device,
                    dtype=self.rot_mlp.fc2.bias.data.dtype,
                )
                self.rot_mlp.fc2.bias.data.copy_(identity_9d)

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
        # Save original dtype and device
        original_dtype = R.dtype
        device = R.device
        
        # Force float32 computation for all operations
        with torch.cuda.amp.autocast(enabled=False):
            # Convert to float32
            m = R.float()
            
            # Pi3's svd_orthogonalize algorithm
            # 1. Reshape if needed
            if m.dim() < 3:
                m = m.reshape((-1, 3, 3))
            
            # 2. Normalize each row and transpose
            m_normalized = torch.nn.functional.normalize(m, p=2, dim=-1)
            m_transpose = torch.transpose(m_normalized, dim0=-1, dim1=-2)

            m_transpose = m_transpose.cpu()
            
            # 3. SVD decomposition: M^T = U @ S @ V^T
            u, s, v = torch.svd(m_transpose)

            u = u.to(device)
            v = v.to(device)
            
            # 4. Compute determinant to check orientation
            det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
            
            # 5. Build rotation matrix: R = V @ U^T (adjust last column if det < 0)
            r = torch.matmul(
                torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
                u.transpose(-2, -1)
            )
            
            R_ortho = r
        
        # Convert back to original dtype if needed
        if original_dtype == torch.bfloat16:
            R_ortho = R_ortho.to(original_dtype)
        
        return R_ortho

    def forward(
        self,
        pose_feat,
    ):
        """
        Forward pass to predict relative pose from 8 tokens.
        
        Args:
            pose_feat: (B, 8, hidden_size) - 8 pose tokens (4 for trans, 4 for rot)
            
        Returns:
            T_rel: (B, 4, 4) relative SE(3) transform (camera-to-world convention,
                   to be composed with previous T_c2w)
        """
        # Split the 8 tokens into translation and rotation groups
        trans_tokens = pose_feat[:, :4, :]  # (B, 4, hidden_size) - first 4 tokens for translation
        rot_tokens = pose_feat[:, 4:8, :]   # (B, 4, hidden_size) - last 4 tokens for rotation
        
        # Flatten and concatenate 4 tokens for each group
        trans_feat = trans_tokens.reshape(pose_feat.shape[0], -1)  # (B, 4*hidden_size)
        rot_feat = rot_tokens.reshape(pose_feat.shape[0], -1)      # (B, 4*hidden_size)
        
        # Pass through MLPs
        rel_trans = self.trans_mlp(trans_feat)  # (B, 3)
        rel_rot_9d = self.rot_mlp(rot_feat)     # (B, 9)
        
        # Reshape to 3x3 matrix and orthogonalize, then build a 4x4 SE(3) matrix
        rel_rot_matrix = rel_rot_9d.reshape(-1, 3, 3)  # (B, 3, 3)
        rel_rot_matrix = self.orthogonalize_rotation(rel_rot_matrix)  # (B, 3, 3)
        
        B = rel_trans.shape[0]
        device = rel_trans.device
        dtype = rel_trans.dtype
        T_rel = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        T_rel[:, :3, :3] = rel_rot_matrix
        T_rel[:, :3, 3] = rel_trans
        
        return T_rel


class StateGateModule(nn.Module):
    """
    Gate module for controlling state updates, similar to GRU gates.
    Architecture follows the design in the reference images:
    - Image Encoder path: img_feat + pose_embed
    - State Token path: state_feat + pose_embed → Self-attention → Cross-attention
    - Cross-attention fusion
    - Feed-Forward + Sigmoid → Gate weights
    """
    def __init__(
        self,
        hidden_size=768,
        enc_embed_dim=1024,  # Encoder embedding dimension (input to img_encoder)
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rope=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enc_embed_dim = enc_embed_dim
        
        # Image encoder: project image features from encoder dimension to decoder dimension
        # img_feat comes from encoder output (enc_embed_dim), needs to be projected to decoder dimension (hidden_size)
        self.img_encoder = nn.Linear(enc_embed_dim, hidden_size, bias=True)
        
        # Self-attention for state tokens (state_feat + pose_embed)
        from dust3r.blocks import Attention
        self.self_attn = Attention(
            dim=hidden_size,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm1 = norm_layer(hidden_size)
        
        # Cross-attention: state tokens (query) attend to image+pose (key/value)
        from dust3r.blocks import CrossAttention
        self.cross_attn = CrossAttention(
            dim=hidden_size,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm_layer(hidden_size)
        
        # Feed-Forward network
        # Import Mlp from blocks (same as used in DecoderBlock)
        from dust3r.blocks import Mlp
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm3 = norm_layer(hidden_size)
        
        # Output projection: map to gate weights (one per state token)
        self.gate_proj = nn.Linear(hidden_size, 1, bias=True)
        
        # Initialize gate projection to output values around 0.5 (neutral gate)
        with torch.no_grad():
            self.gate_proj.weight.data *= 0.01
            if self.gate_proj.bias is not None:
                self.gate_proj.bias.data.fill_(0.0)  # After sigmoid, this gives ~0.5
    
    def forward(
        self,
        state_feat,  # (B, n_state, D)
        img_feat,    # (B, n_img, D) - encoder output, needs projection
        pose_embed, # (B, 1, D) - pose token
        state_pos,   # (B, n_state, 2) or None
        img_pos,     # (B, n_img, 2) or None
    ):
        """
        Compute gate weights for state tokens.
        
        Args:
            state_feat: (B, n_state, D) - historical state tokens
            img_feat: (B, n_img, D) - current frame image features (encoder output)
            pose_embed: (B, 1, D) - pose embedding
            state_pos: (B, n_state, 2) or None - state token positions
            img_pos: (B, n_img, 2) or None - image token positions
            
        Returns:
            gate_weights: (B, n_state) - gate weights in [0, 1] range
        """
        B, n_state, D = state_feat.shape
        
        # 1. Image Encoder path: project img_feat and add pose_embed
        img_proj = self.img_encoder(img_feat)  # (B, n_img, D)
        
        # Expand pose_embed to match img_feat length and add
        # For cross-attention, we'll use img_proj + pose_embed as key/value
        # We can either broadcast pose_embed or concatenate
        # Following the image description, we add pose_embed to img features
        # Since img_feat has n_img tokens, we can either:
        #   a) Broadcast pose_embed: (B, 1, D) -> (B, n_img, D)
        #   b) Concatenate: [img_proj, pose_embed] -> (B, n_img+1, D)
        # We'll use broadcasting for simplicity
        img_with_pose = img_proj + pose_embed.expand(-1, img_proj.shape[1], -1)  # (B, n_img, D)
        
        # Create combined key/value for cross-attention
        # Optionally, we can also include pose_embed separately
        kv_feat = torch.cat([img_with_pose, pose_embed], dim=1)  # (B, n_img+1, D)
        kv_pos = None
        if img_pos is not None:
            # Create position for pose_embed (use -1, -1 as in the codebase)
            pose_pos = -torch.ones(B, 1, 2, device=img_pos.device, dtype=img_pos.dtype)
            kv_pos = torch.cat([img_pos, pose_pos], dim=1)  # (B, n_img+1, 2)
        
        # 2. State Token path: state_feat + pose_embed → Self-attention
        # Add pose_embed to state_feat (broadcast)
        state_with_pose = state_feat + pose_embed.expand(-1, n_state, -1)  # (B, n_state, D)
        
        # Self-attention on state tokens
        state_attn_out = self.self_attn(self.norm1(state_with_pose), state_pos)  # (B, n_state, D)
        state_attn_out = state_with_pose + state_attn_out  # Residual connection
        
        # 3. Cross-attention: state tokens (query) attend to image+pose (key/value)
        # Query: state tokens after self-attention
        # Key/Value: image features + pose
        cross_attn_out, _ = self.cross_attn(
            query=self.norm2(state_attn_out),
            key=kv_feat,
            value=kv_feat,
            qpos=state_pos,
            kpos=kv_pos,
            attn_mask=None,
            return_attn_weights=False,
        )  # (B, n_state, D)
        cross_attn_out = state_attn_out + cross_attn_out  # Residual connection
        
        # 4. Feed-Forward
        ff_out = self.mlp(self.norm3(cross_attn_out))  # (B, n_state, D)
        gate_input = cross_attn_out + ff_out  # Residual connection
        
        # 5. Project to gate weights and apply sigmoid
        gate_logits = self.gate_proj(gate_input).squeeze(-1)  # (B, n_state)
        gate_weights = torch.sigmoid(gate_logits)  # (B, n_state) in [0, 1]
        
        return gate_weights


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

        # 8 tokens: 4 for translation, 4 for rotation
        self.relative_pose_token = nn.Parameter(
            torch.randn(1, 8, self.cut3r_model.dec_embed_dim) * 0.02, requires_grad=True
        )

        # IMU encoder
        self.imu_encoder = IMUEncoder(
            input_dim=imu_config.get('input_dim', 6),
            seq_len=imu_config.get('seq_len', 10),
            output_dim=self.cut3r_model.dec_embed_dim,
            dropout=imu_config.get('dropout', 0.0)
        ).to(device)
        
        self.relative_pose_decoder = RelativePoseDecoder(
            hidden_size=self.cut3r_model.dec_embed_dim,
        ).to(device)
        
        # Gate modules for state update control (optional)
        # Reset Gate: controls how much history state is "forgotten"
        # Update Gate: controls how much new information is incorporated
        self.use_gating = imu_config.get('use_gating', True)  # Default to True for backward compatibility
        
        if self.use_gating:
            self.reset_gate = StateGateModule(
                hidden_size=self.cut3r_model.dec_embed_dim,
                enc_embed_dim=self.cut3r_model.enc_embed_dim,  # Input dimension for img_encoder
                num_heads=self.cut3r_model.dec_num_heads,
                mlp_ratio=4.0,
                rope=self.cut3r_model.rope,
            ).to(device)
            
            self.update_gate = StateGateModule(
                hidden_size=self.cut3r_model.dec_embed_dim,
                enc_embed_dim=self.cut3r_model.dec_embed_dim,  # For update gate, decoded_img_feat is already in decoder dimension (from dec[-1])
                num_heads=self.cut3r_model.dec_num_heads,
                mlp_ratio=4.0,
                rope=self.cut3r_model.rope,
            ).to(device)
        else:
            self.reset_gate = None
            self.update_gate = None
    
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
                # Initialize state for pose accumulation
                self.prev_pose_token = None
                self.prev_state_attn_gate = None  # 初始化前一幀的 attention gate
                # Initialize with identity SE(3) transform (camera-to-world)
                B = feat_i.shape[0]
                eye4 = torch.eye(4, device=feat_i.device, dtype=feat_i.dtype)
                self.prev_T_c2w = eye4.unsqueeze(0).expand(B, -1, -1).clone()
            pose_feat_i = self.cut3r_model.pose_retriever.inquire(global_img_feat_i, mem)

            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
            use_imu_feat = False
            if 'imu' in views[i]:
                imu_data = views[i]['imu']  # (seq_len, 6)
                imu_feat_i = self.imu_encoder(imu_data)
                if use_imu_feat:
                    rel_pose_feat_i = imu_feat_i.unsqueeze(1)  # [B, dec_embed_dim] -> [B, 1, dec_embed_dim]
                else:
                    rel_pose_feat_i = self.relative_pose_token.expand(feat_i.shape[0], -1, -1)  # [B, 1, dec_embed_dim]
            else:
                # 如果没有 IMU 数据，直接使用 relative_pose_token
                rel_pose_feat_i = self.relative_pose_token.expand(feat_i.shape[0], -1, -1)  # [B, 1, dec_embed_dim]
            rel_pose_pos_i = -torch.ones(
                feat_i.shape[0], 8, 2, device=feat_i.device, dtype=pos_i.dtype
            )
            
            rel_pose_attn_mask = None
            # Use keyframe's attention gate for all frames in that interval
            if i > 0 and self.prev_state_attn_gate is not None:
                # Use top 50% tokens instead of fixed threshold
                # self.keyframe_state_attn_gate shape: [B, n_state]
                # Note: torch.quantile requires float32/float64, convert from bf16 if needed
                gate_float = self.prev_state_attn_gate.float()
                threshold = torch.quantile(gate_float, 0.5, dim=-1, keepdim=True)  # [B, 1]
                # Create mask: 0 for top 20% (attend), -inf for bottom 80% (mask out)
                gate_expanded = gate_float.unsqueeze(1)  # [B, 1, n_state]
                mask = torch.zeros_like(gate_expanded)
                mask[gate_expanded < threshold.unsqueeze(1)] = float('-inf')
                # Expand to 8 tokens: [B, 1, n_state] -> [B, 8, n_state]
                rel_pose_attn_mask = mask.expand(-1, 8, -1).to(rel_pose_feat_i.dtype)  # [B, 8, n_state]
            
        else:
            pose_feat_i = None
            pose_pos_i = None
            rel_pose_feat_i = None
            rel_pose_pos_i = None
            rel_pose_attn_mask = None
        
        # Apply Reset Gate BEFORE decoder (as shown in the reference image)
        # Reset gate modulates the historical state before it's fed into the decoder
        if i == 0:
            # First frame: no history to reset, use state_feat directly
            reset_state_feat = state_feat
        elif self.use_gating and self.reset_gate is not None:
            # Compute reset gate weights using current frame features and pose
            if self.pose_head_flag:
                # Use already computed pose_feat_i as pose embedding
                pose_embed_i = pose_feat_i  # (B, 1, D) - use pose_token as pose embedding
            else:
                # If no pose_head, use a zero pose embedding
                pose_embed_i = torch.zeros(
                    feat_i.shape[0], 1, self.cut3r_model.dec_embed_dim,
                    device=feat_i.device, dtype=feat_i.dtype
                )
            
            # Reset Gate: controls how much history state is "forgotten" before decoder
            # This is applied BEFORE the decoder processes the state
            reset_weights = self.reset_gate(
                state_feat=state_feat,  # (B, n_state, D) - historical state
                img_feat=feat_i,  # (B, n_img, D) - current frame encoder features
                pose_embed=pose_embed_i,  # (B, 1, D)
                state_pos=state_pos,
                img_pos=pos_i,
            )  # (B, n_state)
            
            # Apply reset gate to modulate history state: M_reset = M_prev * reset_weights
            reset_state_feat = state_feat * reset_weights.unsqueeze(-1)  # (B, n_state, D)
        else:
            # Gating disabled: use state_feat directly
            reset_state_feat = state_feat
        
        # Decoder rollout (傳入 reset 後的 state_feat 和 relative_pose_token)
        # The decoder now processes the reset-gated state, producing candidate state M̂
        new_state_feat, dec, state_gate = self._recurrent_rollout(
            reset_state_feat,  # Use reset-gated state instead of original state_feat
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
            rel_pose_attn_mask=None,
            return_state_gate=True,
        )
        
        if state_gate is not None:
            self.prev_state_attn_gate = state_gate.detach()

        # 從 decoder 輸出中提取更新後的 pose token
        out_pose_feat_i = dec[-1][:, 0:1]  # 使用 rollout 後的 pose token
        new_mem = self.cut3r_model.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )

        # 統一使用 out_pose_feat_i 來獲取當前幀的 pose token
        pose_token_curr = out_pose_feat_i.squeeze(1)  # [B, D] - 當前幀融合了所有歷史的 pose token
        
        # 初始化 relative_pose 變量 (now as 4x4)
        relative_pose = None
        
        if self.cut3r_model.pose_head_flag and i > 0:
            # 提取 decode 完的 relative pose tokens（在 _decoder 中 concat 在最後）
            # dec[-1] 的格式：[pose_token, img_tokens..., 8 relative_pose_tokens]
            # relative_pose_tokens 是最後 8 個 tokens (4 for trans, 4 for rot)
            decoded_relative_pose_tokens = dec[-1][:, -8:]  # [B, 8, D]
            
            # 將 decode 完的 8 個 relative pose tokens 轉成 4x4 相對位姿矩陣
            T_rel = self.relative_pose_decoder(
                pose_feat=decoded_relative_pose_tokens  # [B, 8, D] -> [B, 4, 4]
            )
            
            # 位姿累加邏輯 (使用 4x4 SE(3) 矩陣, 相機到世界 T_c2w)
            # Loss 中定義的 T_rel 是從 current frame 到 previous frame (curr → prev)
            # 因此要得到 current frame 的位姿，需要使用 T_rel 的逆矩陣
            #   T_c2w_curr = T_c2w_prev @ T_rel^(-1)
            # 其中 T_rel^(-1) 是從 previous frame 到 current frame (prev → curr)
            # Note: torch.inverse() doesn't support BFloat16, so convert to float32 first
            T_rel_inv = torch.inverse(T_rel.float()).to(T_rel.dtype)  # (B, 4, 4)
            current_T_c2w = self.prev_T_c2w @ T_rel_inv  # (B, 4, 4)
            self.prev_T_c2w = current_T_c2w
            
            # 設置 relative_pose 供 loss 使用（必須保持梯度）
            relative_pose = T_rel  # (B, 4, 4)
        
        # 更新 pose tokens
        if self.cut3r_model.pose_head_flag:
            # 更新 prev_pose_token 為 rollout 後的結果，使用統一的 pose_token_curr
            self.prev_pose_token = pose_token_curr.clone().detach()  # (B, pose_token_dim)


        if self.pose_head_flag:
            # 移除 relative_pose_tokens（最後 8 個 tokens）
            def remove_last_8_tokens(x):
                return x[:, :-8]
            
            head_input = [
                dec[0].float(),  # dec[0] 没有额外 token，保持原样
                remove_last_8_tokens(dec[self.cut3r_model.dec_depth * 2 // 4])[:, 1:].float(),  # [pose, img, 8*rel] -> [pose, img] -> [img]
                remove_last_8_tokens(dec[self.cut3r_model.dec_depth * 3 // 4])[:, 1:].float(),  # [pose, img, 8*rel] -> [pose, img] -> [img]
                remove_last_8_tokens(dec[self.cut3r_model.dec_depth]).float(),  # [pose, img, 8*rel] -> [pose, img]
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
        
        # Apply Update Gate AFTER decoder (as shown in the reference image)
        # Update gate controls how the candidate state M̂ (new_state_feat) is combined with original M_prev
        # This is the final state update step
        if i == 0:
            # First frame: use new_state_feat directly (no history to update)
            state_feat = new_state_feat
        elif self.use_gating and self.update_gate is not None:
            # Update Gate: input should be new_state_feat (candidate M̂) and decoded features
            # Extract decoded image features from decoder output
            # dec[-1] format: [pose_token, img_tokens..., 8 relative_pose_tokens] (if pose_head_flag)
            if self.pose_head_flag:
                # Remove relative_pose_tokens (last 8 tokens) to get [pose_token, img_tokens...]
                # Then remove pose_token to get image features only
                decoded_with_pose = dec[-1][:, :-8]  # Remove last 8 relative_pose_tokens
                decoded_img_feat = decoded_with_pose[:, 1:]  # Remove pose_token, keep img_tokens
                # Use decoded pose token as pose embedding
                decoded_pose_embed = out_pose_feat_i  # (B, 1, D) - decoded pose token
                # Position encoding: same as input image tokens (no pose token position needed)
                decoded_img_pos = pos_i  # (B, n_img, 2)
            else:
                # No pose token, use all decoder output as image features
                decoded_img_feat = dec[-1]  # (B, n_img, D)
                # Use zero pose embedding
                decoded_pose_embed = torch.zeros(
                    feat_i.shape[0], 1, self.cut3r_model.dec_embed_dim,
                    device=feat_i.device, dtype=feat_i.dtype
                )
                decoded_img_pos = pos_i  # (B, n_img, 2)
            
            # Update Gate: controls how much new information (M̂) is incorporated
            # Input: new_state_feat (candidate M̂), decoded image features, and decoded pose embedding
            update_weights = self.update_gate(
                state_feat=new_state_feat,  # Use candidate state M̂ (decoder output)
                img_feat=decoded_img_feat,  # Use decoded image features (not encoder features)
                pose_embed=decoded_pose_embed,  # Use decoded pose token
                state_pos=state_pos,
                img_pos=decoded_img_pos,
            )  # (B, n_state)
            
            # Apply update gate to combine original state and candidate state:
            # M_new = (1 - z) * M_prev + z * M̂
            # This matches the reference image: update gate blends M_prev with candidate M̂
            state_feat = (1 - update_weights.unsqueeze(-1)) * state_feat + \
                         update_weights.unsqueeze(-1) * new_state_feat
        else:
            # Gating disabled: use new_state_feat directly (simple replacement)
            state_feat = new_state_feat
        
        mem = new_mem
        
        # CRITICAL FIX: Detach decoder features since they come from frozen CUT3R model
        self.dec_feat_prev = dec[-1].clone().detach()  # 使用最新的解碼輸出
        
        # 將相對位姿與全局位姿加入 res 字典，供 loss 與可視化使用
        if self.cut3r_model.pose_head_flag and i > 0 and relative_pose is not None:
            # relative_pose: (B, 4, 4) - 相對 SE(3) 變換
            res["relative_pose"] = relative_pose
        if self.pose_head_flag and hasattr(self, "prev_T_c2w") and self.prev_T_c2w is not None:
            # camera_pose: (B, 4, 4) - camera-to-world 變換矩陣
            res["camera_pose"] = self.prev_T_c2w.clone()

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
    
    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, f_rel_pose=None, pos_rel_pose=None, rel_pose_attn_mask=None, return_attn=False):
        """
        自定義的 decoder，接收外部準備好的 relative_pose_tokens (8 tokens)
        
        Args:
            f_rel_pose: [B, 8, dec_embed_dim] - 準備好的 8 個 relative pose tokens (4 trans + 4 rot)
            pos_rel_pose: [B, 8, 2] - relative pose tokens 的位置編碼
            rel_pose_attn_mask: Optional[Tensor] - [B, 8, n_state] attention mask for rel_pose_tokens
                                只讓 rel_pose_tokens 跟前一幀相關的 state feature 做 attention
            return_attn: bool - 是否回傳 cross-attention maps（用於計算 state gate）
        """
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        
        # 1. 先投影 f_img 到 decoder 維度
        f_img = self.cut3r_model.decoder_embed(f_img)  # [B, N, enc_embed_dim] -> [B, N, dec_embed_dim]
        
        # 2. 如果提供了 relative_pose_tokens (8 tokens)，concat 到 f_img 後面
        has_rel_pose = f_rel_pose is not None and pos_rel_pose is not None
        if has_rel_pose:
            f_img = torch.cat([f_img, f_rel_pose], dim=1)  # [B, N+8, dec_embed_dim]
            pos_img = torch.cat([pos_img, pos_rel_pose], dim=1)  # [B, N+8, 2]
        
        # 3. 如果有 pose_head，concat pose_token 到最前面
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)  # [B, 1+..., dec_embed_dim]
            pos_img = torch.cat([pos_pose, pos_img], dim=1)  # [B, 1+..., 2]
        
        final_output.append((f_state, f_img))
        
        # 收集 cross-attention maps (用於計算 state gate)
        cross_attn_state_maps = [] if return_attn else None

        # 5. Decoder blocks
        for blk_state, blk_img in zip(self.cut3r_model.dec_blocks_state, self.cut3r_model.dec_blocks):
            # 為 image decoder block 準備 cross-attention mask (image tokens attend to state)
            # f_img shape: [B, n_img_tokens, D] where n_img_tokens = 1(pose) + N(img) + 8(rel_pose)
            # f_state shape: [B, n_state, D]
            # 只需要為最後 8 個 tokens (rel_pose_tokens) 應用 mask
            img_cross_attn_mask = None
            if has_rel_pose and rel_pose_attn_mask is not None:
                # rel_pose_attn_mask: [B, 8, n_state] - 針對 8 個 rel_pose tokens
                # 需要擴展成 [B, n_img_tokens, n_state]，其他 token 不加 mask
                B, n_img_tokens, _ = f_img.shape
                n_state = f_state.shape[1]
                # 創建全零 mask (允許所有 attention)
                img_cross_attn_mask = torch.zeros(B, n_img_tokens, n_state,
                                                   device=f_img.device, dtype=f_img.dtype)
                # 將最後 8 個 tokens (rel_pose) 的 mask 設置為提供的 mask
                img_cross_attn_mask[:, -8:, :] = rel_pose_attn_mask

            if (
                self.cut3r_model.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # Note: gradient checkpointing 時無法回傳 attention，此時 return_attn 應為 False
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
                # blk_state: state tokens attend to image (cross-attn from state to image)
                # 我們需要這個 attention map 來計算 gate
                if return_attn:
                    # 真正回傳 cross-attention weights
                    f_state, _, state_cross_attn = blk_state(
                        *final_output[-1][::+1], pos_state, pos_img,
                        return_cross_attn=True
                    )
                    # state_cross_attn: (B, num_heads, n_state, n_img)
                    # 收集這些 attention maps 用於後續計算 gate
                    cross_attn_state_maps.append(state_cross_attn)
                else:
                    f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img)

                # blk_img: image tokens attend to state, 使用 cross_attn_mask
                f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state, cross_attn_mask=img_cross_attn_mask)
            final_output.append((f_state, f_img))
        
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.cut3r_model.dec_norm_state(final_output[-1][0]),
            self.cut3r_model.dec_norm(final_output[-1][1]),
        )
        
        if return_attn:
            return zip(*final_output), cross_attn_state_maps
        return zip(*final_output), None
    
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
        rel_pose_attn_mask=None,  # 新增：attention mask for rel_pose_token
        return_state_gate=False,  # 是否回傳當前幀的 state gate
    ):
        """
        使用自定義的 _decoder，傳遞 relative_pose_token 和對應的 attention mask
        """
        decoder_output, cross_attn = self._decoder(
            state_feat, state_pos, current_feat, current_pos, 
            pose_feat, pose_pos,
            f_rel_pose=rel_pose_feat,
            pos_rel_pose=rel_pose_pos,
            rel_pose_attn_mask=rel_pose_attn_mask,
            return_attn=return_state_gate
        )
        # decoder_output 是 zip(*final_output) 的結果，會產生兩個 iterator
        # 第一個 iterator 是所有層的 state，第二個是所有層的 img
        state_outputs, img_outputs = decoder_output
        state_outputs = list(state_outputs)
        img_outputs = list(img_outputs)
        
        new_state_feat = state_outputs[-1]  # 最後一層的 state
        dec = img_outputs  # decoder 的 image outputs
        
        # 計算當前幀的 state gate（參考 TTT3R 的方法）
        # TTT3R: 使用 state → image 的 attention 來判斷哪些 state tokens 是 "active" 的
        state_gate = None
        if return_state_gate and cross_attn is not None and len(cross_attn) > 0:
            # cross_attn 是一個 list，每個元素是 state → image 的 attention (來自 blk_state)
            # 每個 attention map: (B, num_heads, n_state, n_img)

            # 參考 TTT3R:
            # cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 
            #     'l h nstate nimg -> 1 nstate nimg (l h)')
            # state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
            # update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None]
            
            # 聚合所有層的 attention: stack along layer dim
            # cross_attn[i]: (B, num_heads, n_state, n_img)
            stacked_attn = torch.stack(cross_attn, dim=0)  # (num_layers, B, num_heads, n_state, n_img)

            # 對 image tokens 和 (layers, heads) 取平均，得到每個 state token 的 "activity"
            # 這個值表示：這個 state token 整體上對 image 的關注程度
            state_activity = stacked_attn.mean(dim=(0, 2, 4))  # (B, n_state) - mean over layers, heads, n_img

            # 使用 sigmoid 將 raw attention scores 轉換成 [0, 1] 的 soft gate
            # 高 activity 的 state tokens 得到接近 1 的 gate
            state_gate = torch.sigmoid(state_activity)

        return new_state_feat, dec, state_gate
    
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

            # MEMORY OPTIMIZATION: Only use precomputed state for the FIRST frame
            # Subsequent frames reuse the recurrently updated state from previous iterations
            # This reduces memory usage from N×state_size to 1×state_size per batch
            if i == 0:
                # First frame: load precomputed state if available
                if use_precomputed_states and 'precomputed_state' in view:
                    precomp_state = view['precomputed_state']
                    state_feat = precomp_state['state_feat'].to(device)
                    state_pos = precomp_state['state_pos'].to(device)
                    init_state_feat = precomp_state['init_state_feat'].to(device)
                    mem = precomp_state['mem'].to(device)
                    init_mem = precomp_state['init_mem'].to(device)
                    # Detach precomputed states to prevent gradient flow
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
                    if mem.dim() == 4:
                        if mem.shape[1] == 1:
                            mem = mem.squeeze(1)
                        else:
                            mem = mem.squeeze(1)
                        init_mem = init_mem.squeeze(1) if init_mem.dim() == 4 else init_mem
                else:
                    # No precomputed state: initialize from scratch
                    state_feat, state_pos = self.cut3r_model._init_state(feat_i, pos_i)
                    mem = self.cut3r_model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                    init_state_feat = state_feat.clone()
                    init_mem = mem.clone()
            # else: i > 0, reuse state from previous iteration (recurrent update)
            #       state_feat, mem already exist and were updated by previous decoder step
                
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
