import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, zeros_
import math

# Import for proper output format
from dust3r.model import ARCroco3DStereoOutput
from dust3r.utils.camera import Mlp

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
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
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
            elif isinstance(m, nn.BatchNorm1d):
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
        pose_encoding_type="absT_quaR",
    ):
        super().__init__()

        self.pose_encoding_type = pose_encoding_type
        if self.pose_encoding_type == "absT_quaR":
            self.target_dim = 7

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=self.target_dim,
            drop=0,
        )

    def forward(
        self,
        pose_feat,
    ):
        """
        pose_feat: BxC
        preliminary_cameras: cameras in opencv coordinate.
        """

        pred_cameras = self.mlp(pose_feat)  # Bx7, 3 for absT, 4 for quaR
        return pred_cameras


class CUT3RIMU(nn.Module):
    """
    Enhanced CUT3R model with IMU encoder and new relative pose retriever.
    This version uses relative pose estimation instead of direct pose token addition.
    """
    def __init__(self, cut3r_model, imu_config):
        super().__init__()
        self.cut3r_model = cut3r_model
        
        # Get device from the base model
        device = next(cut3r_model.parameters()).device
        
        # IMU encoder
        self.imu_encoder = IMUEncoder(
            input_dim=imu_config.get('input_dim', 6),
            seq_len=imu_config.get('seq_len', 10),
            output_dim=cut3r_model.dec_embed_dim,
            dropout=imu_config.get('dropout', 0.1)
        ).to(device)
        
        # New VIFT-style IMU-aware pose retriever
        self.imu_pose_retriever = VIFTIMUAwarePoseRetriever(
            img_feat_dim=cut3r_model.enc_embed_dim,  # image feature dim (encoder)
            imu_feat_dim=cut3r_model.dec_embed_dim,   # imu feature dim (decoder)
            pose_token_dim=cut3r_model.dec_embed_dim, # output pose token dim
            num_layers=2,
            num_heads=cut3r_model.dec_num_heads,
            mlp_ratio=4.0,
            dropout=imu_config.get('dropout', 0.1),
        ).to(device)
        # 新增 relative pose decoder
        self.relative_pose_decoder = RelativePoseDecoder(hidden_size=cut3r_model.dec_embed_dim).to(device)
        # 新增 transformer fusion 融合 pose token
        self.pose_token_transformer = nn.TransformerEncoderLayer(
            d_model=cut3r_model.dec_embed_dim,
            nhead=cut3r_model.dec_num_heads,
            dim_feedforward=int(cut3r_model.dec_embed_dim * 4.0),
            dropout=imu_config.get('dropout', 0.1),
            activation='gelu',
            batch_first=True,
        ).to(device)
        
        # 新增 MLP 融合兩個 pose token
        self.pose_token_fusion_mlp = nn.Sequential(
            nn.Linear(cut3r_model.dec_embed_dim * 2, cut3r_model.dec_embed_dim),
            nn.GELU(),
            nn.Dropout(imu_config.get('dropout', 0.1)),
            nn.Linear(cut3r_model.dec_embed_dim, cut3r_model.dec_embed_dim),
        ).to(device)
        
        # Disable original pose retriever
        if hasattr(cut3r_model, 'pose_retriever'):
            for param in cut3r_model.pose_retriever.parameters():
                param.requires_grad = False
    
    def _reset_sequence_state(self):
        """Reset sequence-specific state variables"""
        self.prev_pose_token = None
        self.feat_i_prev = None
        self.prev_camera_pose = None

        # 基礎模型狀態
        self.state_feat = None
        self.mem = None
        self.init_state_feat = None
        self.init_mem = None
    
    def forward(self, views, ret_state=False):
        """
        Forward pass using relative pose estimation
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
        IMU-enhanced decoder step with new stateless relative pose retriever
        """        
        # 初始化 relative_pose 變量
        relative_pose = None
        
        if self.pose_head_flag:
            global_img_feat_i = self.cut3r_model._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.cut3r_model.pose_token.expand(feat_i.shape[0], -1, -1)
                self.prev_pose_token = None
                relative_pose_token = None
            else:
                imu_data = views[i]['imu']  # (seq_len, 6)
                imu_feat = self.imu_encoder(imu_data)  # (batch_size, embed_dim)                
                # Use IMU-aware pose retriever
                global_img_feat_prev = self.cut3r_model._get_img_level_feat(self.feat_i_prev)
                if global_img_feat_prev.dim() == 3:
                    global_img_feat_prev = global_img_feat_prev.squeeze(1)
                if global_img_feat_i.dim() == 3:
                    global_img_feat_i = global_img_feat_i.squeeze(1)
                if imu_feat.dim() == 3:
                    imu_feat = imu_feat.squeeze(1)
                relative_pose_token = self.imu_pose_retriever(
                    global_img_feat_prev, global_img_feat_i, imu_feat
                )
                
                # 檢查是否有前一個 pose token
                if self.prev_pose_token is not None:
                    # Transformer fusion 融合 prev pose token 和 relative pose token
                    prev_token = self.prev_pose_token.squeeze(1)  # (B, token_dim)
                    # Concatenate prev_token and relative_pose_token for transformer fusion
                    fusion_input = torch.stack([prev_token, relative_pose_token], dim=1)  # (B, 2, token_dim)
                    fusion_output = self.pose_token_transformer(fusion_input)  # (B, 2, token_dim)
                    
                    # 使用 MLP 融合兩個 transformer 輸出 token
                    fused_tokens = torch.cat([fusion_output[:, 0], fusion_output[:, 1]], dim=-1)  # (B, token_dim*2)
                    pose_feat_i = self.pose_token_fusion_mlp(fused_tokens)  # (B, token_dim)
                    pose_feat_i = pose_feat_i.unsqueeze(1)  # (B, 1, token_dim)
                else:
                    # 如果沒有前一個 pose token，直接使用 relative_pose_token
                    pose_feat_i = relative_pose_token.unsqueeze(1)  # (B, 1, token_dim)
                
                relative_pose = self.relative_pose_decoder(relative_pose_token)

            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        
        # Decoder rollout
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

        # 從 decoder 輸出中提取更新後的 pose token
        out_pose_feat_i = dec[-1][:, 0:1]  # 使用 rollout 後的 pose token
        
        # 更新 prev_pose_token 為 rollout 後的結果
        if self.cut3r_model.pose_head_flag:
            self.prev_pose_token = out_pose_feat_i.clone().detach()

        # Generate output  
        head_input = [
            dec[0].float(),
            dec[self.cut3r_model.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.cut3r_model.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.cut3r_model.dec_depth].float(),
        ]
        res = self.cut3r_model._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)  # update global state
        mem = init_mem
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
        
        # Always update feat_i_prev for the next frame (unless we're at the last frame)
        if i < len(views) - 1:
            self.feat_i_prev = feat_i
        
        # 將 relative_pose 加入 res 字典，供 loss 使用
        if relative_pose is not None:
            res["relative_pose"] = relative_pose
        else:
            pass

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
    
    def _recurrent_rollout(self, *args, **kwargs):
        """
        Recurrent rollout - delegated to base model
        """
        return self.cut3r_model._recurrent_rollout(*args, **kwargs)
    
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
        return self.imu_pose_retriever
    
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
        # Reset sequence state at the beginning
        self._reset_sequence_state()
        
        shape, feat_ls, pos = self.cut3r_model._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self.cut3r_model._init_state(feat[0], pos[0])
        mem =  torch.zeros_like(state_feat)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            res, (state_feat, mem) = self._forward_decoder_step(
                views,
                i,
                feat_i,
                pos_i,
                shape[i],
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            )
            ress.append(res)
            all_state_args.append((state_feat, state_pos, init_state_feat, mem, init_mem))
        
        if ret_state:
            return ress, views, all_state_args
        return ress, views 


class VIFTIMUAwarePoseRetriever(nn.Module):
    """
    VIFT-style IMU-aware pose retriever using concatenation and causal transformer:
    - Input: image_feat_t_minus_1, image_feat_t, imu_feat_t
    - Output: relative pose token
    - Network: Causal transformer with concatenated features
    """
    def __init__(self, img_feat_dim, imu_feat_dim, pose_token_dim, num_layers=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        self.imu_feat_dim = imu_feat_dim
        self.pose_token_dim = pose_token_dim
        self.embed_dim = pose_token_dim
        
        # 先融合兩個圖像特徵 - 使用 Transformer Encoder
        self.img_fusion_proj = nn.Linear(img_feat_dim, self.embed_dim)  # 投影到統一維度
        self.img_fusion_norm1 = nn.LayerNorm(self.embed_dim)
        self.img_fusion_norm2 = nn.LayerNorm(self.embed_dim)
        
        # 圖像特徵融合的 Transformer Encoder
        img_fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.img_fusion_transformer = nn.TransformerEncoder(img_fusion_layer, num_layers=1)
        
        # 輸出投影回原始維度
        self.img_fusion_output = nn.Linear(self.embed_dim, img_feat_dim)
        
        # 計算融合後的輸入維度
        self.fused_input_dim = img_feat_dim + imu_feat_dim  # fused_img + imu
        
        # 特徵投影層 - 將融合特徵投影到統一維度
        self.feature_proj = nn.Linear(self.fused_input_dim, self.embed_dim)
        
        # 位置編碼
        self.pos_embedding = None  # 將在 forward 中動態生成
        
        # 因果 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        for p in encoder_layer.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, nonlinearity='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 輸出投影層
        self.output_proj = nn.Linear(self.embed_dim, self.pose_token_dim)
        nn.init.kaiming_normal_(self.output_proj.weight, nonlinearity='linear')
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def positional_embedding(self, seq_length):
        """生成正弦/餘弦位置編碼"""
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * 
                            -(math.log(10000.0) / self.embed_dim))
        pos_embedding = torch.zeros(seq_length, self.embed_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)
    
    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """生成因果掩碼，確保模型只能看到當前及之前的時間步"""
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
            diagonal=1
        )
    
    def forward(self, img_feat_t_minus_1, img_feat_t, imu_feat_t):
        # 確保所有輸入都是2D張量
        if img_feat_t_minus_1.dim() == 3:
            img_feat_t_minus_1 = img_feat_t_minus_1.squeeze(1)
        if img_feat_t.dim() == 3:
            img_feat_t = img_feat_t.squeeze(1)
        if imu_feat_t.dim() == 3:
            imu_feat_t = imu_feat_t.squeeze(1)
        
        # 第一步：使用 Transformer 融合兩個圖像特徵
        # 投影兩個圖像特徵到統一維度
        img_feat_t_minus_1_proj = self.img_fusion_proj(img_feat_t_minus_1)  # (B, embed_dim)
        img_feat_t_proj = self.img_fusion_proj(img_feat_t)  # (B, embed_dim)
        
        # 堆疊成序列進行 Transformer 處理
        img_features = torch.stack([img_feat_t_minus_1_proj, img_feat_t_proj], dim=1)  # (B, 2, embed_dim)
        
        # 通過 Transformer Encoder 融合
        img_features = self.img_fusion_transformer(img_features)  # (B, 2, embed_dim)
        
        # 取平均或最後一個 token 作為融合結果
        img_fused = img_features.mean(dim=1)  # (B, embed_dim)
        
        # 投影回原始維度
        img_fused = self.img_fusion_output(img_fused)  # (B, img_dim)
        
        # 第二步：融合圖像和IMU特徵
        fused_features = torch.cat([img_fused, imu_feat_t], dim=-1)  # (B, img_dim + imu_dim)
        
        # 投影到統一維度
        fused_features = self.feature_proj(fused_features)  # (B, embed_dim)
        
        # 添加 batch 維度以適應 transformer
        fused_features = fused_features.unsqueeze(1)  # (B, 1, embed_dim)
        
        # 生成位置編碼
        pos_embedding = self.positional_embedding(1).to(fused_features.device)  # (1, 1, embed_dim)
        fused_features += pos_embedding
        
        # 生成因果掩碼（對於單一 token，掩碼為空）
        mask = self.generate_square_subsequent_mask(1, fused_features.device)
        
        # 通過因果 Transformer Encoder
        output = self.transformer_encoder(fused_features, mask=mask, is_causal=True)  # (B, 1, embed_dim)
        
        # 提取輸出
        output = output.squeeze(1)  # (B, embed_dim)
        
        # 最終輸出
        rel_pose_token = self.output_proj(output)  # (B, pose_token_dim)
        
        return rel_pose_token