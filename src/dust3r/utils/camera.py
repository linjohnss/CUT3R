from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from croco.models.blocks import Mlp
from dust3r.heads.postprocess import postprocess_pose

inf = float("inf")


class PoseDecoder(nn.Module):
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


class RelativePoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        mlp_ratio=4,
        pose_encoding_type="absT_rot9d",
        num_prompt_tokens=8,
        num_attn_heads=8,
    ):
        super().__init__()

        self.pose_encoding_type = pose_encoding_type
        self.hidden_size = hidden_size
        self.num_prompt_tokens = num_prompt_tokens
        self.num_attn_heads = num_attn_heads

        # =====================================================================
        # Cross-Attention to Previous Pose Token
        # =====================================================================
        # Prompt tokens (query) attend to previous pose token (key/value)
        # This conditions current predictions on previous frame's context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_cross_q = nn.LayerNorm(hidden_size)
        self.norm_cross_kv = nn.LayerNorm(hidden_size)

        # =====================================================================
        # Attention Pooling: Learnable queries for translation and rotation
        # =====================================================================
        # These queries learn to "ask" the prompt tokens for relevant information
        # Translation query: learns to focus on tokens encoding displacement info
        # Rotation query: learns to focus on tokens encoding orientation info
        self.trans_query = nn.Parameter(
            torch.randn(1, 1, hidden_size) * 0.02
        )
        self.rot_query = nn.Parameter(
            torch.randn(1, 1, hidden_size) * 0.02
        )

        # Multi-head cross-attention for pooling
        # Query: learnable query (1 token)
        # Key/Value: prompt tokens (num_prompt_tokens tokens)
        self.trans_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.rot_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attn_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Layer norms for attention pooling
        self.norm_trans_q = nn.LayerNorm(hidden_size)
        self.norm_rot_q = nn.LayerNorm(hidden_size)
        self.norm_pool_kv = nn.LayerNorm(hidden_size)

        # =====================================================================
        # MLP for processing attention-pooled features
        # =====================================================================
        output_dim = int(hidden_size * mlp_ratio)

        # Separate MLPs for translation and rotation pathways
        self.trans_mlp = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
        self.rot_mlp = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

        # =====================================================================
        # Output heads for translation and rotation
        # =====================================================================
        self.fc_t = nn.Linear(output_dim, 3)  # Translation head: 3D vector
        self.fc_rot = nn.Linear(output_dim, 6)  # Rotation head: 6D representation (2 rows of 3)

        # Initialize rotation head to output Identity matrix (first 2 rows)
        # Identity rotation in 6D: [1, 0, 0, 0, 1, 0]
        nn.init.zeros_(self.fc_rot.weight)
        identity_6d = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        self.fc_rot.bias.data.copy_(identity_6d)

        # Translation head: zero-initialized for identity transform
        nn.init.zeros_(self.fc_t.weight)
        nn.init.zeros_(self.fc_t.bias)

    def orthogonalize_rotation(self, R):
        """
        Orthogonalize rotation matrix using Gram-Schmidt (6D representation).
        Uses the first two rows to construct a valid rotation matrix.

        Args:
            R: (B, 2, 3) - first two rows of rotation matrix

        Returns:
            R_ortho: (B, 3, 3) orthogonalized rotation matrices
        """
        # R is (B, 2, 3) - first two rows
        x = R[:, 0]  # First row (B, 3)
        y = R[:, 1]  # Second row (B, 3)

        # Normalize x to get first basis vector
        x_n = F.normalize(x, dim=-1)

        # Compute z = cross(x_n, y) to get third basis vector direction
        z = torch.cross(x_n, y, dim=-1)
        z_n = F.normalize(z, dim=-1)

        # Recompute y to ensure orthogonality: y_n = cross(z_n, x_n)
        y_n = torch.cross(z_n, x_n, dim=-1)

        # Stack as ROWS: [x_n, y_n, z_n] -> (B, 3, 3)
        R_ortho = torch.stack([x_n, y_n, z_n], dim=1)

        return R_ortho

    def forward(self, pose_feat, prev_pose_token=None):
        """
        Forward pass to predict relative pose from prompt tokens.

        Architecture:
        1. Cross-attention: prompt tokens attend to previous pose token
        2. Attention pooling: learnable queries aggregate updated prompt tokens
        3. MLP processing and pose prediction

        Args:
            pose_feat: (B, num_prompt_tokens, hidden_size) - prompt tokens from decoder
            prev_pose_token: (B, hidden_size) - previous frame's SINGLE pose token (optional)

        Returns:
            T_rel: (B, 4, 4) relative SE(3) transform
        """
        B, num_tokens, hidden_dim = pose_feat.shape
        assert num_tokens == self.num_prompt_tokens, \
            f"Expected {self.num_prompt_tokens} tokens, got {num_tokens}"

        # =====================================================================
        # Step 1: Cross-attention to previous pose token
        # =====================================================================
        if prev_pose_token is not None:
            # Normalize inputs
            q = self.norm_cross_q(pose_feat)  # (B, num_prompt_tokens, hidden_size)
            kv = self.norm_cross_kv(prev_pose_token.unsqueeze(1))  # (B, 1, hidden_size)

            # Cross-attention: prompt tokens attend to previous pose token
            pose_feat_attended, _ = self.cross_attn(
                query=q,   # (B, num_prompt_tokens, hidden_size)
                key=kv,    # (B, 1, hidden_size)
                value=kv,  # (B, 1, hidden_size)
            )  # Output: (B, num_prompt_tokens, hidden_size)

            # Residual connection
            pose_feat = pose_feat + pose_feat_attended

        # =====================================================================
        # Step 2: Attention Pooling - aggregate prompt tokens with learnable queries
        # =====================================================================

        # Normalize key/value (updated prompt tokens)
        kv = self.norm_pool_kv(pose_feat)  # (B, num_prompt_tokens, hidden_size)

        # Expand queries for batch dimension
        trans_q = self.trans_query.expand(B, -1, -1)  # (B, 1, hidden_size)
        rot_q = self.rot_query.expand(B, -1, -1)      # (B, 1, hidden_size)

        # Normalize queries
        trans_q = self.norm_trans_q(trans_q)
        rot_q = self.norm_rot_q(rot_q)

        # Cross-attention: learnable queries attend to prompt tokens
        # Translation attention pooling
        trans_feat, trans_attn_weights = self.trans_attn(
            query=trans_q,  # (B, 1, hidden_size)
            key=kv,         # (B, num_prompt_tokens, hidden_size)
            value=kv,       # (B, num_prompt_tokens, hidden_size)
        )  # trans_feat: (B, 1, hidden_size)

        # Rotation attention pooling
        rot_feat, rot_attn_weights = self.rot_attn(
            query=rot_q,    # (B, 1, hidden_size)
            key=kv,         # (B, num_prompt_tokens, hidden_size)
            value=kv,       # (B, num_prompt_tokens, hidden_size)
        )  # rot_feat: (B, 1, hidden_size)

        # Squeeze the sequence dimension
        trans_feat = trans_feat.squeeze(1)  # (B, hidden_size)
        rot_feat = rot_feat.squeeze(1)      # (B, hidden_size)

        # =====================================================================
        # MLP processing for translation and rotation
        # =====================================================================
        trans_feat = self.trans_mlp(trans_feat)  # (B, output_dim)
        rot_feat = self.rot_mlp(rot_feat)        # (B, output_dim)

        # =====================================================================
        # Predict translation and rotation
        # =====================================================================
        # Use float32 for pose prediction to avoid numerical issues with bfloat16
        with torch.cuda.amp.autocast(enabled=False):
            rel_trans = self.fc_t(trans_feat.float())     # (B, 3)
            rel_rot_6d = self.fc_rot(rot_feat.float())    # (B, 6)

        # Reshape to 2x3 matrix (first two rows) and orthogonalize
        rel_rot_matrix = rel_rot_6d.reshape(-1, 2, 3)  # (B, 2, 3)
        rel_rot_matrix = self.orthogonalize_rotation(rel_rot_matrix)  # (B, 3, 3)

        # =====================================================================
        # Construct SE(3) transformation matrix
        # =====================================================================
        device = rel_trans.device
        dtype = rel_trans.dtype
        T_rel = torch.zeros((B, 4, 4), device=device, dtype=dtype)
        T_rel[:, :3, :3] = rel_rot_matrix
        T_rel[:, :3, 3] = rel_trans
        T_rel[:, 3, 3] = 1.0

        return T_rel


class PoseEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        mlp_ratio=4,
        pose_mode=("exp", -inf, inf),
        pose_encoding_type="absT_quaR",
    ):
        super().__init__()
        self.pose_encoding_type = pose_encoding_type
        self.pose_mode = pose_mode

        if self.pose_encoding_type == "absT_quaR":
            self.target_dim = 7

        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim,
            out_dim=hidden_size,
            n_harmonic_functions=10,
            append_input=True,
        )
        self.pose_encoder = Mlp(
            in_features=self.embed_pose.out_dim,
            hidden_features=int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            drop=0,
        )

    def forward(self, camera):
        pose_enc = camera_to_pose_encoding(
            camera,
            pose_encoding_type=self.pose_encoding_type,
        ).to(camera.dtype)
        pose_enc = postprocess_pose(pose_enc, self.pose_mode, inverse=True)
        pose_feat = self.embed_pose(pose_enc)
        pose_feat = self.pose_encoder(pose_feat)
        return pose_feat


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        The harmonic embedding layer supports the classical
        Nerf positional encoding described in
        `NeRF <https://arxiv.org/abs/2003.08934>`_
        and the integrated position encoding in
        `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.

        During the inference you can provide the extra argument `diag_cov`.

        If `diag_cov is None`, it converts
        rays parametrized with a `ray_bundle` to 3D points by
        extending each ray according to the corresponding length.
        Then it converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]

        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.


        If `diag_cov is not None`, it approximates
        conical frustums following a ray bundle as gaussians,
        defined by x, the means of the gaussians and diag_cov,
        the diagonal covariances.
        Then it converts each gaussian
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]::

            [
                sin(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                sin(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),
                ...
                sin(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                cos(f_1*x[..., i]) * exp(0.5 * f_1**2 * diag_cov[..., i,]),
                cos(f_2*x[..., i]) * exp(0.5 * f_2**2 * diag_cov[..., i,]),,
                ...
                cos(f_N * x[..., i]) * exp(0.5 * f_N**2 * diag_cov[..., i,]),
                x[..., i],              # only present if append_input is True.
            ]

        where N equals `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.

        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (embed.sin(), embed.cos(), x)
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(n_harmonic_functions, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.register_buffer(
            "_zero_half_pi",
            torch.tensor([0.0, 0.5 * torch.pi]),
            persistent=False,
        )
        self.append_input = append_input

    def forward(
        self, x: torch.Tensor, diag_cov: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
            diag_cov: An optional tensor of shape `(..., dim)`
                representing the diagonal covariance matrices of our Gaussians, joined with x
                as means of the Gaussians.

        Returns:
            embedding: a harmonic embedding of `x` of shape
            [..., (n_harmonic_functions * 2 + int(append_input)) * num_points_per_ray]
        """

        embed = x[..., None] * self._frequencies

        embed = embed[..., None, :, :] + self._zero_half_pi[..., None, None]

        embed = embed.sin()
        if diag_cov is not None:
            x_var = diag_cov[..., None] * torch.pow(self._frequencies, 2)
            exp_var = torch.exp(-0.5 * x_var)

            embed = embed * exp_var[..., None, :, :]

        embed = embed.reshape(*x.shape[:-1], -1)

        if self.append_input:
            return torch.cat([embed, x], dim=-1)
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int, n_harmonic_functions: int, append_input: bool
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )


class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, out_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def camera_to_pose_encoding(
    camera,
    pose_encoding_type="absT_quaR",
):
    """
    Inverse to pose_encoding_to_camera
    camera: opencv, cam2world
    """
    if pose_encoding_type == "absT_quaR":

        quaternion_R = matrix_to_quaternion(camera[:, :3, :3])

        pose_encoding = torch.cat([camera[:, :3, 3], quaternion_R], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR",
):
    """
    Args:
        pose_encoding: A tensor of shape `BxC`, containing a batch of
                        `B` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """

    if pose_encoding_type == "absT_quaR":

        abs_T = pose_encoding[:, :3]
        quaternion_R = pose_encoding[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    c2w_mats = torch.eye(4, 4).to(R.dtype).to(R.device)
    c2w_mats = c2w_mats[None].repeat(len(R), 1, 1)
    c2w_mats[:, :3, :3] = R
    c2w_mats[:, :3, 3] = abs_T

    return c2w_mats


def quaternion_conjugate(q):
    """Compute the conjugate of quaternion q (w, x, y, z)."""

    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return q_conj


def quaternion_multiply(q1, q2):
    """Multiply two quaternions q1 and q2."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    q_vec = q[..., 1:]
    q_w = q[..., :1]

    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    v_rot = v + q_w * t + torch.cross(q_vec, t, dim=-1)
    return v_rot


def rotation_matrix_to_9d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 9D vector (flatten).
    
    Args:
        R: (B, 3, 3) rotation matrix
        
    Returns:
        rot_9d: (B, 9) flattened rotation matrix
    """
    return R.reshape(*R.shape[:-2], 9)


def rotation_9d_to_matrix(rot_9d: torch.Tensor) -> torch.Tensor:
    """
    Convert 9D vector to 3x3 rotation matrix (reshape).
    
    Args:
        rot_9d: (B, 9) flattened rotation matrix
        
    Returns:
        R: (B, 3, 3) rotation matrix
    """
    return rot_9d.reshape(*rot_9d.shape[:-1], 3, 3)


def pose_12d_to_matrix(pose_12d: torch.Tensor) -> torch.Tensor:
    """
    Convert 12D pose representation (3 translation + 9D rotation) to 4x4 matrix.

    Args:
        pose_12d: (..., 12) where last 12 = [tx, ty, tz, rot_9d]

    Returns:
        T: (..., 4, 4) SE(3) matrix
    """
    from dust3r.utils.camera import rotation_9d_to_matrix

    trans = pose_12d[..., :3]
    rot_9d = pose_12d[..., 3:12]
    R = rotation_9d_to_matrix(rot_9d)

    T = torch.eye(4, device=pose_12d.device, dtype=pose_12d.dtype)
    # Broadcast to batch by expanding
    expand_shape = (*pose_12d.shape[:-1], 4, 4)
    T = T.expand(expand_shape).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = trans
    return T


def pose_matrix_to_12d(T: torch.Tensor) -> torch.Tensor:
    """
    Convert 4x4 pose matrix to 12D representation (3 translation + 9D rotation).

    Args:
        T: (..., 4, 4) SE(3) matrix

    Returns:
        pose_12d: (..., 12) where last 12 = [tx, ty, tz, rot_9d]
    """
    from dust3r.utils.camera import rotation_matrix_to_9d

    trans = T[..., :3, 3]
    R = T[..., :3, :3]
    rot_9d = rotation_matrix_to_9d(R)
    return torch.cat([trans, rot_9d], dim=-1)


def relative_pose_absT_quatR(t1, q1, t2, q2):
    """Compute the relative translation and quaternion between two poses."""

    q1_inv = quaternion_conjugate(q1)

    q_rel = quaternion_multiply(q1_inv, q2)

    delta_t = t2 - t1
    t_rel = rotate_vector(q1_inv, delta_t)
    return t_rel, q_rel
