import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.model import ARCroco3DStereo
from accelerate import Accelerator
import re


def custom_sort_key(key):
    text = key.split("/")
    if len(text) > 1:
        text, num = text[0], text[-1]
        return (text, int(num))
    else:
        return (key, -1)


def merge_chunk_dict(old_dict, curr_dict, add_number):
    new_dict = {}
    for key, value in curr_dict.items():

        match = re.search(r"(\d+)$", key)
        if match:

            num_part = int(match.group()) + add_number

            new_key = re.sub(r"(\d+)$", str(num_part), key, 1)
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    new_dict = old_dict | new_dict
    return {k: new_dict[k] for k in sorted(new_dict.keys(), key=custom_sort_key)}


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(
    batch,
    model,
    criterion,
    accelerator: Accelerator,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=not inference):
        if inference:
            output, state_args = model(batch, ret_state=True)
            preds, batch = output.ress, output.views
            result = dict(views=batch, pred=preds)
            return result[ret] if ret else result, state_args
        else:
            output = model(batch)
            preds, batch = output.ress, output.views

        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(batch, preds) if criterion is not None else None

    result = dict(views=batch, pred=preds, loss=loss)
    return result[ret] if ret else result


def loss_of_one_batch_tbptt(
    batch,
    model,
    criterion,
    chunk_size,
    loss_scaler,
    optimizer,
    accelerator: Accelerator,
    log_writer=None,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)
    all_preds = []
    all_loss = 0.0
    all_loss_details = {}
    with torch.cuda.amp.autocast(enabled=not inference):
        with torch.no_grad():
            (feat, pos, shape), (
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            ) = accelerator.unwrap_model(model)._forward_encoder(batch)
        feat = [f.detach() for f in feat]
        pos = [p.detach() for p in pos]
        shape = [s.detach() for s in shape]
        init_state_feat = init_state_feat.detach()
        init_mem = init_mem.detach()
        pose_token_buffer = []  # Sliding window buffer across chunks

        for chunk_id in range((len(batch) - 1) // chunk_size + 1):
            preds = []
            chunk = []
            state_feat = state_feat.detach()
            state_pos = state_pos.detach()
            mem = mem.detach()
            pose_token_buffer = [(idx, t.detach()) for idx, t in pose_token_buffer]  # Detach buffer at chunk boundary
            if chunk_id < ((len(batch) - 1) // chunk_size + 1) - 4:
                with torch.no_grad():
                    for in_chunk_idx in range(chunk_size):
                        i = chunk_id * chunk_size + in_chunk_idx
                        if i >= len(batch):
                            break
                        res, (state_feat, mem), pose_token_buffer = accelerator.unwrap_model(
                            model
                        )._forward_decoder_step(
                            batch,
                            i,
                            feat_i=feat[i],
                            pos_i=pos[i],
                            shape_i=shape[i],
                            init_state_feat=init_state_feat,
                            init_mem=init_mem,
                            state_feat=state_feat,
                            state_pos=state_pos,
                            mem=mem,
                            pose_token_buffer=pose_token_buffer,
                        )
                        preds.append(res)
                        all_preds.append({k: v.detach() for k, v in res.items()})
                        chunk.append(batch[i])
                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, chunk_id * chunk_size
                    )
                    del loss
            else:
                for in_chunk_idx in range(chunk_size):
                    i = chunk_id * chunk_size + in_chunk_idx
                    if i >= len(batch):
                        break
                    res, (state_feat, mem), pose_token_buffer = accelerator.unwrap_model(
                        model
                    )._forward_decoder_step(
                        batch,
                        i,
                        feat_i=feat[i],
                        pos_i=pos[i],
                        shape_i=shape[i],
                        init_state_feat=init_state_feat,
                        init_mem=init_mem,
                        state_feat=state_feat,
                        state_pos=state_pos,
                        mem=mem,
                        pose_token_buffer=pose_token_buffer,
                    )
                    preds.append(res)
                    all_preds.append({k: v.detach() for k, v in res.items()})
                    chunk.append(batch[i])
                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, chunk_id * chunk_size
                    )
                    loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),
                        update_grad=True,
                        clip_grad=1.0,
                    )
                    optimizer.zero_grad()
                    del loss
    result = dict(
        views=batch,
        pred=all_preds,
        loss=(all_loss / ((len(batch) - 1) // chunk_size + 1), all_loss_details),
        already_backprop=True,
    )
    return result[ret] if ret else result


@torch.no_grad()
def inference(groups, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    res, state_args = loss_of_one_batch(groups, model, None, None, inference=True)
    result = to_cpu(res)
    return result, state_args


@torch.no_grad()
def inference_step(view, state_args, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for name in view.keys():  # pseudo_focal
        if name in ignore_keys:
            continue
        if isinstance(view[name], tuple) or isinstance(view[name], list):
            view[name] = [x.to(device, non_blocking=True) for x in view[name]]
        else:
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=False):
        state_feat, state_pos, init_state_feat, mem, init_mem = state_args
        pred, _, _ = model.inference_step(
            view, state_feat, state_pos, init_state_feat, mem, init_mem
        )

    res = dict(pred=pred)
    result = to_cpu(res)
    return result


@torch.no_grad()
def inference_recurrent(groups, model, device, verbose=True, ref_frame_indices_fn=None,
                        keyframe_indices=None, on_frame_processed=None,
                        buffer_pruning_fn=None):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps (one at a time)")
    # Keep views on CPU initially
    cpu_views = []
    for view in groups:
        cpu_view = {}
        for name, value in view.items():
            if name in ignore_keys:
                cpu_view[name] = value
            else:
                # Keep on CPU for now
                cpu_view[name] = value
        cpu_views.append(cpu_view)
    with torch.cuda.amp.autocast(enabled=False):
        preds, batch, state_args = model.forward_recurrent(
            cpu_views, device, ret_state=True,
            ref_frame_indices_fn=ref_frame_indices_fn,
            keyframe_indices=keyframe_indices,
            on_frame_processed=on_frame_processed,
            buffer_pruning_fn=buffer_pruning_fn,
        )
        res = dict(views=batch, pred=preds)
    result = to_cpu(res)
    return result, state_args


def check_if_same_size(pairs):
    shapes1 = [img1["img"].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2["img"].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(
        shapes2[0] == s for s in shapes2
    )


def get_pred_pts3d(gt, pred, use_pose=False, inplace=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:

        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:

        assert use_pose is True
        return (
            pred["pts3d_in_other_view"]
            if inplace
            else pred["pts3d_in_other_view"].clone()
        )

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)

        for iter in range(10):

            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)

            w = dis.clip_(min=1e-8).reciprocal()

            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)

    return scaling


# =====================================================================
# Keyframe-only buffer: MUSt3R-style overlap NN
# =====================================================================

def make_kf_only_callbacks(**params):
    """Sliding-window keyframe buffer (sw_kf_48) with 3D overlap score.

    Maintains a KD-tree of world-frame 3D points from keyframes.
    For each new frame, queries KD-tree with its high-confidence points;
    large NN distances mean new area is visible -> insert as keyframe.
    Buffer keeps keyframes within a temporal window + the most recent 1 frame.

    Params:
        kf_window: 48               — temporal window for keyframe retention
        keyframe_overlap_thr: 0.1   — NN distance threshold for new area
        overlap_percentile: 85      — percentile of NN distances as score
        min_conf_keyframe: 1.2      — confidence gate for keyframe + point filter
        kf_x_subsamp: 4             — spatial subsampling for speed
        depth_normalize: True       — divide distances by depth (nn-norm mode)

    Returns:
        (ref_frame_indices_fn, on_frame_processed, keyframe_indices, buffer_pruning_fn)
    """
    num_init_frames = params.get('num_init_frames', 2)
    kf_window = params.get('kf_window', 48)
    keyframe_indices = set()
    for i in range(num_init_frames):
        keyframe_indices.add(i)  # First num_init_frames frames are always keyframes

    # ── Buffer pruning: keep keyframes within temporal window + latest 1 frame ──
    def kf_only_buffer_pruning(pose_token_buffer, _kf_indices):
        latest_idx = pose_token_buffer[-1][0]
        # Only keep keyframes within the temporal window
        kf_entries = [(idx, f) for idx, f in pose_token_buffer
                      if idx in keyframe_indices and (latest_idx - idx) <= kf_window]
        # Non-kf: keep only the latest frame
        non_kf = [(idx, f) for idx, f in pose_token_buffer
                  if idx not in keyframe_indices]
        recent = non_kf[-1:] if non_kf else []
        seen = set()
        result = []
        for entry in kf_entries + recent:
            if entry[0] not in seen:
                seen.add(entry[0])
                result.append(entry)
        # Ensure latest is always included
        if latest_idx not in seen:
            result.append(pose_token_buffer[-1])
        result.sort(key=lambda x: x[0])
        return result

    # ── Ref selection: use most recent buffer entries as references ──
    def ref_frame_indices_fn(frame_idx, pose_token_buffer):
        if frame_idx == 0:
            return None
        refs = [idx for idx, _ in pose_token_buffer if idx < frame_idx]
        if not refs:
            return None
        # Return only the most recent entries (model caps at num_prompt_tokens)
        # _assemble takes first N from this list, so return most recent last
        return refs

    from scipy.spatial import KDTree as _KDTree
    import numpy as _np

    min_conf_kf = params.get('min_conf_keyframe', 1.2)
    overlap_thr = params.get('keyframe_overlap_thr', 0.1)
    percentile = params.get('overlap_percentile', 85)
    kf_subsamp = params.get('kf_x_subsamp', 4)
    depth_normalize = params.get('depth_normalize', True)  # nn-norm mode
    quadrant_divider = params.get('quadrant_divider', 2)   # 2 -> 8 quadrants

    _pose_history = {}       # frame_idx -> (4, 4) c2w tensor (CPU)

    # ── Quadrant-aware KDTree (MUSt3R-style) ──
    # Ref: must3r/slam/nns.py QuandrantSearcher + must3r/slam/tools.py get_quadrant_id
    _n_quadrants = 2 * quadrant_divider ** 2
    _quadrant_pts = [[] for _ in range(_n_quadrants)]    # accumulated pts per quadrant
    _quadrant_trees = [None] * _n_quadrants               # KDTree per quadrant

    def _get_quadrant_id(rays, eps=1e-5):
        """Assign each ray direction to a quadrant on the unit sphere."""
        rays = rays / _np.linalg.norm(rays, axis=-1, keepdims=True).clip(eps)
        thetas = (_np.arccos(rays[:, -1]) / _np.pi).clip(eps, 1 - eps)
        phis = (_np.arctan2(rays[:, 1], rays[:, 0]) / _np.pi).clip(-1 + eps, 1 - eps)
        theta_idx = _np.floor(thetas * quadrant_divider).astype(int)
        phis_idx = _np.floor(phis * quadrant_divider).astype(int) + quadrant_divider
        return (theta_idx + phis_idx * quadrant_divider).astype(int)

    def _add_to_overlap_tree(pts_world_np, cam_center_np):
        """Add keyframe points to quadrant-aware KDTree."""
        rays = pts_world_np - cam_center_np[None]
        quad_ids = _get_quadrant_id(rays)
        for q in _np.unique(quad_ids):
            mask = quad_ids == q
            if len(_quadrant_pts[q]) == 0:
                _quadrant_pts[q] = pts_world_np[mask]
            else:
                _quadrant_pts[q] = _np.concatenate([_quadrant_pts[q], pts_world_np[mask]])
            _quadrant_trees[q] = _KDTree(_quadrant_pts[q])

    def _query_overlap_tree(pts_world_np, cam_center_np):
        """Query quadrant-aware KDTree for NN distances."""
        rays = pts_world_np - cam_center_np[None]
        quad_ids = _get_quadrant_id(rays)
        dists = _np.full(pts_world_np.shape[0], _np.inf)
        for q in _np.unique(quad_ids):
            mask = quad_ids == q
            tree = _quadrant_trees[q]
            if tree is not None:
                d, _ = tree.query(pts_world_np[mask], k=1, workers=4)
                dists[mask] = d
        return dists

    def _accumulate_c2w(frame_idx, result):
        """Compute c2w from relative poses to known references."""
        if frame_idx == 0:
            c2w = torch.eye(4, dtype=torch.float32)
            _pose_history[0] = c2w
            return c2w

        rel_poses = result.get('relative_poses')
        ref_indices = result.get('ref_frame_indices')

        if rel_poses is not None and ref_indices is not None:
            K = rel_poses.shape[1]
            for k, ref_idx in enumerate(ref_indices):
                if k >= K:
                    break
                if ref_idx in _pose_history:
                    T_rel = rel_poses[0, k].cpu().float()
                    T_rel_inv = torch.inverse(T_rel)
                    c2w = _pose_history[ref_idx] @ T_rel_inv
                    _pose_history[frame_idx] = c2w
                    return c2w

        # Fallback: use previous frame's c2w
        if (frame_idx - 1) in _pose_history:
            c2w = _pose_history[frame_idx - 1].clone()
        else:
            c2w = torch.eye(4, dtype=torch.float32)
        _pose_history[frame_idx] = c2w
        return c2w

    def _compute_overlap_score(pts_world_np, depths_np, cam_center_np):
        """NN overlap score using quadrant-aware KDTree (nn-norm mode)."""
        has_any_tree = any(t is not None for t in _quadrant_trees)
        if not has_any_tree:
            return float('inf')

        dists = _query_overlap_tree(pts_world_np, cam_center_np)

        if depth_normalize:
            dists = dists / (_np.abs(depths_np) + 1e-9)

        dists[_np.isposinf(dists)] = _np.finfo(dists.dtype).max
        return float(_np.percentile(dists, percentile))

    def _extract_world_pts(result, c2w, subsamp):
        """Extract high-conf pts in world frame + local depths."""
        pts3d_local = result.get('pts3d_in_self_view')
        conf = result.get('conf_self')
        if pts3d_local is None or conf is None:
            return None, None

        pts = pts3d_local[0].cpu().float()   # (H, W, 3)
        c = conf[0].cpu().float()            # (H, W)

        if subsamp:
            pts = pts[::subsamp, ::subsamp]
            c = c[::subsamp, ::subsamp]

        msk = c > min_conf_kf
        if msk.sum() == 0:
            return None, None

        pts_masked = pts[msk]                        # (N, 3) in camera frame
        depths = pts_masked[:, 2].numpy()            # Z as depth

        # Transform to world frame: P_world = c2w @ [P_cam; 1]
        ones = torch.ones(pts_masked.shape[0], 1)
        pts_homo = torch.cat([pts_masked, ones], dim=-1)   # (N, 4)
        pts_world = (c2w @ pts_homo.T).T[:, :3]           # (N, 3)

        return pts_world.numpy(), depths

    def on_frame_processed(frame_idx, result):
        c2w = _accumulate_c2w(frame_idx, result)
        pts_world, depths = _extract_world_pts(result, c2w, kf_subsamp)
        cam_center = c2w[:3, 3].numpy()

        # Initial frames are always keyframes (like MUSt3R's num_init_frames)
        if frame_idx < num_init_frames:
            if pts_world is not None:
                _add_to_overlap_tree(pts_world, cam_center)
            return

        if pts_world is None:
            return

        overlap_score = _compute_overlap_score(pts_world, depths, cam_center)

        conf = result.get('conf_self')
        median_conf = conf[0].median().item() if conf is not None else 0.0

        is_kf = (overlap_score > overlap_thr) and (median_conf > min_conf_kf)

        if is_kf:
            keyframe_indices.add(frame_idx)
            _add_to_overlap_tree(pts_world, cam_center)

    return ref_frame_indices_fn, on_frame_processed, keyframe_indices, kf_only_buffer_pruning


# =====================================================================
# Shared pose accumulation utilities
# =====================================================================

def _so3_log(R):
    """Logarithmic map SO(3) -> so(3). R: (*, 3, 3) -> (*, 3)."""
    cos_theta = 0.5 * (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0)
    cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    omega = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    sin_theta = torch.sin(theta)
    small = (theta.abs() < 1e-4)
    scale = torch.where(small, 0.5 * torch.ones_like(theta), theta / (2.0 * sin_theta))
    return scale.unsqueeze(-1) * omega


def _so3_exp(omega):
    """Exponential map so(3) -> SO(3). omega: (*, 3) -> (*, 3, 3). Rodrigues formula."""
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = omega / theta
    K = torch.zeros(omega.shape[:-1] + (3, 3), device=omega.device, dtype=omega.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def _pgo_on_window(pose_history, constraints, window_start, window_end, n_iter=10):
    """Decoupled PGO: SO(3) rotation averaging + translation least-squares.

    Fixes over coupled SE(3) PGO:
    (1) R and t optimized separately — no scale coupling,
    (2) SO(3) Jacobian J=-I is exact for right perturbation,
    (3) Consistent convention: right perturbation + right multiplication,
    (4) clipped_inv_0.3 weight: w = max(1/|j-ref|, 0.3) balances
        nearby edge accuracy with long-range drift correction.

    Phase 1 — SO(3) Gauss-Newton (iterative):
        error  e = log(R_j^T @ R_ref @ R_rel^T),  J_j = -I
    Phase 2 — Translation weighted least-squares (linear, 1 step):
        error  e = t_j - (t_ref - R_ref_opt @ R_rel^T @ t_rel),  J_j=+I, J_ref=-I
    """
    W = window_end - window_start
    if W <= 0:
        return
    B = pose_history[0].shape[0]

    relevant = [(j, r, rp) for j, r, rp in constraints
                if window_start <= j < window_end and 0 <= r < window_end]
    if not relevant:
        return

    _eye3 = torch.eye(3)

    # ── Phase 1: SO(3) rotation averaging ──
    for _ in range(n_iter):
        H = torch.zeros(3 * W, 3 * W)
        b = torch.zeros(3 * W)
        H[:3, :3] += 1e4 * _eye3  # anchor first pose in window

        for (j, ref_idx, rel_pose) in relevant:
            R_ref = pose_history[ref_idx][:, :3, :3].float()
            R_j = pose_history[j][:, :3, :3].float()
            R_target = torch.bmm(R_ref, rel_pose[:, :3, :3].float().transpose(-1, -2))
            R_err = torch.bmm(R_j.transpose(-1, -2), R_target)
            e = _so3_log(R_err).mean(0)  # (3,)

            w = max(1.0 / max(abs(j - ref_idx), 1), 0.3)
            js = (j - window_start) * 3

            H[js:js+3, js:js+3] += w * _eye3
            b[js:js+3] += w * e

            if ref_idx >= window_start:
                rs = (ref_idx - window_start) * 3
                H[rs:rs+3, rs:rs+3] += w * _eye3
                H[js:js+3, rs:rs+3] -= w * _eye3
                H[rs:rs+3, js:js+3] -= w * _eye3
                b[rs:rs+3] -= w * e

        H.diagonal().add_(1e-6)
        delta = torch.linalg.solve(H, b)

        for wi in range(W):
            d = delta[wi*3:(wi+1)*3]
            dR = _so3_exp(d.unsqueeze(0).expand(B, -1))
            idx = window_start + wi
            pose_history[idx] = pose_history[idx].clone()
            pose_history[idx][:, :3, :3] = torch.bmm(
                pose_history[idx][:, :3, :3].clone(), dR
            )

    # ── Phase 2: Translation least-squares (linear, one-shot) ──
    for bi in range(B):
        Ht = torch.zeros(3 * W, 3 * W)
        bt = torch.zeros(3 * W)
        Ht[:3, :3] += 1e4 * _eye3  # anchor

        for (j, ref_idx, rel_pose) in relevant:
            R_ref = pose_history[ref_idx][bi, :3, :3].float()
            t_ref = pose_history[ref_idx][bi, :3, 3].float()
            t_j = pose_history[j][bi, :3, 3].float()
            R_rel_T = rel_pose[bi, :3, :3].float().T
            t_rel = rel_pose[bi, :3, 3].float()

            t_target = t_ref - R_ref @ (R_rel_T @ t_rel)
            et = t_j - t_target

            w = max(1.0 / max(abs(j - ref_idx), 1), 0.3)
            js = (j - window_start) * 3

            Ht[js:js+3, js:js+3] += w * _eye3
            bt[js:js+3] -= w * et

            if ref_idx >= window_start:
                rs = (ref_idx - window_start) * 3
                Ht[rs:rs+3, rs:rs+3] += w * _eye3
                Ht[js:js+3, rs:rs+3] -= w * _eye3
                Ht[rs:rs+3, js:js+3] -= w * _eye3
                bt[rs:rs+3] += w * et

        Ht.diagonal().add_(1e-6)
        dt = torch.linalg.solve(Ht, bt)

        for wi in range(W):
            d = dt[wi*3:(wi+1)*3]
            pose_history[window_start + wi][bi, :3, 3] += d


def accumulate_poses(predictions, views=None, use_relative_pose=None, reset_mask=None, skip_pgo=False):
    """Accumulate camera poses from model predictions.

    Supports three modes:
    - use_relative_pose=True: Relative pose accumulation + sliding-window PGO
    - use_relative_pose=False: Absolute camera_pose via pose_encoding_to_camera
    - use_relative_pose=None: Auto-detect (use relative if available)

    Args:
        predictions: list of prediction dicts from model output
        views: list of view dicts (used for reset detection if reset_mask is None)
        use_relative_pose: bool or None for auto-detect
        reset_mask: optional bool tensor of shape (N,) indicating reset frames

    Returns:
        list of (B, 4, 4) tensors, one per frame (c2w poses)
    """
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import matrix_cumprod

    if len(predictions) == 0:
        return []

    # Build reset_mask from views if not provided
    if reset_mask is None and views is not None:
        has_reset_key = len(views) > 0 and "reset" in views[0]
        if has_reset_key:
            reset_mask = torch.cat([view["reset"] for view in views], 0)
        else:
            reset_mask = torch.zeros(len(views), dtype=torch.bool)
    elif reset_mask is None:
        reset_mask = torch.zeros(len(predictions), dtype=torch.bool)

    # Auto-detect relative pose availability
    if use_relative_pose is None:
        use_relative_pose = any(
            ("relative_pose" in pred and pred.get("relative_pose") is not None)
            or ("relative_poses" in pred and pred.get("relative_poses") is not None)
            for pred in predictions
        )
    elif use_relative_pose:
        has_rel = any(
            ("relative_pose" in pred and pred.get("relative_pose") is not None)
            or ("relative_poses" in pred and pred.get("relative_poses") is not None)
            for pred in predictions
        )
        if not has_rel:
            print(
                "Warning: use_relative_pose=True but relative_pose not found in model output. "
                "Falling back to camera_pose."
            )
            use_relative_pose = False

    if use_relative_pose:
        pose_history = []
        constraint_buffer = []

        # --- Step 1: Chain accumulation + collect constraints ---
        for i, pred in enumerate(predictions):
            if i == 0:
                B = pred["pts3d_in_self_view"].shape[0]
                curr_T_c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
            else:
                if "relative_poses" in pred and pred["relative_poses"] is not None:
                    rel_poses = pred["relative_poses"].clone().cpu()
                    K = rel_poses.shape[1]
                    ref_indices = pred.get("ref_frame_indices")
                    nearest_ref_idx = None
                    for k in range(K):
                        if ref_indices is not None and k < len(ref_indices):
                            ref_idx = ref_indices[k]
                        else:
                            ref_idx = i - k - 1
                        if ref_idx < 0 or ref_idx >= len(pose_history):
                            continue
                        constraint_buffer.append((i, ref_idx, rel_poses[:, k]))
                        if nearest_ref_idx is None:
                            nearest_ref_idx = ref_idx
                            nearest_rel = rel_poses[:, k]
                    if nearest_ref_idx is not None:
                        T_rel_inv = torch.inverse(nearest_rel.float())
                        curr_T_c2w = torch.bmm(pose_history[nearest_ref_idx], T_rel_inv)
                    else:
                        curr_T_c2w = pose_history[-1].clone()
                elif "relative_pose" in pred and pred["relative_pose"] is not None:
                    T_rel = pred["relative_pose"].clone().cpu()
                    T_rel_inv = torch.inverse(T_rel.float())
                    curr_T_c2w = torch.bmm(pose_history[-1], T_rel_inv)
                    constraint_buffer.append((i, len(pose_history) - 1, T_rel))
                else:
                    curr_T_c2w = pose_history[-1].clone()

            pose_history.append(curr_T_c2w.clone())

        # --- Step 2: Global PGO over entire sequence ---
        N = len(pose_history)
        if not skip_pgo:
            _pgo_on_window(pose_history, constraint_buffer, 1, N, n_iter=10)

        pr_poses = [pose_history[i].clone() for i in range(N)]

        num_reset_frames = reset_mask.sum().item() if reset_mask.any() else 0
        pgo_status = "skipped" if skip_pgo else f"{len(constraint_buffer)} edges"
        print(
            f"Using {'chain-only' if skip_pgo else 'global PGO'} ({N} frames, {pgo_status}, "
            f"{num_reset_frames} resets)"
        )
    else:
        # Absolute camera_pose path
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in predictions
        ]

        # Handle reset: accumulate poses across reset boundaries
        if reset_mask.any():
            pr_poses_cat = torch.cat(pr_poses, 0)
            identity = torch.eye(4, device=pr_poses_cat.device)
            reset_poses = torch.where(
                reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses_cat, identity
            )
            cumulative_bases = matrix_cumprod(reset_poses)
            shifted_bases = torch.cat(
                [identity.unsqueeze(0), cumulative_bases[:-1]], dim=0
            )
            pr_poses_cat = torch.einsum("bij,bjk->bik", shifted_bases, pr_poses_cat)
            pr_poses = list(pr_poses_cat.unsqueeze(1).unbind(0))

        print(f"Using direct camera_pose ({len(pr_poses)} frames)")

    return pr_poses
