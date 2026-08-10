
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import time
import io
import base64
from torch.utils.data import Dataset, DataLoader, Sampler
import re



import random
import math
import os
from dataclasses import dataclass, asdict

import numpy as np

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def set_seed(seed, deterministic=False):
    """
    Seed every RNG the optimisation touches.

    The released code seeds nothing, so patch initialisation, the per-step
    transform schedule (angle/scale/placement) and the target-embedding noise all
    vary run to run and no delta can be separated from run-to-run variance.
    Seeding those three is what makes an n>1 study possible.

    `deterministic` additionally asks torch for deterministic kernels.  Note that
    grid_sample's backward pass -- which every training step goes through, via
    apply_random_patch -- has no deterministic CUDA implementation, so this will
    RAISE rather than silently succeed.  It is here for CPU-side checks, not for
    training runs: seeding alone removes the dominant variance, but bitwise
    reproducibility on GPU is not achievable for this model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class AblationConfig:
    """
    Every knob the paper's analysis sections vary, in one place.

    Defaults reproduce the paper's Table 4 exactly, so an unconfigured run is the
    paper's configuration.  The adapters read this instead of the constants that
    used to be hardcoded at their call sites, which is what lets E5/E6 sweep them
    from job-script flags rather than by editing source between runs.
    """
    # Semantic loss (paper: "Effect of Semantic Loss")
    loss: str = "semantic"          # 'semantic' (Eq. 8) | 'ce' (cross-entropy baseline)
    loss_mode: str = "attention"    # 'token' (exact match) | 'attention' (relaxed)
    tau: float = 0.5                # Table 4 attention temperature
    embed_noise: float = 1e-4       # Table 4 epsilon
    pos_alpha: float = 0.01         # positional-encoding weight

    # Input transformations (paper: "Effect of Transformation and Regularisation")
    transforms: bool = True
    scale_range: tuple = (0.8, 1.2)         # Table 4
    rotation_range: tuple = (-15.0, 15.0)   # Table 4

    def as_dict(self):
        return asdict(self)


def apply_random_patch(image_tensor, patch, verbose=False, scale_range=(0.8, 1.2), rotation_range=(-15, 15),
                       transforms_enabled=True):
    """
    Applies a differentiably transformed patch to a clone of the input image.

    Args:
        image_tensor (Tensor): [C, H, W] image tensor.
        patch (Tensor): [C, h, w] patch tensor.
        scale_range (tuple): (min_scale, max_scale) for random scaling.
        rotation_range (tuple): (min_deg, max_deg) for random rotation.
        transforms_enabled (bool): when False, composite the patch unscaled,
            unrotated and centred -- the paper's "no constraints" ablation cell.

    Returns:
        Tensor: Patched image tensor [C, H, W].
    """
    C, H, W = image_tensor.shape
    device = image_tensor.device
    _, ph, pw = patch.shape

    # Random scale and rotation
    if transforms_enabled:
        angle_deg = random.uniform(*rotation_range)
        scale = random.uniform(*scale_range)
    else:
        angle_deg = 0.0
        scale = 1.0
    angle_rad = math.radians(angle_deg)

    # Estimate the largest possible size after rotation + scale
    
    # diag = math.sqrt(ph ** 2 + pw ** 2)
    # max_dim = int(scale * diag) + 1

    # out_h, out_w = max_dim, max_dim
    
    # Compute minimal bounding rectangle after rotation and scaling
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    
    out_w = int(scale * (pw * cos_a + ph * sin_a)) + 1
    out_h = int(scale * (pw * sin_a + ph * cos_a)) + 1

    # Pad patch to center it in a larger canvas
    pad_y = (out_h - ph) // 2
    pad_x = (out_w - pw) // 2
    padded_patch = F.pad(patch, (pad_x, pad_x, pad_y, pad_y))  # left, right, top, bottom
    padded_patch = padded_patch.unsqueeze(0)  # [1, C, H_pad, W_pad]

    # Affine transform relative to padded_patch size
    # It seems that larger scale leads to small patch here?
    theta = torch.tensor([
        [(1/scale) * math.cos(angle_rad), -(1/scale) * math.sin(angle_rad), 0],
        [(1/scale) * math.sin(angle_rad),  (1/scale) * math.cos(angle_rad), 0]
    ], dtype=torch.float, device=device).unsqueeze(0)

    # Grid size matches padded_patch
    grid = F.affine_grid(theta, size=padded_patch.size(), align_corners=False)

    # Apply the transformation
    transformed_patch = F.grid_sample(
        padded_patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    # Create mask
    patch_mask = (transformed_patch.abs().sum(dim=1, keepdim=True) > 1e-5).float()

    _, _, tph, tpw = transformed_patch.shape

    # Random placement (ensure it fits)
    if tph > H or tpw > W:
        raise ValueError("Transformed patch is too large for the image. Consider reducing scale_range.")
    if transforms_enabled:
        top = random.randint(0, H - tph)
        left = random.randint(0, W - tpw)
    else:
        top = (H - tph) // 2
        left = (W - tpw) // 2

    # Region from image
    region = image_tensor[:, top:top+tph, left:left+tpw].unsqueeze(0)  # [1, C, tph, tpw]

    # Blend
    blended = patch_mask * transformed_patch + (1 - patch_mask) * region

    # Insert back
    patched_image = image_tensor.clone()
    patched_image[:, top:top+tph, left:left+tpw] = blended[0]

    if verbose:
        print(f"Angle: {angle_deg:.2f}°, Scale: {scale:.2f}, Top: {top}, Left: {left}, Size: {tph}x{tpw}")

    return patched_image



def project_patch(patch, scale, shift):
    """
    Project patch values using a deterministic affine transformation: p -> p*scale + shift.

    Args:
        patch (torch.Tensor): Patch tensor in [C,H,W], values in [0,1].
        scale (float or torch.Tensor): Multiplicative factor (amplitude/contrast), can be per-channel.
        shift (float or torch.Tensor): Additive factor (baseline/mean), can be per-channel.

    Returns:
        torch.Tensor: Projected patch.
    """
    C = patch.shape[0]

    # Convert to tensor per channel if needed
    if not torch.is_tensor(scale):
        scale = torch.tensor([scale]*C, device=patch.device)
    if not torch.is_tensor(shift):
        shift = torch.tensor([shift]*C, device=patch.device)

    scale = scale.view(-1,1,1)
    shift = shift.view(-1,1,1)

    return patch * scale + shift


def sinusoidal_positional_encoding(seq_len, dim):
    """
    Create sinusoidal positional encoding [1, seq_len, dim]
    """
    position = torch.arange(0, seq_len, dtype=torch.bfloat16, device='cuda').unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, dim, 2, device='cuda').bfloat16() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim,  device='cuda', dtype=torch.bfloat16)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, seq_len, dim]



def semantic_similarity_loss(logits, labels, embedding_matrix, weights = None, mode="token", ignore_index=-100, verbose=False,
                             tau=0.5, embed_noise=1e-4, pos_alpha=0.01):
        """
        Compute semantic similarity loss between predicted token distributions and target tokens.

        `mode`, `tau`, `embed_noise` and `pos_alpha` were hardcoded here; they are
        parameters so the paper's "Effect of Semantic Loss" ablation (CE / tau=0 /
        tau=0.5 / tau->inf) can be swept from the command line.  Defaults are the
        paper's Table 4 values.

        Returns:
            scalar loss
        """
        # Shift for next-token prediction
        logits = logits[:, :-1, :]         # [B, T-1, V]
        labels = labels[:, 1:]             # [B, T-1]
        B, T, V = logits.size()
        D = embedding_matrix.size(1)
        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)  # [B, T-1, V]
    
        # Expected embeddings for each position
        expected_embeddings = probs @ embedding_matrix  # [B, T-1, d]
    
        # Target embeddings
        target_embeddings = embedding_matrix[labels.clamp(min=0)]  # clamp to avoid -100 indexing
    
        # Mask for valid tokens
        mask = (labels != ignore_index).float()  # [B, T-1]
    
        if mode == "token":
            # Compute cosine similarity for each position
            sim = F.cosine_similarity(expected_embeddings, target_embeddings, dim=-1)  # [B, T-1]
            sim = sim * mask  # zero out ignored positions
            # print(sim)
            # print(mask)
            loss = ((1 - sim) * mask.float()).sum() / mask.sum()
            #print(loss)        

        elif mode == "attention":
            """
            Attention-based semantic similarity:
            - Each predicted embedding attends to all target embeddings.
            - Uses dot-product attention (parameter-free) with temperature scaling.
            """

            # noise scale
            epsilon = embed_noise
            noise = torch.randn_like(expected_embeddings) * epsilon
            expected_embeddings = expected_embeddings + noise

            # tau controls the distribution; supplied by the caller

            # Mask for valid tokens
            mask = (labels != ignore_index).float()  # [B, T]

            # Compute positional encoding and add to embeddings
            pos_enc = sinusoidal_positional_encoding(T, D)
            alpha = pos_alpha  # try values in [0.01, 1.0]

            expected_pos = expected_embeddings + alpha * pos_enc
            target_pos   = target_embeddings   + alpha * pos_enc
        
            # Normalize embeddings for cosine-like similarity
            pred_norm = F.normalize(expected_pos, dim=-1)  # [B, T, D]
            tgt_norm = F.normalize(target_pos, dim=-1)     # [B, T, D]
        
            # Compute attention scores: [B, T, T]
            attn_scores = torch.bmm(pred_norm, tgt_norm.transpose(1, 2))  # dot-product sim
            attn_scores = attn_scores / tau  # apply temperature
            
            # Make a causal mask: shape [T, T], upper triangular = True
            T = labels.size(1)
            causal_mask = torch.triu(torch.ones((T, T), device=labels.device)).bool()  # lower triangle
            # Expand to batch: [B, T, T]
            causal_mask = causal_mask.unsqueeze(0).expand(labels.size(0), -1, -1)
            
            # Combine causal mask with padding mask for target
            # padding_mask: True = valid token, False = pad
            padding_mask = mask.bool().unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
            
            # Final mask: True = keep, False = mask
            final_mask = causal_mask & padding_mask
            
            # Apply -inf to masked positions
            attn_scores = attn_scores.masked_fill(~final_mask, float('-inf'))
            
            # Replace -inf rows with 0 temporarily to avoid NaN
            row_has_valid = final_mask.sum(dim=-1) > 0  # [B, T]
            attn_scores[~row_has_valid] = 0.0
            
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # Compute attended target representations
            attended_targets = torch.bmm(attn_weights, target_embeddings)  # [B, T, D]
            
            # Cosine similarity between prediction and attended target
            sim = F.cosine_similarity(expected_embeddings, attended_targets, dim=-1) # [B, T]
            #print(sim)
            # Apply mask and compute mean loss
            sim = sim * mask
            loss = ((1 - sim) * mask).sum() / mask.sum()

            if verbose:
                batch_idx = 0  # first element in batch
                # Find valid predicted tokens (rows)
                valid_pred_idx = (labels[batch_idx] != -100).nonzero(as_tuple=True)[0]
                # Find valid target tokens (columns)
                valid_tgt_idx = (labels[batch_idx] != -100).nonzero(as_tuple=True)[0]
            
                # Extract the submatrix corresponding to valid tokens
                attn_matrix_valid = attn_weights[batch_idx][valid_pred_idx][:, valid_tgt_idx]
            
                # Convert to float32 for printing / rounding
                attn_matrix_valid = attn_matrix_valid.detach().to(torch.float32)
            
                print(f"\nAttention matrix")
                print(attn_matrix_valid)
                print(f"\nSimilarity")
                print(sim)
                print(f"tau:{tau}")
                print(f"alpha:{alpha}")
                print(f"epsilon:{epsilon}")
            
        else:
            raise ValueError("mode must be 'token' or 'attention'")
    
        return loss