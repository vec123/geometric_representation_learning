import torch
import torch.nn.functional as F

def kl_divergence_loss(mean, log_var):
    """Computes the KL divergence between N(mean, exp(log_var)) and N(0, 1)."""
    # -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).mean()

def geometric_clustering_loss(logits, edge_index, smoothness_weight=1.0, balance_weight=0.1, entropy_weight=0.1):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    # Entropy
    entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
    
    # Smoothness Loss
    row, col = edge_index
    smoothness_loss = torch.mean((probs[row] - probs[col]).pow(2))
    
    # Balance Loss
    mean_probs = torch.mean(probs, dim=0)
    balance_loss = torch.mean((mean_probs - 0.5)**2)
    
    return (smoothness_weight * smoothness_loss + 
            balance_weight * balance_loss + 
            entropy_weight * entropy)

def laplacian_loss(pred_pos):
    """pred_pos: [Batch, N_samples, 3] where N_samples is square."""
    b, n, c = pred_pos.shape
    grid_size = int(n**0.5)
    
    grid = pred_pos.view(b, grid_size, grid_size, c)
    
    # Finite difference
    diff_u = grid[:, 1:, :, :] - grid[:, :-1, :, :]
    diff_v = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    
    return torch.mean(diff_u.pow(2)) + torch.mean(diff_v.pow(2))

def chamfer_loss(pred_pos, target_pos, target_mask):
    """
    Highly performant Chamfer Distance using torch.cdist.
    pred_pos: [B, N, 3]
    target_pos: [B, M, 3]
    target_mask: [B, M] (bool or float)
    """
    # 1. Compute pairwise distance matrix [B, N, M]
    # cdist is significantly faster than manual broadcasting
    dist_sq = torch.cdist(pred_pos, target_pos, p=2).pow(2)
    
    # 2. Masking: Use a large value for padded targets
    dist_sq = dist_sq.masked_fill(~target_mask.unsqueeze(1), 1e6)
    
    # 3. Term 1: Min distance from pred to nearest target
    term1 = dist_sq.min(dim=2)[0].mean()
    
    # 4. Term 2: Min distance from target to nearest pred
    # Use masked_fill to ignore the padded targets in the min calculation
    dist_t_to_p = dist_sq.min(dim=1)[0]
    term2 = (dist_t_to_p * target_mask).sum() / (target_mask.sum() + 1e-8)
    
    return term1 + term2

def combined_surface_loss(pred_pos, target_pos, target_mask, laplacian_weight=0.1):
    c_loss = chamfer_loss(pred_pos, target_pos, target_mask)
    l_loss = laplacian_loss(pred_pos)
    return c_loss + (laplacian_weight * l_loss)