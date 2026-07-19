from typing import Dict, List, Optional
import torch
from src.learning.config.models import LossConfig, LossTermConfig
from src.learning.losses import losses as loss_module

class LossComposer:
    """Compose losses from config at runtime."""
    
    AVAILABLE_LOSSES = {
        'recon': loss_module.combined_surface_loss,
        'kl': loss_module.kl_divergence_loss,
        'contrastive': loss_module.contrastive_alignment_loss,
        'chamfer': loss_module.chamfer_loss,
        'laplacian': loss_module.laplacian_loss,
    }
    
    def __init__(self, config: LossConfig):
        self.config = config
        self._validate()
    
    def _validate(self):
        """Check all loss names are available."""
        for term in self.config.terms:
            if term.name not in self.AVAILABLE_LOSSES:
                raise ValueError(f"Unknown loss: {term.name}")
    
    def compute(self, loss_dict: Dict[str, torch.Tensor]) -> tuple:
        """Compute total loss from individual loss terms.
        
        Args:
            loss_dict: {"recon": tensor, "kl": tensor, "contrastive": tensor, ...}
        
        Returns:
            (total_loss, loss_breakdown_dict)
        """
        total_loss = torch.zeros(1, device=list(loss_dict.values())[0].device)
        breakdown = {}
        
        for term in self.config.terms:
            if term.name not in loss_dict:
                if not term.kwargs.get('optional', False):
                    raise ValueError(f"Loss term '{term.name}' not in loss_dict")
                continue
            
            loss_val = loss_dict[term.name]
            weighted = term.weight * loss_val
            total_loss = total_loss + weighted
            breakdown[term.name] = weighted.item()
        
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss: {total_loss.item()}")
        
        return total_loss, breakdown
