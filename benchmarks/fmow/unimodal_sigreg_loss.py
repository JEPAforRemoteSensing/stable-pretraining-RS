"""Unimodal SIGReg loss for RGB-only JEPA training.

Combines SIGReg (sliced Epps-Pulley goodness-of-fit) with a multi-view
invariance objective — the same formulation as LeJEPA but for a single
modality:

    loss = lamb * sigreg(all_views)  +  (1 - lamb) * invariance(views, center)

Gradient flow:
    loss -> encoder_rgb, projector_rgb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.methods.lejepa import SlicedEppsPulley


class UnimodalSIGRegLoss(nn.Module):
    """Unimodal SIGReg + invariance loss.

    Args:
        lamb: Weight on SIGReg term; ``(1 - lamb)`` weights invariance.
        num_slices: Number of random 1-D projections for SIGReg.
        t_max: EP integration upper bound.
        n_points: EP quadrature nodes.
    """

    def __init__(
        self,
        lamb: float = 0.02,
        num_slices: int = 256,
        t_max: float = 5.0,
        n_points: int = 17,
    ):
        super().__init__()
        self.lamb = lamb
        self.sigreg = SlicedEppsPulley(
            num_slices=num_slices, t_max=t_max, n_points=n_points
        )

    def forward(self, proj: torch.Tensor):
        """Compute the combined loss.

        Args:
            proj: Projected embeddings of shape ``(V, B, K)`` where ``V`` is
                  the number of views, ``B`` is the batch size, and ``K`` is
                  the projection dimension.

        Returns:
            Tuple of ``(total_loss, loss_dict)``.
        """
        V, B, K = proj.shape

        # SIGReg: all views concatenated → (V*B, K)
        sigreg_loss = self.sigreg(proj.reshape(V * B, K))

        # Invariance: MSE between each view and the mean-view center
        center = proj.mean(dim=0, keepdim=True)  # (1, B, K)
        inv_loss = F.mse_loss(proj, center.expand_as(proj))

        total = self.lamb * sigreg_loss + (1.0 - self.lamb) * inv_loss

        loss_dict = {
            "sigreg/rgb": sigreg_loss.detach(),
            "inv/rgb": inv_loss.detach(),
        }

        return total, loss_dict
