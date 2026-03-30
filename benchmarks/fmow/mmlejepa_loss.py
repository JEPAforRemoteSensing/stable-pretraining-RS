"""MMLeJEPA loss: per-modality SIGReg + invariance (no cross-modal term).

Each encoder is regularised independently; cross-modal alignment is handled
separately by the retrieval probes (MSE in the forward function).

Reuses ``SlicedEppsPulley`` from ``stable_pretraining.methods.lejepa``.
"""

import torch
import torch.nn as nn

from stable_pretraining.methods.lejepa import SlicedEppsPulley


class MMLeJEPALoss(nn.Module):
    """Per-modality SIGReg + invariance loss.

    Each modality's projected embeddings are regularised independently:

    1. **SIGReg**: sliced Epps-Pulley test that pushes projections toward
       an isotropic Gaussian.
    2. **Invariance**: MSE between each view's projection and the
       modality-wise centre, enforcing view consistency.

    No cross-modal term — encoder gradients come only from this loss.
    Cross-modal alignment is achieved via the retrieval probes trained
    with MSE on detached encoder outputs.

    Args:
        lamb: ``loss = lamb * sigreg + (1-lamb) * invariance``.
        num_slices: Random 1-D projections for SIGReg.
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

    def forward(self, proj_rgb, proj_ms):
        """Compute the combined loss.

        Args:
            proj_rgb: Projected RGB embeddings ``(V, B, K)``.
            proj_ms:  Projected MS embeddings  ``(V, B, K)``.

        Returns:
            Tuple of ``(total_loss, loss_dict)``.
        """
        # --- SIGReg: push each modality toward isotropic Gaussian ---
        sigreg_rgb = self.sigreg(proj_rgb.reshape(-1, proj_rgb.size(-1)))
        sigreg_ms = self.sigreg(proj_ms.reshape(-1, proj_ms.size(-1)))
        sigreg_loss = sigreg_rgb + sigreg_ms

        # --- Invariance: pull views to their modality-wise centre ---
        inv_rgb = (proj_rgb.mean(0) - proj_rgb).square().mean()
        inv_ms = (proj_ms.mean(0) - proj_ms).square().mean()
        inv_loss = inv_rgb + inv_ms

        total = self.lamb * sigreg_loss + (1 - self.lamb) * inv_loss

        loss_dict = {
            "sigreg/rgb": sigreg_rgb.detach(),
            "sigreg/ms": sigreg_ms.detach(),
            "sigreg/total": sigreg_loss.detach(),
            "invariance/rgb": inv_rgb.detach(),
            "invariance/ms": inv_ms.detach(),
            "invariance/total": inv_loss.detach(),
        }

        return total, loss_dict
