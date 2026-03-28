"""MMLeJEPA loss: SIGReg + invariance + cross-modal alignment.

Reuses ``SlicedEppsPulley`` from ``stable_pretraining.methods.lejepa``
for the SIGReg regularisation term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.methods.lejepa import SlicedEppsPulley


class MMLeJEPALoss(nn.Module):
    """Multi-modal LeJEPA loss.

    Combines three objectives:

    1. **SIGReg** (per modality): sliced Epps-Pulley goodness-of-fit test
       that pushes projected embeddings toward an isotropic Gaussian.
    2. **Invariance** (per modality): MSE between each view's projection
       and the modality-wise center, enforcing view consistency.
    3. **Cross-modal alignment**: cosine similarity between the RGB and MS
       projection centres, pulling the two modalities together.

    Args:
        lamb: Trade-off weight.  ``loss = lamb * sigreg + (1-lamb) * invariance``.
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

    def forward(self, proj_rgb, proj_ms):
        """Compute the combined loss.

        Args:
            proj_rgb: Projected RGB embeddings ``(V, B, K)``.
            proj_ms:  Projected MS embeddings  ``(V, B, K)``.

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where *loss_dict* contains
            individual components for logging.
        """
        # --- SIGReg: push each modality toward isotropic Gaussian ---
        sigreg_rgb = self.sigreg(proj_rgb.reshape(-1, proj_rgb.size(-1)))
        sigreg_ms = self.sigreg(proj_ms.reshape(-1, proj_ms.size(-1)))
        sigreg_loss = sigreg_rgb + sigreg_ms

        # --- Invariance: pull views to modality centre ---
        inv_rgb = (proj_rgb.mean(0) - proj_rgb).square().mean()
        inv_ms = (proj_ms.mean(0) - proj_ms).square().mean()
        inv_loss = inv_rgb + inv_ms

        # --- Cross-modal: pull RGB and MS centres together ---
        cross_loss = F.cosine_similarity(
            proj_rgb.mean(0), proj_ms.mean(0), dim=-1
        ).mean()

        total = self.lamb * sigreg_loss + (1 - self.lamb) * inv_loss - cross_loss

        loss_dict = {
            "sigreg/rgb": sigreg_rgb.detach(),
            "sigreg/ms": sigreg_ms.detach(),
            "sigreg/total": sigreg_loss.detach(),
            "invariance/rgb": inv_rgb.detach(),
            "invariance/ms": inv_ms.detach(),
            "invariance/total": inv_loss.detach(),
            "cross_modal/cosine": cross_loss.detach(),
        }

        return total, loss_dict
