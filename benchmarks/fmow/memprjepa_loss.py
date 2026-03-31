"""MEMPrJEPA loss: independent SIGReg per encoder + cross-modal retrieval probes.

Each encoder is trained independently with SIGReg only (no cross-modal
coupling between encoders).  Two retrieval probes are trained with MSE to
bridge the modalities; encoder outputs are stop-grad'd for the probe losses
so the probes do not influence the encoders.

Gradient flow summary:
  sigreg_rgb  -> encoder_rgb, projector_rgb  (only)
  sigreg_ms   -> encoder_ms,  projector_ms   (only)
  probe_rgb2ms_loss -> probe_rgb2ms  (only, encoder outputs are detached)
  probe_ms2rgb_loss -> probe_ms2rgb  (only, encoder outputs are detached)
"""

import torch.nn.functional as F
import torch.nn as nn

from stable_pretraining.methods.lejepa import SlicedEppsPulley


class MEMPrJEPALoss(nn.Module):
    """MEMPrJEPA loss.

    Combines two objectives:

    1. **SIGReg** (per encoder, independent): sliced Epps-Pulley
       goodness-of-fit test that pushes each modality's projected embeddings
       toward an isotropic Gaussian.  No coupling between modalities.
    2. **Retrieval probe losses**: MSE between each probe's output and the
       target modality's embedding.  Encoder outputs are stop-grad'd so only
       the probe weights receive gradient.

    Args:
        num_slices: Number of random 1-D projections for SIGReg.
        t_max: EP integration upper bound.
        n_points: EP quadrature nodes.
        probe_scale: Weight on probe MSE losses relative to SIGReg.
    """

    def __init__(
        self,
        num_slices: int = 256,
        t_max: float = 5.0,
        n_points: int = 17,
        probe_scale: float = 1.0,
    ):
        super().__init__()
        self.probe_scale = probe_scale
        self.sigreg = SlicedEppsPulley(
            num_slices=num_slices, t_max=t_max, n_points=n_points
        )

    def forward(self, proj_rgb, proj_ms, pred_ms, pred_rgb, tgt_ms, tgt_rgb):
        """Compute the combined loss.

        Args:
            proj_rgb:  RGB projections ``(V, B, K)`` – input to SIGReg.
            proj_ms:   MS  projections ``(V, B, K)`` – input to SIGReg.
            pred_ms:   ``probe_rgb2ms`` output ``(B, D)`` – prediction of MS
                       embedding from RGB embedding.
            pred_rgb:  ``probe_ms2rgb`` output ``(B, D)`` – prediction of RGB
                       embedding from MS embedding.
            tgt_ms:    Stop-grad MS  embeddings ``(B, D)`` – probe_rgb2ms target.
            tgt_rgb:   Stop-grad RGB embeddings ``(B, D)`` – probe_ms2rgb target.

        Returns:
            Tuple of ``(total_loss, loss_dict)``.
        """
        # Independent SIGReg – no gradient coupling between modalities
        sigreg_rgb = self.sigreg(proj_rgb.reshape(-1, proj_rgb.size(-1)))
        sigreg_ms = self.sigreg(proj_ms.reshape(-1, proj_ms.size(-1)))

        # Probe MSE losses – tgt_* are already detached in the forward fn
        probe_rgb2ms_loss = F.mse_loss(pred_ms, tgt_ms)
        probe_ms2rgb_loss = F.mse_loss(pred_rgb, tgt_rgb)

        total = (
            sigreg_rgb
            + sigreg_ms
            + self.probe_scale * (probe_rgb2ms_loss + probe_ms2rgb_loss)
        )

        loss_dict = {
            "sigreg/rgb": sigreg_rgb.detach(),
            "sigreg/ms": sigreg_ms.detach(),
            "probe/rgb2ms": probe_rgb2ms_loss.detach(),
            "probe/ms2rgb": probe_ms2rgb_loss.detach(),
        }

        return total, loss_dict
