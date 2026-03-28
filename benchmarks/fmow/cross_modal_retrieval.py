"""Cross-modal image retrieval callback.

Evaluates retrieval quality across four directions:
  S1->S1 (RGB->RGB), S1->S2 (RGB->MS), S2->S2 (MS->MS), S2->S1 (MS->RGB)

Uses label-based relevance: a gallery item is relevant if it shares the
same class as the query.  All validation samples serve as both queries and
gallery (self-matches excluded).
"""

import types

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging


def _wrap_validation_step(fn, input_rgb, input_ms, name):
    """Wrap the module's ``validation_step`` to cache dual-modality embeddings."""

    def wrapped(self, batch, batch_idx, fn=fn, name=name):
        batch = fn(batch, batch_idx)

        with torch.no_grad():
            emb_rgb = F.normalize(batch[input_rgb].float(), dim=1, p=2)
            emb_ms = F.normalize(batch[input_ms].float(), dim=1, p=2)

        idx = self.all_gather(batch["sample_idx"])
        emb_rgb = self.all_gather(emb_rgb)
        emb_ms = self.all_gather(emb_ms)

        if self.local_rank == 0:
            self._xmodal_embeds_rgb[idx] = emb_rgb
            self._xmodal_embeds_ms[idx] = emb_ms

        return batch

    return wrapped


def _retrieval_f1_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k):
    """Compute mean retrieval F1@k using cosine similarity.

    Args:
        query_emb: ``(Q, D)`` L2-normalised query embeddings.
        gallery_emb: ``(G, D)`` L2-normalised gallery embeddings.
        query_labels: ``(Q,)`` integer labels.
        gallery_labels: ``(G,)`` integer labels.
        k: Number of items to retrieve.

    Returns:
        Mean F1@k (float).
    """
    scores = query_emb @ gallery_emb.T  # (Q, G)
    _, topk_idx = scores.topk(k, dim=1)  # (Q, k)

    precisions = []
    recalls = []

    # Pre-compute class counts in gallery
    class_counts = {}
    gl = gallery_labels.cpu().numpy()
    for lbl in gl:
        class_counts[int(lbl)] = class_counts.get(int(lbl), 0) + 1

    ql = query_labels.cpu().numpy()
    topk_labels = gallery_labels[topk_idx]  # (Q, k)

    for i in range(len(query_labels)):
        q_lbl = int(ql[i])
        relevant = (topk_labels[i] == q_lbl).sum().item()
        total_relevant = class_counts.get(q_lbl, 1)
        precisions.append(relevant / k)
        recalls.append(relevant / total_relevant)

    mean_p = float(np.mean(precisions))
    mean_r = float(np.mean(recalls))
    return 2 * mean_p * mean_r / max(mean_p + mean_r, 1e-8)


class CrossModalRetrieval(Callback):
    """Cross-modal image retrieval evaluator.

    Computes retrieval F1@k for four modality combinations on the
    validation set:

    - **S1->S1**: RGB query vs RGB gallery
    - **S1->S2**: RGB query vs MS gallery
    - **S2->S2**: MS query vs MS gallery
    - **S2->S1**: MS query vs RGB gallery

    Relevance is label-based: a gallery item is relevant if it shares
    the same class as the query.

    Args:
        pl_module: The ``spt.Module`` instance.
        name: Unique callback name.
        input_rgb: Key in batch/outputs for RGB embeddings.
        input_ms: Key in batch/outputs for MS embeddings.
        features_dim: Embedding dimensionality.
        top_k: List of k values for F1@k evaluation.
    """

    def __init__(
        self,
        pl_module,
        name: str,
        input_rgb: str = "embedding_rgb",
        input_ms: str = "embedding_ms",
        features_dim: int = 1024,
        top_k: list[int] | tuple[int, ...] = (5, 10),
    ):
        super().__init__()
        logging.info(f"Setting up CrossModalRetrieval callback ({name})")
        self.name = name
        self.features_dim = features_dim
        self.top_k = list(top_k)

        # Wrap the validation step to cache embeddings
        fn = _wrap_validation_step(
            pl_module.validation_step, input_rgb, input_ms, name
        )
        pl_module.validation_step = types.MethodType(fn, pl_module)

        # Placeholders set in on_validation_epoch_start
        pl_module._xmodal_embeds_rgb = None
        pl_module._xmodal_embeds_ms = None

    @property
    def state_key(self) -> str:
        return f"CrossModalRetrieval[name={self.name}]"

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        val_dataset = trainer.datamodule.val.dataset
        n = len(val_dataset)
        if pl_module.local_rank == 0:
            device = pl_module.device
            pl_module._xmodal_embeds_rgb = torch.zeros(
                (n, self.features_dim), device=device
            )
            pl_module._xmodal_embeds_ms = torch.zeros(
                (n, self.features_dim), device=device
            )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if pl_module.local_rank != 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        logging.info(f"Computing cross-modal retrieval metrics ({self.name})")

        emb_rgb = pl_module._xmodal_embeds_rgb
        emb_ms = pl_module._xmodal_embeds_ms

        # Collect labels from the dataset
        val_dataset = trainer.datamodule.val.dataset
        labels = torch.tensor(
            [val_dataset[i]["label"] for i in range(len(val_dataset))],
            device=emb_rgb.device,
        )

        logs = {}
        directions = {
            "s1_to_s1": (emb_rgb, emb_rgb),
            "s1_to_s2": (emb_rgb, emb_ms),
            "s2_to_s2": (emb_ms, emb_ms),
            "s2_to_s1": (emb_ms, emb_rgb),
        }

        for direction_name, (query_emb, gallery_emb) in directions.items():
            for k in self.top_k:
                f1 = _retrieval_f1_at_k(
                    query_emb, gallery_emb, labels, labels, k
                )
                logs[f"eval/{self.name}_{direction_name}_f1@{k}"] = f1
                logging.info(
                    f"  {direction_name} F1@{k}: {f1:.4f}"
                )

        pl_module.log_dict(logs, on_epoch=True, rank_zero_only=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Free memory
        pl_module._xmodal_embeds_rgb = None
        pl_module._xmodal_embeds_ms = None
