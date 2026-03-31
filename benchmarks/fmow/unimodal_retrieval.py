"""Unimodal image retrieval callback for RGB-only JEPA.

Caches L2-normalised embeddings during validation and computes F1@k using
label-based relevance (same class label = relevant).  All validation samples
serve as both query and gallery.

This is a single-modality counterpart to CrossModalRetrieval.
"""

import types

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging


def _wrap_validation_step(fn, input_key, name):
    """Wrap ``validation_step`` to cache L2-normalised embeddings."""
    attr = f"_uniret_{name}_emb"
    attr_lbl = f"_uniret_{name}_lbl"

    def wrapped(
        self, batch, batch_idx, fn=fn,
        attr=attr, attr_lbl=attr_lbl, input_key=input_key,
    ):
        batch = fn(batch, batch_idx)

        with torch.no_grad():
            emb = F.normalize(batch[input_key].float(), dim=1)

        idx = self.all_gather(batch["sample_idx"])
        emb = self.all_gather(emb)
        lbl = self.all_gather(batch["label"])

        if self.local_rank == 0:
            getattr(self, attr)[idx] = emb
            getattr(self, attr_lbl)[idx] = lbl

        return batch

    return wrapped


def _retrieval_f1_at_k(emb, labels, k):
    """Mean F1@k using cosine similarity (inputs must be L2-normalised).

    Args:
        emb:    ``(N, D)`` L2-normalised embeddings (query == gallery).
        labels: ``(N,)`` integer class labels.
        k: Number of items to retrieve.

    Returns:
        Mean F1@k (float).
    """
    scores = emb @ emb.T  # (N, N)
    _, topk_idx = scores.topk(k, dim=1)  # (N, k)

    gl = labels.cpu().numpy()
    class_counts = {}
    for lbl in gl:
        class_counts[int(lbl)] = class_counts.get(int(lbl), 0) + 1

    topk_labels = labels[topk_idx]  # (N, k)
    precisions, recalls = [], []
    for i in range(len(labels)):
        q_lbl = int(gl[i])
        relevant = (topk_labels[i] == q_lbl).sum().item()
        total_relevant = class_counts.get(q_lbl, 1)
        precisions.append(relevant / k)
        recalls.append(relevant / total_relevant)

    mean_p = float(np.mean(precisions))
    mean_r = float(np.mean(recalls))
    return 2 * mean_p * mean_r / max(mean_p + mean_r, 1e-8)


class UnimodalRetrieval(Callback):
    """Unimodal image retrieval evaluator.

    Caches embeddings from a single output key during validation and computes
    F1@k for all-vs-all retrieval using class-label relevance.

    Args:
        pl_module: The ``spt.Module`` instance.
        name: Unique callback name (used for attribute and log key namespacing).
        input: Key in the outputs dict containing the embeddings to evaluate.
        features_dim: Embedding dimensionality.
        top_k: k values for F1@k evaluation.
    """

    def __init__(
        self,
        pl_module,
        name: str,
        input: str,
        features_dim: int,
        top_k: list[int] | tuple[int, ...] = (5, 10),
    ):
        super().__init__()
        logging.info(f"Setting up UnimodalRetrieval callback ({name})")
        self.name = name
        self.features_dim = features_dim
        self.top_k = list(top_k)

        fn = _wrap_validation_step(pl_module.validation_step, input, name)
        pl_module.validation_step = types.MethodType(fn, pl_module)

        setattr(pl_module, f"_uniret_{name}_emb", None)
        setattr(pl_module, f"_uniret_{name}_lbl", None)

    @property
    def state_key(self) -> str:
        return f"UnimodalRetrieval[name={self.name}]"

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        n = len(trainer.datamodule.val.dataset)
        if pl_module.local_rank == 0:
            dev = pl_module.device
            setattr(
                pl_module, f"_uniret_{self.name}_emb",
                torch.zeros((n, self.features_dim), device=dev),
            )
            setattr(
                pl_module, f"_uniret_{self.name}_lbl",
                torch.zeros(n, dtype=torch.long, device=dev),
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.local_rank != 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        logging.info(f"Computing unimodal retrieval metrics ({self.name})")

        emb = getattr(pl_module, f"_uniret_{self.name}_emb")
        lbl = getattr(pl_module, f"_uniret_{self.name}_lbl")

        logs = {}
        for k in self.top_k:
            f1 = _retrieval_f1_at_k(emb, lbl, k)
            key = f"eval/{self.name}_f1@{k}"
            logs[key] = f1
            logging.info(f"  F1@{k}: {f1:.4f}")

        pl_module.log_dict(logs, on_epoch=True, rank_zero_only=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        setattr(pl_module, f"_uniret_{self.name}_emb", None)
        setattr(pl_module, f"_uniret_{self.name}_lbl", None)
