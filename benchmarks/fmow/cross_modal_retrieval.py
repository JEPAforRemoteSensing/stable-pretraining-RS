"""Cross-modal image retrieval callback (MEMPLeJEPA version).

Evaluates retrieval quality across four directions using both encoder
embeddings and retrieval-probe embeddings:

  S1->S1: encoder_rgb query  vs encoder_rgb gallery  (same-modal RGB)
  S2->S2: encoder_ms  query  vs encoder_ms  gallery  (same-modal MS)
  S1->S2: probe_rgb2ms query vs encoder_ms  gallery  (cross-modal RGB->MS)
  S2->S1: probe_ms2rgb query vs encoder_rgb gallery  (cross-modal MS->RGB)

Uses label-based relevance: a gallery item is relevant when it shares the
same class as the query.  All validation samples serve as both query and
gallery (self-matches are not excluded — they add a small constant boost
equally across all methods, so comparisons remain fair).
"""

import types

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger as logging


def _wrap_validation_step(fn, input_rgb, input_ms, input_rgb2ms, input_ms2rgb, name):
    """Wrap ``validation_step`` to cache all four embedding types."""
    attr_rgb    = f"_xmodal_{name}_rgb"
    attr_ms     = f"_xmodal_{name}_ms"
    attr_rgb2ms = f"_xmodal_{name}_rgb2ms"
    attr_ms2rgb = f"_xmodal_{name}_ms2rgb"

    def wrapped(
        self, batch, batch_idx, fn=fn,
        attr_rgb=attr_rgb, attr_ms=attr_ms,
        attr_rgb2ms=attr_rgb2ms, attr_ms2rgb=attr_ms2rgb,
    ):
        batch = fn(batch, batch_idx)

        with torch.no_grad():
            e_rgb    = F.normalize(batch[input_rgb].float(),    dim=1)
            e_ms     = F.normalize(batch[input_ms].float(),     dim=1)
            e_rgb2ms = F.normalize(batch[input_rgb2ms].float(), dim=1)
            e_ms2rgb = F.normalize(batch[input_ms2rgb].float(), dim=1)

        idx      = self.all_gather(batch["sample_idx"])
        e_rgb    = self.all_gather(e_rgb)
        e_ms     = self.all_gather(e_ms)
        e_rgb2ms = self.all_gather(e_rgb2ms)
        e_ms2rgb = self.all_gather(e_ms2rgb)

        if self.local_rank == 0:
            getattr(self, attr_rgb)[idx]    = e_rgb
            getattr(self, attr_ms)[idx]     = e_ms
            getattr(self, attr_rgb2ms)[idx] = e_rgb2ms
            getattr(self, attr_ms2rgb)[idx] = e_ms2rgb

        return batch

    return wrapped


def _retrieval_f1_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k):
    """Compute mean retrieval F1@k via cosine similarity (L2-normed inputs).

    Args:
        query_emb:     ``(Q, D)`` L2-normalised query embeddings.
        gallery_emb:   ``(G, D)`` L2-normalised gallery embeddings.
        query_labels:  ``(Q,)`` integer class labels.
        gallery_labels:``(G,)`` integer class labels.
        k: Number of items to retrieve.

    Returns:
        Mean F1@k (float).
    """
    scores = query_emb @ gallery_emb.T  # (Q, G)
    _, topk_idx = scores.topk(k, dim=1)  # (Q, k)

    class_counts = {}
    gl = gallery_labels.cpu().numpy()
    for lbl in gl:
        class_counts[int(lbl)] = class_counts.get(int(lbl), 0) + 1

    ql = query_labels.cpu().numpy()
    topk_labels = gallery_labels[topk_idx]  # (Q, k)

    precisions, recalls = [], []
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
    """Cross-modal retrieval evaluator for MEMPLeJEPA.

    Caches four embedding types during validation and computes F1@k for
    all four retrieval directions:

    - **S1->S1**: ``input_rgb`` query  vs ``input_rgb`` gallery
    - **S2->S2**: ``input_ms`` query   vs ``input_ms`` gallery
    - **S1->S2**: ``input_rgb2ms`` query vs ``input_ms`` gallery
    - **S2->S1**: ``input_ms2rgb`` query vs ``input_rgb`` gallery

    Args:
        pl_module: The ``spt.Module`` instance.
        name: Unique callback name.
        input_rgb: Output key for RGB encoder embeddings.
        input_ms: Output key for MS encoder embeddings.
        input_rgb2ms: Output key for the RGB→MS retrieval probe embeddings.
        input_ms2rgb: Output key for the MS→RGB retrieval probe embeddings.
        features_dim: Embedding dimensionality.
        top_k: List of k values for F1@k evaluation.
    """

    def __init__(
        self,
        pl_module,
        name: str,
        input_rgb: str = "embedding_rgb",
        input_ms: str = "embedding_ms",
        input_rgb2ms: str = "embhat_rgb2ms",
        input_ms2rgb: str = "embhat_ms2rgb",
        features_dim: int = 1024,
        top_k: list[int] | tuple[int, ...] = (5, 10),
    ):
        super().__init__()
        logging.info(f"Setting up CrossModalRetrieval callback ({name})")
        self.name = name
        self.features_dim = features_dim
        self.top_k = list(top_k)

        fn = _wrap_validation_step(
            pl_module.validation_step,
            input_rgb, input_ms, input_rgb2ms, input_ms2rgb,
            name,
        )
        pl_module.validation_step = types.MethodType(fn, pl_module)

        for suffix in ("rgb", "ms", "rgb2ms", "ms2rgb"):
            setattr(pl_module, f"_xmodal_{name}_{suffix}", None)

    @property
    def state_key(self) -> str:
        return f"CrossModalRetrieval[name={self.name}]"

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        n = len(trainer.datamodule.val.dataset)
        if pl_module.local_rank == 0:
            dev = pl_module.device
            for suffix in ("rgb", "ms", "rgb2ms", "ms2rgb"):
                setattr(
                    pl_module, f"_xmodal_{self.name}_{suffix}",
                    torch.zeros((n, self.features_dim), device=dev),
                )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.local_rank != 0:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        logging.info(f"Computing cross-modal retrieval metrics ({self.name})")

        emb_rgb    = getattr(pl_module, f"_xmodal_{self.name}_rgb")
        emb_ms     = getattr(pl_module, f"_xmodal_{self.name}_ms")
        emb_rgb2ms = getattr(pl_module, f"_xmodal_{self.name}_rgb2ms")
        emb_ms2rgb = getattr(pl_module, f"_xmodal_{self.name}_ms2rgb")

        val_dataset = trainer.datamodule.val.dataset
        labels = torch.tensor(
            [val_dataset[i]["label"] for i in range(len(val_dataset))],
            device=emb_rgb.device,
        )

        # query_emb, gallery_emb, query_labels (same as gallery for all)
        directions = {
            "s1_to_s1": (emb_rgb,    emb_rgb),
            "s2_to_s2": (emb_ms,     emb_ms),
            "s1_to_s2": (emb_rgb2ms, emb_ms),      # probe(RGB) → MS gallery
            "s2_to_s1": (emb_ms2rgb, emb_rgb),     # probe(MS)  → RGB gallery
        }

        logs = {}
        for direction_name, (query_emb, gallery_emb) in directions.items():
            for k in self.top_k:
                f1 = _retrieval_f1_at_k(query_emb, gallery_emb, labels, labels, k)
                logs[f"eval/{self.name}_{direction_name}_f1@{k}"] = f1
                logging.info(f"  {direction_name} F1@{k}: {f1:.4f}")

        pl_module.log_dict(logs, on_epoch=True, rank_zero_only=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        for suffix in ("rgb", "ms", "rgb2ms", "ms2rgb"):
            setattr(pl_module, f"_xmodal_{self.name}_{suffix}", None)
