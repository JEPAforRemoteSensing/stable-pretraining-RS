"""Unimodal JEPA (SIGReg) benchmark on fMoW RGB (62-class).

Single-encoder pre-training on the RGB modality only.  The objective is
LeJEPA-style: per-view SIGReg (sliced Epps-Pulley) + multi-view invariance,
combined as:

    loss = lamb * sigreg(all_views)  +  (1 - lamb) * invariance(views, center)

Two evaluation protocols run online throughout training:

  1. **Linear classification probe** — top-1 / top-5 accuracy on 62 classes.
  2. **Image retrieval F1@k** — all-vs-all cosine retrieval, label-based
     relevance (same class = relevant), for k in ``top_k``.

Usage::

    python benchmarks/fmow/unijepa_fmow.py

    # Override defaults via CLI:
    python benchmarks/fmow/unijepa_fmow.py \\
        --batch_size 128 --epochs 300 --lr 2e-3 \\
        --wandb_enabled --wandb_project fmow --wandb_run_name unijepa_rgb
"""

import argparse

import albumentations as A
import lightning as pl
import timm
import torch
import torch.nn as nn
import torchmetrics
from albumentations.pytorch import ToTensorV2
from lightning.data import StreamingDataLoader
from torchvision.ops import MLP

import stable_pretraining as spt

from fmow_dataset import FMoWStreamingDataset, NUM_CLASSES
from unimodal_sigreg_loss import UnimodalSIGRegLoss
from unimodal_retrieval import UnimodalRetrieval


# ---------------------------------------------------------------------------
# Transforms (albumentations, RGB only)
# ---------------------------------------------------------------------------

def make_train_transform_rgb(crop_size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.3, 1.0)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Normalize(
            mean=(0.4222, 0.4322, 0.4002),
            std=(0.2567, 0.2519, 0.2582),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def make_eval_transform_rgb():
    return A.Compose([
        A.Normalize(
            mean=(0.4222, 0.4322, 0.4002),
            std=(0.2567, 0.2519, 0.2582),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# Placeholder MS transform — dataset still loads MS tensors but they are
# not used in the forward function.
def make_dummy_transform_ms():
    return A.Compose([A.Normalize(mean=[0] * 10, std=[1] * 10), ToTensorV2()])


# ---------------------------------------------------------------------------
# Forward function
# ---------------------------------------------------------------------------

def unijepa_forward(self, batch, stage):
    """Unimodal JEPA (SIGReg) forward — RGB only.

    Expects ``self`` to have:
        - ``encoder_rgb``:   ResNetV2-18 backbone (3-channel RGB)
        - ``projector_rgb``: MLP projection head (for SIGReg)
        - ``unimodal_loss``: :class:`UnimodalSIGRegLoss` instance

    Batch keys used:
        - ``image_rgb``: ``(B, V, 3, H, W)`` training / ``(B, 3, H, W)`` eval
        - ``label``:     ``(B,)`` integer class labels
        - ``sample_idx``: dataset index (needed by retrieval callback)
    """
    out = {}

    if batch["image_rgb"].dim() == 5:
        # --- Training: multi-view ---
        B, V = batch["image_rgb"].shape[:2]

        rgb_flat = batch["image_rgb"].flatten(0, 1)   # (BV, 3, H, W)

        emb = self.encoder_rgb(rgb_flat)               # (BV, D)
        proj = self.projector_rgb(emb)                 # (BV, K)

        proj_views = proj.reshape(B, V, -1).permute(1, 0, 2)  # (V, B, K)

        loss, loss_dict = self.unimodal_loss(proj_views)
        out["loss"] = loss

        for k, v in loss_dict.items():
            self.log(f"{stage}/{k}", v, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        # Mean-pool across views for downstream callbacks
        D = emb.shape[-1]
        out["embedding_rgb"] = emb.reshape(B, V, D).mean(1)   # (B, D)

    else:
        # --- Eval: single view ---
        out["embedding_rgb"] = self.encoder_rgb(batch["image_rgb"])

    out["label"] = batch["label"]
    if "sample_idx" in batch:
        out["sample_idx"] = batch["sample_idx"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unimodal JEPA (SIGReg) on fMoW RGB (62 classes)")

    # Data
    p.add_argument("--data_dir", type=str, default="data/fmow")
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=12)

    # Architecture
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--proj_dim", type=int, default=128)

    # Loss
    p.add_argument("--lamb", type=float, default=0.02,
                   help="Weight on SIGReg; (1-lamb) weights invariance")
    p.add_argument("--num_slices", type=int, default=256, help="SIGReg random projections")

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--compile", action="store_true")

    # Eval
    p.add_argument("--eval_freq", type=int, default=5)
    p.add_argument("--top_k", type=int, nargs="+", default=[5, 10])
    p.add_argument("--knn_queue_length", type=int, default=10000)
    p.add_argument("--knn_k", type=int, default=20)

    # Checkpointing
    p.add_argument("--save_freq", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="benchmarks/fmow/checkpoints")
    # Logging
    p.add_argument("--wandb_enabled", action="store_true")
    p.add_argument("--wandb_project", type=str, default="fmow")
    p.add_argument("--wandb_run_name", type=str, default="unijepa-rgb")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Data ----
    dummy_ms = make_dummy_transform_ms()

    train_dataset = FMoWStreamingDataset(
        input_dir=f"{args.data_dir}/train",
        transform_rgb=make_train_transform_rgb(),
        transform_ms=dummy_ms,
        num_views=args.num_views,
    )
    val_dataset = FMoWStreamingDataset(
        input_dir=f"{args.data_dir}/val",
        transform_rgb=make_eval_transform_rgb(),
        transform_ms=dummy_ms,
        num_views=1,
    )
    test_dataset = FMoWStreamingDataset(
        input_dir=f"{args.data_dir}/test",
        transform_rgb=make_eval_transform_rgb(),
        transform_ms=dummy_ms,
        num_views=1,
    )

    dl_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **dl_kwargs,
    )
    val_loader  = StreamingDataLoader(val_dataset,  batch_size=args.batch_size, **dl_kwargs)
    test_loader = StreamingDataLoader(test_dataset, batch_size=args.batch_size, **dl_kwargs)

    data = spt.data.DataModule(train=train_loader, val=val_loader, test=test_loader)

    # ---- Model components ----
    encoder_rgb = timm.create_model(
        "resnetv2_18", pretrained=False, num_classes=args.embed_dim, in_chans=3,
    )

    projector_rgb = MLP(
        args.embed_dim, [4096, 4096, args.proj_dim], norm_layer=nn.BatchNorm1d,
    )

    unimodal_loss = UnimodalSIGRegLoss(
        lamb=args.lamb,
        num_slices=args.num_slices,
        t_max=5.0,
        n_points=17,
    )

    # ---- Module ----
    total_steps = len(train_loader) * args.epochs

    module = spt.Module(
        encoder_rgb=encoder_rgb,
        projector_rgb=projector_rgb,
        unimodal_loss=unimodal_loss,
        forward=unijepa_forward,
        optim={
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {
                "type": "LinearWarmupCosineAnnealing",
                "peak_step": len(train_loader) / total_steps,  # 1 epoch warmup
                "start_factor": 0.01,
                "end_lr": args.lr / 1000,
                "total_steps": total_steps,
            },
            "interval": "step",
        },
    )

    if args.compile:
        module = torch.compile(module, mode="reduce-overhead")

    # ---- Callbacks ----
    callbacks = []

    # Linear classification probe (top-1 / top-5)
    callbacks.append(
        spt.callbacks.OnlineProbe(
            module,
            name="probe_rgb",
            input="embedding_rgb",
            target="label",
            probe=nn.Linear(args.embed_dim, NUM_CLASSES),
            loss=nn.CrossEntropyLoss(),
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(NUM_CLASSES),
                "top5": torchmetrics.classification.MulticlassAccuracy(
                    NUM_CLASSES, top_k=5
                ),
            },
            optimizer={"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-6},
        )
    )

    # KNN probe
    callbacks.append(
        spt.callbacks.OnlineKNN(
            name="knn_rgb",
            input="embedding_rgb",
            target="label",
            queue_length=args.knn_queue_length,
            input_dim=args.embed_dim,
            k=args.knn_k,
            metrics={
                "top1": torchmetrics.classification.MulticlassAccuracy(
                    NUM_CLASSES, validate_args=False,
                ),
            },
        )
    )

    # RankMe (dimensional collapse detection)
    callbacks.append(
        spt.callbacks.RankMe(
            name="rankme_rgb",
            target="embedding_rgb",
            queue_length=1000,
            target_shape=args.embed_dim,
        )
    )

    # Image retrieval F1@k
    callbacks.append(
        UnimodalRetrieval(
            pl_module=module,
            name="retrieval_rgb",
            input="embedding_rgb",
            features_dim=args.embed_dim,
            top_k=tuple(args.top_k),
        )
    )

    # Checkpointing
    callbacks.append(
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename="unijepa-{epoch:03d}",
            save_top_k=-1,
            every_n_epochs=args.save_freq,
            save_last=True,
        )
    )

    # LR monitor
    callbacks.append(
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step")
    )

    # ---- Trainer ----
    logger = (
        pl.pytorch.loggers.WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            log_model=False,
        )
        if args.wandb_enabled
        else True
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        precision=args.precision,
        devices=args.devices,
        accelerator="gpu",
        gradient_clip_val=args.grad_clip,
        check_val_every_n_epoch=args.eval_freq,
        num_sanity_val_steps=0,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
    )

    # ---- Run ----
    manager = spt.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    main()
