"""MMLeJEPA benchmark on fMoW (62-class cross-modal image retrieval).

Multi-modal LeJEPA with two ResNetV2-18 encoders (RGB + multispectral),
SIGReg regularisation, and cross-modal alignment.

Usage::

    python benchmarks/fmow/mmlejepa_fmow.py

    # Override defaults via CLI:
    python benchmarks/fmow/mmlejepa_fmow.py \\
        --batch_size 128 --epochs 300 --lr 2e-3 --num_views 4 \\
        --wandb_enabled --wandb_project fmow --wandb_run_name mmlejepa_62cls
"""

import argparse
from pathlib import Path

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

from cross_modal_retrieval import CrossModalRetrieval
from fmow_dataset import FMoWStreamingDataset, NUM_CLASSES
from mmlejepa_loss import MMLeJEPALoss


# ---------------------------------------------------------------------------
# Transforms (albumentations)
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


def make_train_transform_ms(crop_size=96):
    return A.Compose([
        A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.3, 1.0)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.Normalize(
            mean=[0] * 10, std=[1] * 10, max_pixel_value=255.0,
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


def make_eval_transform_ms():
    return A.Compose([
        A.Normalize(
            mean=[0] * 10, std=[1] * 10, max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Forward function
# ---------------------------------------------------------------------------

def mmlejepa_forward(self, batch, stage):
    """MEMPLeJEPA forward: dual-encoder SSL with per-modality SIGReg + retrieval probes.

    Expects ``self`` to have:
        - ``encoder_rgb``, ``encoder_ms``: backbone encoders (trained with SIGReg only)
        - ``projector_rgb``, ``projector_ms``: MLP projection heads
        - ``ret_probe_rgb2ms``: MLP probe mapping RGB→MS embedding space
        - ``ret_probe_ms2rgb``: MLP probe mapping MS→RGB embedding space
        - ``mmlejepa_loss``: :class:`MMLeJEPALoss` instance

    Batch keys:
        - ``image_rgb``: ``(B, V, 3, H, W)`` (training) or ``(B, 3, H, W)`` (eval)
        - ``image_ms``:  ``(B, V, 10, H, W)`` (training) or ``(B, 10, H, W)`` (eval)
        - ``label``: ``(B,)`` integer class labels

    Retrieval probes are trained with MSE on detached encoder outputs:
        - ``MSE(ret_probe_rgb2ms(emb_rgb.detach()), emb_ms.detach())``
        - ``MSE(ret_probe_ms2rgb(emb_ms.detach()), emb_rgb.detach())``
    """
    out = {}

    if batch["image_rgb"].dim() == 5:
        # --- Training: multi-view ---
        B, V = batch["image_rgb"].shape[:2]

        rgb_flat = batch["image_rgb"].flatten(0, 1)  # (BV, 3, H, W)
        ms_flat = batch["image_ms"].flatten(0, 1)  # (BV, 10, H, W)

        emb_rgb = self.encoder_rgb(rgb_flat)  # (BV, D)
        emb_ms = self.encoder_ms(ms_flat)  # (BV, D)

        proj_rgb = self.projector_rgb(emb_rgb)  # (BV, K)
        proj_ms = self.projector_ms(emb_ms)  # (BV, K)

        # Reshape to (V, B, K) for loss computation
        proj_rgb_views = proj_rgb.reshape(B, V, -1).permute(1, 0, 2)
        proj_ms_views = proj_ms.reshape(B, V, -1).permute(1, 0, 2)

        sigreg_loss, loss_components = self.mmlejepa_loss(proj_rgb_views, proj_ms_views)

        # Mean over views for probe training and retrieval callbacks
        D = emb_rgb.shape[-1]
        emb_rgb_mean = emb_rgb.reshape(B, V, D).mean(1)  # (B, D)
        emb_ms_mean = emb_ms.reshape(B, V, D).mean(1)   # (B, D)

        # --- Retrieval probes: trained on detached encoder outputs ---
        emb_rgb_det = emb_rgb_mean.detach()
        emb_ms_det = emb_ms_mean.detach()

        embhat_rgb2ms = self.ret_probe_rgb2ms(emb_rgb_det)  # (B, D)
        embhat_ms2rgb = self.ret_probe_ms2rgb(emb_ms_det)   # (B, D)

        probe_loss_rgb2ms = nn.functional.mse_loss(embhat_rgb2ms, emb_ms_det)
        probe_loss_ms2rgb = nn.functional.mse_loss(embhat_ms2rgb, emb_rgb_det)
        probe_loss = probe_loss_rgb2ms + probe_loss_ms2rgb

        loss = sigreg_loss + self.ret_probe_weight * probe_loss
        out["loss"] = loss

        # Log loss components
        for k, v in loss_components.items():
            self.log(f"{stage}/{k}", v, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/probe_loss/rgb2ms", probe_loss_rgb2ms.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/probe_loss/ms2rgb", probe_loss_ms2rgb.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        out["embedding_rgb"] = emb_rgb_mean
        out["embedding_ms"] = emb_ms_mean
        out["embhat_rgb2ms"] = embhat_rgb2ms
        out["embhat_ms2rgb"] = embhat_ms2rgb

    else:
        # --- Eval: single view ---
        emb_rgb = self.encoder_rgb(batch["image_rgb"])
        emb_ms = self.encoder_ms(batch["image_ms"])
        out["embedding_rgb"] = emb_rgb
        out["embedding_ms"] = emb_ms
        out["embhat_rgb2ms"] = self.ret_probe_rgb2ms(emb_rgb.detach())
        out["embhat_ms2rgb"] = self.ret_probe_ms2rgb(emb_ms.detach())

    out["label"] = batch["label"]
    if "sample_idx" in batch:
        out["sample_idx"] = batch["sample_idx"]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MMLeJEPA on fMoW (62 classes)")

    # Data
    p.add_argument("--data_dir", type=str, default="data/fmow")
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=12)

    # Architecture
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--proj_dim", type=int, default=128)

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lamb", type=float, default=0.02, help="SIGReg weight")
    p.add_argument("--num_slices", type=int, default=256, help="SIGReg slices")
    p.add_argument("--ret_probe_weight", type=float, default=1.0, help="retrieval probe MSE loss weight")
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
    p.add_argument("--resume", type=str, default=None)

    # Logging
    p.add_argument("--wandb_enabled", action="store_true")
    p.add_argument("--wandb_project", type=str, default="fmow")
    p.add_argument("--wandb_run_name", type=str, default="mmlejepa-62cls")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Data ----
    train_dataset = FMoWStreamingDataset(
        input_dir=f"{args.data_dir}/train",
        transform_rgb=make_train_transform_rgb(),
        transform_ms=make_train_transform_ms(),
        num_views=args.num_views,
    )
    val_dataset = FMoWStreamingDataset(
        input_dir=f"{args.data_dir}/val",
        transform_rgb=make_eval_transform_rgb(),
        transform_ms=make_eval_transform_ms(),
        num_views=1,
    )
    test_dataset = FMoWStreamingDataset(
        input_dir=f"{args.data_dir}/test",
        transform_rgb=make_eval_transform_rgb(),
        transform_ms=make_eval_transform_ms(),
        num_views=1,
    )

    dl_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **dl_kwargs,
    )
    val_loader = StreamingDataLoader(
        val_dataset, batch_size=args.batch_size, **dl_kwargs,
    )
    test_loader = StreamingDataLoader(
        test_dataset, batch_size=args.batch_size, **dl_kwargs,
    )

    data = spt.data.DataModule(train=train_loader, val=val_loader, test=test_loader)

    # ---- Model components ----
    encoder_rgb = timm.create_model(
        "resnetv2_18", pretrained=False, num_classes=args.embed_dim, in_chans=3,
    )
    encoder_ms = timm.create_model(
        "resnetv2_18", pretrained=False, num_classes=args.embed_dim, in_chans=10,
    )
    projector_rgb = MLP(
        args.embed_dim, [4096, 4096, args.proj_dim], norm_layer=nn.BatchNorm1d,
    )
    projector_ms = MLP(
        args.embed_dim, [4096, 4096, args.proj_dim], norm_layer=nn.BatchNorm1d,
    )
    ret_probe_rgb2ms = MLP(
        args.embed_dim, [2048, args.embed_dim], norm_layer=nn.BatchNorm1d,
    )
    ret_probe_ms2rgb = MLP(
        args.embed_dim, [2048, args.embed_dim], norm_layer=nn.BatchNorm1d,
    )

    # ---- Loss ----
    mmlejepa_loss = MMLeJEPALoss(
        lamb=args.lamb,
        num_slices=args.num_slices,
        t_max=5.0,
        n_points=17,
    )

    # ---- Module ----
    total_steps = len(train_loader) * args.epochs

    module = spt.Module(
        encoder_rgb=encoder_rgb,
        encoder_ms=encoder_ms,
        projector_rgb=projector_rgb,
        projector_ms=projector_ms,
        ret_probe_rgb2ms=ret_probe_rgb2ms,
        ret_probe_ms2rgb=ret_probe_ms2rgb,
        mmlejepa_loss=mmlejepa_loss,
        ret_probe_weight=args.ret_probe_weight,
        forward=mmlejepa_forward,
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

    # Online linear probes (one per modality)
    for modality, dim in [("rgb", args.embed_dim), ("ms", args.embed_dim)]:
        callbacks.append(
            spt.callbacks.OnlineProbe(
                module,
                name=f"probe_{modality}",
                input=f"embedding_{modality}",
                target="label",
                probe=nn.Linear(dim, NUM_CLASSES),
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

    # Online KNN probes (one per modality)
    # Note: OnlineKNN infers num_classes from cached labels, which can be < 62
    # early in training if the queue hasn't seen all classes. We use a large
    # queue and avoid top_k to minimise mismatches.
    for modality, dim in [("rgb", args.embed_dim), ("ms", args.embed_dim)]:
        callbacks.append(
            spt.callbacks.OnlineKNN(
                name=f"knn_{modality}",
                input=f"embedding_{modality}",
                target="label",
                queue_length=args.knn_queue_length,
                input_dim=dim,
                k=args.knn_k,
                metrics={
                    "top1": torchmetrics.classification.MulticlassAccuracy(
                        NUM_CLASSES, validate_args=False,
                    ),
                },
            )
        )

    # RankMe (dimensional collapse detection, one per modality)
    for modality, dim in [("rgb", args.embed_dim), ("ms", args.embed_dim)]:
        callbacks.append(
            spt.callbacks.RankMe(
                name=f"rankme_{modality}",
                target=f"embedding_{modality}",
                queue_length=1000,
                target_shape=dim,
            )
        )

    # Cross-modal retrieval
    callbacks.append(
        CrossModalRetrieval(
            pl_module=module,
            name="xmodal_retrieval",
            input_rgb="embedding_rgb",
            input_ms="embedding_ms",
            input_rgb2ms="embhat_rgb2ms",
            input_ms2rgb="embhat_ms2rgb",
            features_dim=args.embed_dim,
            top_k=tuple(args.top_k),
        )
    )

    # Checkpointing
    callbacks.append(
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename="mmlejepa-{epoch:03d}",
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
        else True  # default TensorBoard logger
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
