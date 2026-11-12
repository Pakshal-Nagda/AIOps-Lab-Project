#!/usr/bin/env python3
"""
Project Medhā - Sanskrit ASR Fine-Tuning & MLflow Experiment Tracking
=====================================================================
Fine-tunes the Su-śrotā Conformer-CTC model (models/base_asr.nemo) on domain-specific
recitation audio with end-to-end MLflow logging and model checkpointing.

Features:
- Dual hardware support: Seamlessly runs on GPU (CUDA / Kaggle / Colab) or CPU.
- Standard EncDecCTCModelBPE fine-tuning with CTC loss and SentencePiece tokenizer.
- Encoder freezing option (--freeze-encoder): trains 3-4x faster for small datasets or CPU.
- Comprehensive MLflow tracking: logs hyperparameters, loss curves, validation WER, and artifacts.
- Post-training export: optionally triggers ONNX quantization for immediate deployment.

Usage:
    # 1. Standard training on GPU/Kaggle:
    python scripts/train.py --train-manifest data/train_manifest.json --val-manifest data/val_manifest.json --epochs 15

    # 2. Fast CPU fine-tuning (freezing encoder, decoder-only tuning):
    python scripts/train.py --train-manifest data/train_manifest.json --val-manifest data/val_manifest.json --epochs 10 --device cpu --freeze-encoder

    # 3. View MLflow experiments dashboard:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""

import os
import sys

# Prevent protobuf runtime mismatch warnings and tokenizer deadlocks
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import json
import argparse
import logging
from typing import Optional
import torch
from omegaconf import DictConfig

import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import nemo.collections.asr as na
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train_medha")


def setup_data_and_optimizer(
    model: na.models.EncDecCTCModelBPE,
    train_manifest: str,
    val_manifest: Optional[str],
    batch_size: int,
    num_workers: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    device_type: str
):
    """Configures NeMo DataLoaders and CosineAnnealing AdamW Optimizer."""
    is_cuda = (device_type == "cuda")

    # 1. Configure Training Dataset
    train_ds_cfg = DictConfig({
        "manifest_filepath": train_manifest,
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": is_cuda,
        "max_duration": 18.0,
        "min_duration": 0.2,
        "is_tarred": False
    })
    model.setup_training_data(train_ds_cfg)

    # 2. Configure Validation Dataset (if provided)
    if val_manifest and os.path.exists(val_manifest):
        val_ds_cfg = DictConfig({
            "manifest_filepath": val_manifest,
            "sample_rate": 16000,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": is_cuda,
            "max_duration": 18.0,
            "min_duration": 0.2,
            "is_tarred": False
        })
        model.setup_validation_data(val_ds_cfg)

    # 3. Configure Optimizer and LR Schedule
    optim_cfg = DictConfig({
        "name": "adamw",
        "lr": lr,
        "betas": [0.9, 0.98],
        "weight_decay": 1e-3,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": warmup_steps,
            "min_lr": min_lr
        }
    })
    model.setup_optimization(optim_cfg)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Su-śrotā Sanskrit Conformer-CTC Model with MLflow")
    parser.add_argument("--base-model", type=str, default="models/base_asr.nemo", help="Path to base .nemo checkpoint")
    parser.add_argument("--train-manifest", type=str, default="data/train_manifest.json", help="Path to training manifest.json")
    parser.add_argument("--val-manifest", type=str, default="data/val_manifest.json", help="Path to validation manifest.json")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory to save fine-tuned model")
    parser.add_argument("--output-name", type=str, default="finetuned_asr.nemo", help="Filename of fine-tuned checkpoint")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for Cosine schedule")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear warmup steps")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    
    # Hardware & Optimization Options
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze Conformer encoder layers; train CTC head only")
    parser.add_argument("--export-onnx", action="store_true", help="Export to quantized ONNX right after training")
    
    # MLflow Tracking
    parser.add_argument("--experiment-name", type=str, default="medha-sanskrit-asr-finetune", help="MLflow experiment name")
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI (database backend)")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    args = parser.parse_args()

    # 1. Validation of paths
    if not os.path.exists(args.base_model):
        logger.error(f"Base model checkpoint '{args.base_model}' not found! Run scripts/download_model.py first.")
        sys.exit(1)
    if not os.path.exists(args.train_manifest):
        logger.error(f"Training manifest '{args.train_manifest}' not found! Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    final_output_path = os.path.join(args.output_dir, args.output_name)

    # 2. Hardware selection
    if args.device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = args.device

    logger.info(f"Using compute hardware: {device_type.upper()}")
    if device_type == "cpu":
        logger.warning("Training on CPU. For faster execution, consider --freeze-encoder or running on free Kaggle GPU.")

    # 3. Load Base Checkpoint
    logger.info(f"Restoring base model from {args.base_model}...")
    model = na.models.EncDecCTCModelBPE.restore_from(args.base_model, map_location=device_type)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model restored! Total parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 4. Optional: Freeze Conformer Encoder for fast lightweight fine-tuning
    if args.freeze_encoder:
        logger.info("Option --freeze-encoder active: Freezing Conformer encoder layers. Only CTC head will be updated.")
        model.encoder.freeze()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after freezing encoder: {trainable_params:,}")

    # 5. Setup Data & Optimizer
    setup_data_and_optimizer(
        model=model,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest if os.path.exists(args.val_manifest) else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers if device_type == "cuda" else 0,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        device_type=device_type
    )

    # Count dataset samples
    with open(args.train_manifest, "r", encoding="utf-8") as f:
        train_sample_count = sum(1 for line in f if line.strip())

    val_sample_count = 0
    if os.path.exists(args.val_manifest):
        with open(args.val_manifest, "r", encoding="utf-8") as f:
            val_sample_count = sum(1 for line in f if line.strip())

    # 6. Initialize MLflow Experiment Tracking
    logger.info(f"Initializing MLflow Experiment: '{args.experiment_name}' at '{args.mlflow_uri}'...")
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.mlflow_uri,
        run_name=args.run_name
    )

    # Log initial hyperparameters
    mlf_logger.log_hyperparams({
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "device": device_type,
        "freeze_encoder": args.freeze_encoder,
        "train_samples": train_sample_count,
        "val_samples": val_sample_count,
        "vocab_size": model.tokenizer.vocab_size
    })

    # 7. Setup Lightning Callbacks & Trainer
    callbacks = [
        LearningRateMonitor(logging_interval="step")
    ]

    use_val = os.path.exists(args.val_manifest) and val_sample_count > 0
    if use_val:
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="medha-best-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        callbacks.append(checkpoint_cb)

    precision = "16-mixed" if device_type == "cuda" else "32-true"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=device_type,
        devices=1,
        precision=precision,
        logger=mlf_logger,
        callbacks=callbacks,
        log_every_n_steps=max(1, min(10, train_sample_count // args.batch_size)),
        enable_checkpointing=use_val
    )

    # 8. Run Model Training
    logger.info("=" * 60)
    logger.info(f"STARTING FINE-TUNING FOR {args.epochs} EPOCHS...")
    logger.info("=" * 60)

    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current checkpoint...")

    # 9. Save Fine-Tuned .nemo Model Checkpoint
    logger.info(f"Saving fine-tuned .nemo model to {final_output_path}...")
    model.save_to(final_output_path)
    logger.info(f"Model saved successfully! ({round(os.path.getsize(final_output_path) / 1024 / 1024, 1)} MB)")

    # 10. Log Artifacts to MLflow
    try:
        client = mlflow.tracking.MlflowClient(args.mlflow_uri)
        run_id = mlf_logger.run_id
        client.log_artifact(run_id, final_output_path, artifact_path="model_checkpoints")
        logger.info(f"Logged model artifact to MLflow run: {run_id}")
    except Exception as e:
        logger.warning(f"Could not log artifact to MLflow: {e}")

    # 11. Optional ONNX Quantized Export
    if args.export_onnx:
        logger.info("Exporting fine-tuned model to quantized INT8 ONNX...")
        from subprocess import run
        export_cmd = [
            sys.executable, "scripts/convert_nemo_to_onnx.py",
            "--input_nemo", final_output_path,
            "--output_dir", args.output_dir
        ]
        run(export_cmd, check=True)
        logger.info("ONNX export completed! Production serving model is ready.")

    logger.info("=" * 60)
    logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Fine-tuned model: {final_output_path}")
    logger.info(f"To inspect experiment runs: mlflow ui --backend-store-uri {args.mlflow_uri}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
