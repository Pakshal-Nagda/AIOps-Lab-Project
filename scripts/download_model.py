#!/usr/bin/env python3
"""
Project Medhā - Model Download & Preparation Script
===================================================
Downloads the official Su-śrotā base checkpoint from Hugging Face:
    repo: prathoshap/sushrota-sanskrit-asr
    file: sushrota_sanskrit_asr_v13b.nemo

Cleans the raw multilingual hybrid RNNT-CTC checkpoint directly into a modern,
standard monolingual Sanskrit Conformer-CTC (EncDecCTCModelBPE) architecture
and saves it as `models/base_asr.nemo`.

This ensures standard NeMo can directly load `models/base_asr.nemo` in `train.py`
without any custom forks, monkeypatches, or dependency conflicts.

Usage:
    python scripts/download_model.py
"""

import os
import sys

# Prevent protobuf gencode/runtime version mismatch issues
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import shutil
import tarfile
import tempfile
import yaml
import torch
import sentencepiece as spm
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("download_model")

MODELS_DIR = "models"
OUTPUT_NEMO = os.path.join(MODELS_DIR, "base_asr.nemo")


def clean_raw_checkpoint(raw_nemo_path: str, output_path: str):
    """
    Transforms raw AI4Bharat hybrid multilingual checkpoint into a modern,
    standard monolingual Sanskrit EncDecCTCModelBPE checkpoint.
    """
    import nemo.collections.asr as na

    logger.info("Extracting and cleaning checkpoint into modern EncDecCTCModelBPE...")
    temp_dir = tempfile.mkdtemp(prefix="nemo_clean_")

    try:
        with tarfile.open(raw_nemo_path, "r:*") as tar:
            # 1. Extract config
            cfg_member = next((m for m in tar.getmembers() if m.name.endswith("model_config.yaml")), None)
            if not cfg_member:
                raise FileNotFoundError("Could not find model_config.yaml in raw archive")
            cfg = yaml.safe_load(tar.extractfile(cfg_member))

            # 2. Extract Sanskrit SentencePiece tokenizer and vocabulary
            sa_tok_member = next(
                (m for m in tar.getmembers() if "789437b25962494a8ef6a7a1500bbaf4_tokenizer.model" in m.name
                 or (m.name.endswith(".model") and "sa" in m.name)),
                None
            )
            if not sa_tok_member:
                sa_tok_member = next((m for m in tar.getmembers() if m.name.endswith(".model")), None)

            tok_dir = os.path.join(temp_dir, "tokenizer")
            os.makedirs(tok_dir, exist_ok=True)
            tok_path = os.path.join(tok_dir, "tokenizer.model")
            vocab_path = os.path.join(tok_dir, "vocab.txt")

            with open(tok_path, "wb") as f:
                f.write(tar.extractfile(sa_tok_member).read())

            sa_vocab_member = next(
                (m for m in tar.getmembers() if "f333604d956b4f47b2ede1f8aab5597a_vocab.txt" in m.name
                 or (m.name.endswith("vocab.txt") and "sa" in m.name)),
                None
            )
            if sa_vocab_member:
                with open(vocab_path, "wb") as f:
                    f.write(tar.extractfile(sa_vocab_member).read())
            else:
                with open(vocab_path, "w", encoding="utf-8") as f:
                    f.write("")

            # 3. Read vocabulary pieces
            sp = spm.SentencePieceProcessor()
            sp.load(tok_path)
            vocab_list = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
            logger.info(f"Loaded Sanskrit tokenizer with {len(vocab_list)} tokens.")

            # 4. Construct clean EncDecCTCModelBPE configuration
            clean_cfg = {
                "sample_rate": 16000,
                "preprocessor": cfg["preprocessor"],
                "encoder": cfg["encoder"],
                "decoder": {
                    "_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                    "feat_in": 512,
                    "num_classes": len(vocab_list),
                    "vocabulary": vocab_list
                },
                "tokenizer": {
                    "dir": tok_dir,
                    "type": "bpe",
                    "model_path": tok_path,
                    "vocab_path": vocab_path
                }
            }
            clean_cfg["preprocessor"]["_target_"] = "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"
            clean_cfg["encoder"]["_target_"] = "nemo.collections.asr.modules.ConformerEncoder"

            # 5. Instantiate clean model
            clean_model = na.models.EncDecCTCModelBPE(cfg=OmegaConf.create(clean_cfg))

            # 6. Load encoder weights
            ckpt_member = next((m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt")), None)
            if not ckpt_member:
                raise FileNotFoundError("Could not find model_weights.ckpt in raw archive")
            ckpt = torch.load(tar.extractfile(ckpt_member), map_location="cpu")

            logger.info("Transferring ConformerEncoder weights...")
            enc_sd = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
            clean_model.encoder.load_state_dict(enc_sd)

            # 7. Transfer Sanskrit slice into ConvASRDecoder
            logger.info("Transferring Sanskrit CTC projection slice to ConvASRDecoder...")
            cols = list(range(4096, 4096 + len(vocab_list))) + [5632]
            clean_model.decoder.decoder_layers[0].weight.data = ckpt["ctc_decoder.decoder_layers.0.weight"][cols].clone()
            clean_model.decoder.decoder_layers[0].bias.data = ckpt["ctc_decoder.decoder_layers.0.bias"][cols].clone()

            # 8. Save to output .nemo
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            logger.info(f"Saving clean modern NeMo checkpoint to: {output_path}...")
            clean_model.save_to(output_path)
            clean_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Clean NeMo checkpoint saved successfully ({clean_size_mb:.1f} MB)!")

            # 9. Verify standard restore
            logger.info("Verifying standard restore on clean checkpoint...")
            verified = na.models.EncDecCTCModelBPE.restore_from(output_path)
            logger.info(f"Verification successful: Restored {type(verified).__name__} cleanly with 0 hacks.")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    repo_id = "prathoshap/sushrota-sanskrit-asr"
    filename = "sushrota_sanskrit_asr_v13b.nemo"

    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Checking/Downloading '{filename}' from '{repo_id}'...")
    raw_cached_path = hf_hub_download(repo_id=repo_id, filename=filename)

    logger.info(f"Downloaded raw checkpoint: {raw_cached_path}")
    clean_raw_checkpoint(raw_cached_path, OUTPUT_NEMO)

    logger.info("=" * 60)
    logger.info(f"SUCCESS: Clean base model ready at '{OUTPUT_NEMO}'!")
    logger.info("Can be loaded in train.py via: EncDecCTCModelBPE.restore_from('models/base_asr.nemo')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
