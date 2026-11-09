#!/usr/bin/env python3
"""
Project Medhā - NeMo to Quantized ONNX Model Converter
======================================================
Converts a NeMo (.nemo) ASR checkpoint (clean Sanskrit EncDecCTCModelBPE or legacy)
into an ONNX acoustic model and preprocessor, directly storing the INT8 quantized
version (model_asr.onnx). No FP32 model is kept on disk.

Usage:
    python scripts/convert_nemo_to_onnx.py --input_nemo models/base_asr.nemo --output_dir models/
"""

import os
import sys

# Prevent protobuf gencode/runtime version mismatch issues
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import shutil
import tarfile
import tempfile
import numpy as np
import yaml

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("convert_nemo_to_onnx")


class SanskritConformerCTC(nn.Module):
    """
    Combines ConformerEncoder with Sanskrit CTC linear projection head.
    Outputs logits with shape [batch, time, num_classes].
    """
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(audio_signal=audio_signal, length=length)
        logits = self.head(encoded).transpose(1, 2)
        return logits


class PreprocessorWrapper(nn.Module):
    """
    Wraps NeMo AudioToMelSpectrogramPreprocessor for clean ONNX export.
    """
    def __init__(self, preprocessor: nn.Module):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        features, feature_lengths = self.preprocessor(input_signal=audio_signal, length=length)
        return features, feature_lengths


def extract_nemo_archive(nemo_path: str, extract_to: str):
    """Extracts .nemo tar archive or verifies existing directory."""
    if os.path.isdir(nemo_path):
        logger.info(f"Input path '{nemo_path}' is already a directory.")
        return nemo_path

    logger.info(f"Extracting '{nemo_path}' to temporary directory '{extract_to}'...")
    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extractall(path=extract_to)
    return extract_to


def find_sanskrit_tokenizer_file(cfg: dict, extract_dir: str) -> str:
    """Finds the Sanskrit sentencepiece .model file in the extracted archive."""
    tok_cfg = cfg.get("tokenizer", {})

    # Check langs dict (AI4Bharat IndicConformer format)
    langs = tok_cfg.get("langs", {})
    if "sa" in langs:
        sa_model = langs["sa"].get("model_path", "").replace("nemo:", "").strip()
        full_path = os.path.join(extract_dir, sa_model)
        if os.path.exists(full_path):
            return full_path

    # Check tokenizers_dict (older NeMo multilingual format)
    tok_dict = tok_cfg.get("tokenizers_dict", {})
    if "sa" in tok_dict:
        sa_model = tok_dict["sa"].get("model_path", "").replace("nemo:", "").strip()
        full_path = os.path.join(extract_dir, sa_model)
        if os.path.exists(full_path):
            return full_path

    # Check monolingual model_path
    if "model_path" in tok_cfg:
        sa_model = tok_cfg["model_path"].replace("nemo:", "").strip()
        full_path = os.path.join(extract_dir, sa_model)
        if os.path.exists(full_path):
            return full_path

    # Fallback search for any .model file in extract_dir
    model_files = [f for f in os.listdir(extract_dir) if f.endswith(".model")]
    if model_files:
        logger.warning(f"Could not find explicit 'sa' tokenizer in config. Using: {model_files[0]}")
        return os.path.join(extract_dir, model_files[0])

    raise FileNotFoundError(f"No SentencePiece tokenizer (.model) found in {extract_dir}")


def export_vocabulary_json(sp_model_path: str, output_json_path: str) -> list:
    """
    Extracts vocabulary from SentencePiece model and saves sanskrit_vocab.json
    with <blank> at index 0 followed by tokenizer pieces.
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    vocab = ["<blank>"]
    for i in range(sp.get_piece_size()):
        vocab.append(sp.id_to_piece(i))

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved Sanskrit vocabulary ({len(vocab)} tokens) to {output_json_path}")
    return vocab


def convert_nemo(
    nemo_path: str,
    output_dir: str,
    export_preprocessor: bool = True,
    opset_version: int = 18
):
    """
    Converts .nemo to INT8 Quantized ONNX (model_asr.onnx).
    The temporary FP32 ONNX model is deleted immediately after quantization.
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="nemo_onnx_")

    try:
        extract_dir = extract_nemo_archive(nemo_path, temp_dir)

        # 1. Load config
        config_path = os.path.join(extract_dir, "model_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing model_config.yaml in {extract_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 2. Extract Vocabulary & Sanskrit Tokenizer
        sp_model_path = find_sanskrit_tokenizer_file(cfg, extract_dir)
        vocab_json_path = os.path.join(output_dir, "sanskrit_vocab.json")
        vocab = export_vocabulary_json(sp_model_path, vocab_json_path)

        # 3. Load Checkpoint Weights
        ckpt_path = os.path.join(extract_dir, "model_weights.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing model_weights.ckpt in {extract_dir}")

        logger.info("Loading PyTorch model weights...")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 4. Instantiate and export Preprocessor (if requested)
        if export_preprocessor and "preprocessor" in cfg:
            from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
            logger.info("Building and exporting preprocessor to ONNX...")
            prep_cfg = cfg["preprocessor"].copy()
            prep_cfg.pop("_target_", None)
            prep_module = AudioToMelSpectrogramPreprocessor(**prep_cfg).eval()

            prep_wrapper = PreprocessorWrapper(prep_module)
            prep_onnx_path = os.path.join(output_dir, "preprocessor.onnx")

            dummy_audio = torch.randn(1, 16000, dtype=torch.float32)
            dummy_len = torch.tensor([16000], dtype=torch.int64)

            torch.onnx.export(
                prep_wrapper,
                (dummy_audio, dummy_len),
                prep_onnx_path,
                input_names=["audio_signal", "length"],
                output_names=["features", "feature_lengths"],
                dynamic_axes={
                    "audio_signal": {0: "batch", 1: "samples"},
                    "length": {0: "batch"},
                    "features": {0: "batch", 2: "time"},
                    "feature_lengths": {0: "batch"},
                },
                opset_version=opset_version,
            )
            logger.info(f"Preprocessor ONNX saved to {prep_onnx_path} ({os.path.getsize(prep_onnx_path) / 1024:.1f} KB)")

        # 5. Build Conformer Encoder
        from nemo.collections.asr.modules import ConformerEncoder
        logger.info("Instantiating ConformerEncoder...")
        enc_cfg = cfg["encoder"].copy()
        enc_cfg.pop("_target_", None)
        encoder = ConformerEncoder(**enc_cfg).eval()

        enc_sd = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
        encoder.load_state_dict(enc_sd, strict=False)

        # 6. Extract Sanskrit CTC Projection Layer
        ctc_w = ckpt.get("ctc_decoder.decoder_layers.0.weight")
        if ctc_w is None:
            ctc_w = ckpt.get("decoder.decoder_layers.0.weight")

        ctc_b = ckpt.get("ctc_decoder.decoder_layers.0.bias")
        if ctc_b is None:
            ctc_b = ckpt.get("decoder.decoder_layers.0.bias")

        if ctc_w is None or ctc_b is None:
            raise KeyError("Could not find ctc_decoder or decoder weights in checkpoint")

        total_classes = ctc_w.shape[0]
        feat_in = ctc_w.shape[1]

        if total_classes == 5633 or total_classes == 5632:
            # Multilingual IndicConformer (AI4Bharat) layout:
            # Sanskrit slice has 256 tokens at offset 4096, blank is at index 5632
            logger.info("Detected multilingual IndicConformer CTC head. Extracting Sanskrit slice...")
            off = 4096
            v_size = 256
            blank_idx = 5632 if total_classes > 5632 else 0
            cols = [blank_idx] + list(range(off, off + v_size))
            head_w = ctc_w[cols].clone()
            head_b = ctc_b[cols].clone()
        elif total_classes == len(vocab):
            # Standard NeMo EncDecCTCModelBPE layout:
            # Row 0..255 are tokens, Row 256 is blank.
            # In our serving vocabulary, index 0 is <blank> followed by tokens 1..256.
            logger.info("Detected standard NeMo CTC head (257 classes with blank at end). Aligning to vocabulary...")
            cols = [total_classes - 1] + list(range(total_classes - 1))
            head_w = ctc_w[cols].clone()
            head_b = ctc_b[cols].clone()
        else:
            logger.warning(f"Unexpected CTC classes count: {total_classes}. Attempting to adapt to vocabulary length {len(vocab)}.")
            head_w = ctc_w[:len(vocab)].clone()
            head_b = ctc_b[:len(vocab)].clone()

        head = nn.Conv1d(feat_in, head_w.shape[0], kernel_size=1)
        head.weight.data = head_w
        head.bias.data = head_b
        head.eval()

        # 7. Construct Full Model
        full_model = SanskritConformerCTC(encoder, head).eval()

        # 8. Export Temporary FP32 ONNX in temp_dir (will NOT be kept on disk)
        temp_fp32_onnx = os.path.join(temp_dir, "temp_model_fp32.onnx")
        logger.info("Exporting Conformer-CTC model to temporary FP32 ONNX graph...")

        dummy_features = torch.randn(1, 80, 100, dtype=torch.float32)
        dummy_flen = torch.tensor([100], dtype=torch.int64)

        torch.onnx.export(
            full_model,
            (dummy_features, dummy_flen),
            temp_fp32_onnx,
            input_names=["audio_signal", "length"],
            output_names=["logits"],
            dynamic_axes={
                "audio_signal": {0: "batch", 2: "time"},
                "length": {0: "batch"},
                "logits": {0: "batch", 1: "time"},
            },
            opset_version=opset_version,
        )

        # 9. Dynamic INT8 Quantization (MatMul-only) -> model_asr.onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        final_quantized_onnx = os.path.join(output_dir, "model_asr.onnx")
        logger.info(f"Quantizing directly to INT8: {final_quantized_onnx} (FP32 model is not retained)...")

        quantize_dynamic(
            model_input=temp_fp32_onnx,
            model_output=final_quantized_onnx,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        int8_size_mb = os.path.getsize(final_quantized_onnx) / (1024 * 1024)
        logger.info(f"INT8 Quantized model saved to {final_quantized_onnx} ({int8_size_mb:.1f} MB)")

        # 10. Verification Pass with ONNX Runtime
        logger.info("Running verification pass using ONNX Runtime...")
        import onnxruntime as ort

        asr_sess = ort.InferenceSession(final_quantized_onnx, providers=["CPUExecutionProvider"])
        test_feats = np.random.randn(1, 80, 100).astype(np.float32)
        test_flen = np.array([100], dtype=np.int64)

        logits = asr_sess.run(None, {"audio_signal": test_feats, "length": test_flen})[0]
        logger.info(f"Verification output shape: {logits.shape} (Expected classes: {len(vocab)})")
        assert logits.shape[-1] == len(vocab), f"Class mismatch: {logits.shape[-1]} != {len(vocab)}"

        logger.info("=" * 60)
        logger.info("SUCCESS: Quantized model_asr.onnx generated successfully!")
        logger.info(f"Output files in {output_dir}:")
        for f in os.listdir(output_dir):
            if f.endswith((".onnx", ".json")):
                sz = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
                logger.info(f"  - {f} ({sz:.2f} MB)")
        logger.info("=" * 60)

    finally:
        # Deleting temp_dir cleans up all intermediate FP32 .onnx and .onnx.data files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo ASR model to Quantized model_asr.onnx")
    parser.add_argument(
        "--input_nemo", "-i",
        type=str,
        default="models/base_asr.nemo",
        help="Path to input .nemo checkpoint (default: models/base_asr.nemo)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="models",
        help="Directory to save exported ONNX models (default: models)"
    )
    parser.add_argument(
        "--no-preprocessor",
        action="store_true",
        help="Skip preprocessor ONNX export"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_nemo):
        logger.error(f"Input file '{args.input_nemo}' does not exist.")
        sys.exit(1)

    convert_nemo(
        nemo_path=args.input_nemo,
        output_dir=args.output_dir,
        export_preprocessor=not args.no_preprocessor,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
