"""
Project Medhā - MLflow Model Packaging, Registry & Serving Interface
====================================================================
Packages the quantized Su-śrotā Sanskrit/Prakrit ASR model as a standard
MLflow `pyfunc` (PythonModel) with explicit signature and input examples,
registers it in the MLflow Model Registry, and assigns the `champion` alias.

This enables serving the production model via:
    mlflow models serve -m "models:/sushrota-asr@champion" -p 5001 --env-manager local --host 0.0.0.0

Endpoints exposed by MLflow serving:
    - GET  /ping         -> 200 OK (Readiness probe)
    - GET  /version      -> MLflow server version
    - POST /invocations  -> Inference endpoint (JSON in -> JSON out)

Input payload format for /invocations:
    {
      "dataframe_records": [
        {"audio_path": "data/processed_audio/clip_sample.wav"}
      ]
    }
    OR Base64 encoded audio:
    {
      "dataframe_records": [
        {"audio_base64": "<base64_encoded_wav_bytes>"}
      ]
    }

Usage:
    python scripts/register_mlflow_model.py \
        --model-dir models \
        --model-name sushrota-asr \
        --alias champion \
        --mlflow-uri sqlite:///mlflow.db
"""

import os
import sys
import io
import json
import base64
import argparse
import logging
from typing import Dict, Any

import numpy as np
import soundfile as sf
import librosa
import pandas as pd
import onnxruntime as ort

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow import MlflowClient

# Prevent protobuf warnings
os.environ["TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("mlflow_register")


class SushrotaASRPyfunc(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for quantized Su-śrotā Conformer-CTC Sanskrit/Prakrit ASR.
    Supports file paths, raw numpy audio, or base64-encoded audio clips.
    """
    def load_context(self, context):
        """Initializes ONNX Runtime sessions for preprocessor and acoustic model."""
        prep_path = context.artifacts["preprocessor"]
        asr_path = context.artifacts["model_asr"]
        vocab_path = context.artifacts["vocab"]

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4

        self.prep_session = ort.InferenceSession(prep_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.asr_session = ort.InferenceSession(asr_path, sess_options=opts, providers=['CPUExecutionProvider'])

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

    def _decode_audio(self, row: Dict[str, Any]):
        """Loads and normalizes audio to 16kHz mono float32."""
        if row.get("audio_base64"):
            audio_bytes = base64.b64decode(row["audio_base64"])
            wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        elif row.get("audio_path"):
            wav, sr = sf.read(row["audio_path"], dtype="float32")
        else:
            raise ValueError("Row must contain either 'audio_path' or 'audio_base64'.")

        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        duration = len(wav) / 16000.0
        return wav, duration

    def _transcribe(self, wav: np.ndarray) -> str:
        """Runs ONNX preprocessor + ASR and greedy CTC decodes logits."""
        # 1. Feature extraction
        signal = np.expand_dims(wav, axis=0).astype(np.float32)
        length = np.array([signal.shape[1]], dtype=np.int64)

        prep_inputs = {
            self.prep_session.get_inputs()[0].name: signal,
            self.prep_session.get_inputs()[1].name: length
        }
        features, feat_length = self.prep_session.run(None, prep_inputs)

        # 2. Acoustic model
        asr_inputs = {
            self.asr_session.get_inputs()[0].name: features,
            self.asr_session.get_inputs()[1].name: feat_length
        }
        logits = self.asr_session.run(None, asr_inputs)[0]  # [1, time, classes]

        # 3. Greedy CTC decoding: argmax -> collapse repeats -> remove blank (idx 0)
        pred_ids = np.argmax(logits[0], axis=-1)
        out, prev = [], -1
        for i in pred_ids:
            i = int(i)
            if i != prev and i != 0:
                if i < len(self.vocab):
                    out.append(self.vocab[i])
            prev = i

        # Replace SentencePiece whitespace token with standard space
        transcription = "".join(out).replace(" ", " ").replace("▁", " ").strip()
        return transcription

    def predict(self, context, model_input, params=None):
        """
        Accepts DataFrame or list of dicts with 'audio_path' or 'audio_base64'.
        Returns DataFrame with 'transcription', 'audio_duration_seconds', and 'status'.
        """
        if isinstance(model_input, pd.DataFrame):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, list):
            records = model_input
        elif isinstance(model_input, dict):
            if "dataframe_records" in model_input:
                records = model_input["dataframe_records"]
            else:
                records = [model_input]
        else:
            raise TypeError(f"Unsupported model_input type: {type(model_input)}")

        transcriptions = []
        durations = []

        for rec in records:
            try:
                wav, dur = self._decode_audio(rec)
                text = self._transcribe(wav)
                transcriptions.append(text)
                durations.append(round(dur, 3))
            except Exception as e:
                transcriptions.append(f"ERROR: {str(e)}")
                durations.append(0.0)

        return pd.DataFrame({
            "transcription": transcriptions,
            "audio_duration_seconds": durations,
            "status": ["success" if not t.startswith("ERROR:") else "error" for t in transcriptions]
        })


def main():
    parser = argparse.ArgumentParser(description="Package & Register Sushrota ASR with MLflow Model Registry")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with ONNX models")
    parser.add_argument("--model-name", type=str, default="sushrota-asr", help="Registered model name in Registry")
    parser.add_argument("--alias", type=str, default="champion", help="Model Registry alias (e.g. champion)")
    parser.add_argument("--experiment-name", type=str, default="medha-model-serving", help="MLflow experiment name")
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db", help="MLflow SQLite tracking URI")
    args = parser.parse_args()

    prep_path = os.path.join(args.model_dir, "preprocessor.onnx")
    asr_path = os.path.join(args.model_dir, "model_asr.onnx")
    vocab_path = os.path.join(args.model_dir, "sanskrit_vocab.json")

    for p in [prep_path, asr_path, vocab_path]:
        if not os.path.exists(p):
            logger.error(f"Required artifact '{p}' not found! Run ONNX export first.")
            sys.exit(1)

    # 1. Setup MLflow Tracking & Registry Client
    logger.info(f"Connecting to MLflow Tracking at '{args.mlflow_uri}'...")
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    client = MlflowClient(args.mlflow_uri)

    # 2. Build Schema Signature with optional inputs
    input_schema = Schema([
        ColSpec("string", "audio_path", required=False),
        ColSpec("string", "audio_base64", required=False)
    ])
    output_schema = Schema([
        ColSpec("string", "transcription"),
        ColSpec("double", "audio_duration_seconds"),
        ColSpec("string", "status")
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Input example
    sample_wav_path = "data/val_manifest.json"
    if os.path.exists("data/val_manifest.json"):
        with open("data/val_manifest.json", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if os.path.exists(item["audio_filepath"]):
                    sample_wav_path = item["audio_filepath"]
                    break

    input_example = pd.DataFrame([{
        "audio_path": sample_wav_path,
        "audio_base64": ""
    }])

    # 3. Log pyfunc model and register to Registry
    logger.info(f"Packaging pyfunc model and registering under name '{args.model_name}'...")
    artifacts = {
        "preprocessor": prep_path,
        "model_asr": asr_path,
        "vocab": vocab_path
    }
    prep_data_path = prep_path + ".data"
    if os.path.exists(prep_data_path):
        artifacts["preprocessor.onnx.data"] = prep_data_path

    with mlflow.start_run(run_name="package-pyfunc-serving"):
        mlflow.log_param("architecture", "Sanskrit-Conformer-CTC-Quantized")
        mlflow.log_param("inference_runtime", "onnxruntime")
        mlflow.log_param("sample_rate", 16000)

        info = mlflow.pyfunc.log_model(
            name="sushrota_asr_model",
            python_model=SushrotaASRPyfunc(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "onnxruntime>=1.16.0",
                "numpy>=1.24.0",
                "soundfile>=0.12.0",
                "librosa>=0.10.0",
                "pandas>=2.0.0"
            ],
            registered_model_name=args.model_name
        )
        logger.info(f"Model successfully packaged and logged! Model URI: {info.model_uri}")

    # 4. Set Registry Alias
    versions = client.search_model_versions(f"name='{args.model_name}'")
    latest_version = max(int(v.version) for v in versions)
    logger.info(f"Assigning alias '{args.alias}' to model version {latest_version}...")
    client.set_registered_model_alias(args.model_name, args.alias, str(latest_version))

    served_model_uri = f"models:/{args.model_name}@{args.alias}"
    logger.info("=" * 65)
    logger.info("MLFLOW MODEL REGISTRY & SERVING SETUP COMPLETE!")
    logger.info("=" * 65)
    logger.info(f"Registered Model:   {args.model_name} (Version {latest_version})")
    logger.info(f"Assigned Alias:     {args.alias} -> Version {latest_version}")
    logger.info(f"Model Serving URI:  {served_model_uri}")
    logger.info("To serve the model with MLflow, run:")
    logger.info(f"    mlflow models serve -m \"{served_model_uri}\" -p 5001 --env-manager local --host 0.0.0.0")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
