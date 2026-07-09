"""
Export the trained DistilBERT to ONNX, quantize it to INT8, and check what we lost.

For the demo the model runs in a small Lambda, so it has to be small and CPU-fast. This exports to
ONNX, applies dynamic INT8 quantization, then reports three things that decide whether it's Lambda-
viable: final artifact size, batch=1 CPU latency (p50/p99), and how much accuracy the quantization
cost vs the original model on the test set.

Runs as a SageMaker Processing Job on a CPU instance (x86, like Lambda -- so the latency is
representative). Reads the trained model + test set from S3, writes the quantized model + tokenizer
+ metrics back to S3.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENIGN = 0
ATTACK_LABELS = {1: "prompt_injection", 2: "jailbreak"}


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Same definitions as evaluation/evaluate.py -- keep in sync."""
    m = {"macro_f1": float(f1_score(labels, preds, average="macro"))}
    present = [l for l in ATTACK_LABELS if (labels == l).any()]
    for label_id, r in zip(present, recall_score(labels, preds, labels=present, average=None, zero_division=0)):
        m[f"recall_{ATTACK_LABELS[label_id]}"] = float(r)
    benign = labels == BENIGN
    attack = labels != BENIGN
    m["false_positive_rate"] = float((preds[benign] != BENIGN).mean()) if benign.any() else None
    m["binary_attack_recall"] = float((preds[attack] != BENIGN).mean()) if attack.any() else None
    return m


def maybe_extract_model(model_dir: Path) -> None:
    """SageMaker downloads model.tar.gz but doesn't unpack it -- do it here."""
    if (model_dir / "config.json").exists():
        return
    tarball = model_dir / "model.tar.gz"
    if tarball.exists():
        import tarfile

        logger.info("Extracting %s", tarball)
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(model_dir)


class LogitsOnly(torch.nn.Module):
    # torch.onnx.export wants plain tensors, but HF models return a dataclass -- unwrap the logits.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(model, out_path: Path, max_length: int) -> None:
    dummy = {
        "input_ids": torch.ones(1, max_length, dtype=torch.long),
        "attention_mask": torch.ones(1, max_length, dtype=torch.long),
    }
    torch.onnx.export(
        LogitsOnly(model).eval(),
        (dummy["input_ids"], dummy["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # let both batch and sequence length vary at inference time
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


def predict(sess, tokenizer, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    preds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="np")
        logits = sess.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})[0]
        preds.append(logits.argmax(axis=-1))
    return np.concatenate(preds)


def measure_latency(sess, tokenizer, texts: list[str], max_length: int, n: int) -> dict:
    """Batch=1 CPU latency, the real hot-path case in a Lambda."""
    sample = texts[:n]
    for t in sample[: min(5, len(sample))]:  # warm up
        enc = tokenizer([t], truncation=True, max_length=max_length, return_tensors="np")
        sess.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    lat_ms = []
    for t in sample:
        enc = tokenizer([t], truncation=True, max_length=max_length, return_tensors="np")
        start = time.perf_counter()
        sess.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        lat_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(lat_ms)
    return {"n": len(arr), "p50_ms": float(np.percentile(arr, 50)), "p99_ms": float(np.percentile(arr, 99)), "mean_ms": float(arr.mean())}


def mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/opt/ml/processing/model")
    parser.add_argument("--data-dir", default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_extract_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    fp32_path = out_dir / "model.onnx"
    int8_path = out_dir / "model_int8.onnx"

    logger.info("Exporting to ONNX...")
    export_onnx(model, fp32_path, args.max_length)
    logger.info("Quantizing to INT8...")
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    sizes = {
        "safetensors_fp32_mb": mb(model_dir / "model.safetensors"),
        "onnx_fp32_mb": mb(fp32_path),
        "onnx_int8_mb": mb(int8_path),
    }
    logger.info("Sizes: %s", sizes)

    # accuracy: torch baseline vs onnx int8, on the full test set
    df = pd.read_parquet(Path(args.data_dir) / "test.parquet")
    texts, labels = df["text"].tolist(), df["label"].to_numpy()

    with torch.no_grad():
        torch_preds = []
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            enc = tokenizer(batch, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
            torch_preds.append(model(**enc).logits.argmax(dim=-1).numpy())
        torch_preds = np.concatenate(torch_preds)

    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    int8_preds = predict(int8_sess, tokenizer, texts, args.max_length, args.batch_size)

    results = {
        "sizes": sizes,
        "test_metrics": {
            "torch_fp32": compute_metrics(torch_preds, labels),
            "onnx_int8": compute_metrics(int8_preds, labels),
        },
        "int8_latency_cpu_batch1": measure_latency(int8_sess, tokenizer, texts, args.max_length, args.latency_samples),
        "n_test": int(len(labels)),
    }
    logger.info("Results: %s", results)

    (out_dir / "quantization_metrics.json").write_text(json.dumps(results, indent=2))
    tokenizer.save_pretrained(out_dir)  # ship the tokenizer next to the model for the Lambda
    fp32_path.unlink()  # only the INT8 model is the deploy artifact; drop the fp32 ONNX

    print("===QUANT_RESULTS_JSON_BEGIN===")
    print(json.dumps(results))
    print("===QUANT_RESULTS_JSON_END===")


if __name__ == "__main__":
    main()
