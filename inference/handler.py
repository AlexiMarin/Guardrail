"""
Lambda handler for the public demo.

Runs the quantized DistilBERT (ONNX INT8) directly inside the Lambda -- no SageMaker endpoint, so it
scales to zero and costs nothing at rest. Sits behind API Gateway.

Request:  POST {"prompt": "..."}
Response: {"label": "benign|prompt_injection|jailbreak", "scores": {...}, "latency_ms": N}

The model + tokenizer are baked into the container image (see Dockerfile); loaded once at cold start.
"""

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

LABELS = ["benign", "prompt_injection", "jailbreak"]
MAX_LENGTH = 256
# reject oversized inputs before we spend any compute on them (the model only sees 256 tokens anyway)
MAX_PROMPT_CHARS = 20000

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.environ.get("LAMBDA_TASK_ROOT", "."), "model"))

# Cloudflare Turnstile: the portfolio UI solves an invisible challenge and sends the token; we verify
# it here so people can't script the endpoint for their own projects. Enforced only when a secret is set.
TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET", "")
TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

# loaded once per container, reused across warm invocations
_session = ort.InferenceSession(os.path.join(MODEL_DIR, "model_int8.onnx"), providers=["CPUExecutionProvider"])
_tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
_tokenizer.enable_truncation(max_length=MAX_LENGTH)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def classify(prompt: str) -> dict:
    enc = _tokenizer.encode(prompt)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
    logits = _session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0][0]
    scores = _softmax(logits)
    return {
        "label": LABELS[int(scores.argmax())],
        "scores": {label: float(s) for label, s in zip(LABELS, scores)},
    }


def _response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def verify_turnstile(token: str, ip: str) -> bool:
    data = urllib.parse.urlencode({"secret": TURNSTILE_SECRET, "response": token or "", "remoteip": ip}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(TURNSTILE_URL, data=data), timeout=5) as r:
            return bool(json.load(r).get("success"))
    except Exception:
        return False  # fail closed -- if we can't verify, don't run


def handler(event, context):
    try:
        payload = json.loads(event.get("body") or "{}")
    except ValueError:
        return _response(400, {"error": "body must be JSON with a 'prompt' field"})

    # gate on Turnstile before spending any compute (only when a secret is configured)
    if TURNSTILE_SECRET:
        source_ip = event.get("requestContext", {}).get("http", {}).get("sourceIp", "")
        if not verify_turnstile(payload.get("turnstile_token"), source_ip):
            return _response(403, {"error": "turnstile verification failed"})

    prompt = payload.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return _response(400, {"error": "missing 'prompt'"})
    if len(prompt) > MAX_PROMPT_CHARS:
        return _response(413, {"error": f"prompt too long (max {MAX_PROMPT_CHARS} chars)"})

    start = time.perf_counter()
    result = classify(prompt)
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return _response(200, result)
