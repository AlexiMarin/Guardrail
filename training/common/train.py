"""
Shared fine-tuning code used by training/{distilbert,deberta,qwen}/train.py.

Kept in one place so all 3 models train the exact same way -- only the checkpoint changes.
Each per-model script just calls build_arg_parser()/run() with its own checkpoint. Runs as a
SageMaker Training Job with source_dir="training/" so `import common...` works.
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common.data import load_split, tokenize_dataset
from common.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_LABELS = 3


def compute_class_weights(train_ds, num_labels: int) -> torch.Tensor:
    """Inverse-frequency weights (sklearn's 'balanced' formula) for the loss.

    prompt_injection is only ~0.85% of the train set, so without weighting the model would mostly
    ignore it and still look accurate overall. This makes the loss care about the rare class.
    """
    # int() instead of .tolist() -- the newer datasets Column type doesn't have .tolist() and crashed here.
    counts = Counter(int(x) for x in train_ds["labels"])
    total = sum(counts.values())
    weights = [total / (num_labels * counts.get(i, 1)) for i in range(num_labels)]
    logger.info("Class weights (label -> weight): %s", dict(enumerate(weights)))
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    """Trainer with a class-weighted CrossEntropyLoss -- see compute_class_weights()."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Match logits' device AND dtype -- bf16 models (Qwen) give bf16 logits but class_weights is
        # float32, so a plain .to(device) leaves a dtype mismatch that crashes.
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(device=logits.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_arg_parser(default_checkpoint: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", default=default_checkpoint)
    parser.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--val-dir", default=os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Mixed-precision training (recommended for Qwen).")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Saves memory at the cost of compute -- needed to fit Qwen on a 24GB GPU.",
    )
    parser.add_argument(
        "--optim", default="adamw_torch",
        help="Trainer optimizer. 'adafactor' uses much less memory than adamw -- needed for Qwen on 24GB.",
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Subsample train set -- for a quick smoke test, not a real run.",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None,
        help="Subsample val set -- for a quick smoke test, not a real run.",
    )
    return parser


def run(args: argparse.Namespace, model_checkpoint: str | None = None) -> dict:
    model_checkpoint = model_checkpoint or args.model_checkpoint
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        # base LMs like Qwen have no pad token; reuse eos (attention_mask still masks it out)
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_LABELS)
    if args.gradient_checkpointing:
        # KV-cache clashes with gradient checkpointing; we don't need it here anyway
        model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_raw = load_split(args.train_dir, "train")
    val_raw = load_split(args.val_dir, "val")
    if args.max_train_samples is not None:
        train_raw = train_raw.select(range(min(len(train_raw), args.max_train_samples)))
    if args.max_eval_samples is not None:
        val_raw = val_raw.select(range(min(len(val_raw), args.max_eval_samples)))

    train_ds = tokenize_dataset(train_raw, tokenizer, args.max_length)
    val_ds = tokenize_dataset(val_raw, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to="none",
    )

    class_weights = compute_class_weights(train_ds, NUM_LABELS)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    output_data_dir = Path(args.output_data_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    (output_data_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))
    logger.info("Eval metrics: %s", eval_metrics)

    return eval_metrics
