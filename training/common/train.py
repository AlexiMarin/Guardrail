"""
Shared fine-tuning entry point used by training/{distilbert,deberta,qwen}/train.py.

Kept here instead of duplicated 3x so hyperparameter handling and metrics stay identical across
the 3-model comparison the README's report depends on (see docs/PLAN.md -> "Modelos candidatos").
Each per-model script just calls build_arg_parser()/run() with its own checkpoint.

Runs as a SageMaker Training Job with source_dir="training/" so `import common...` resolves as a
sibling package regardless of which model's train.py is the entry_point.
"""

import argparse
import json
import logging
import os
from pathlib import Path

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
    return parser


def run(args: argparse.Namespace, model_checkpoint: str | None = None) -> dict:
    model_checkpoint = model_checkpoint or args.model_checkpoint
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        # Base causal LMs (e.g. Qwen2.5) usually ship with no pad token; eos doubles as pad
        # for classification -- attention_mask still hides it from the loss/pooling.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_LABELS)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = tokenize_dataset(load_split(args.train_dir, "train"), tokenizer, args.max_length)
    val_ds = tokenize_dataset(load_split(args.val_dir, "val"), tokenizer, args.max_length)

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
        report_to="none",
    )

    trainer = Trainer(
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
