"""
Qwen2.5-1.5B candidate -- a small LLM fine-tuned as a classifier. Smaller batch + bf16 than the
encoders since it's ~10x their size.

Fitting it on a single 24GB GPU took two tricks: gradient_checkpointing (kept OOMing by a hair
without it), and adafactor (the real memory hog was adamw's optimizer state, ~12GB for 1.5B params;
adafactor keeps way less).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `common` imports when run directly

from common.train import build_arg_parser, run  # noqa: E402

CHECKPOINT = "Qwen/Qwen2.5-1.5B"

if __name__ == "__main__":
    parser = build_arg_parser(CHECKPOINT)
    parser.set_defaults(
        train_batch_size=4, eval_batch_size=8, bf16=True, gradient_checkpointing=True, optim="adafactor"
    )
    args = parser.parse_args()
    run(args)
