"""
Qwen2.5-1.5B candidate -- small LLM fine-tuned as a classifier, Llama-Guard style
(see docs/PLAN.md -> Modelos candidatos). Smaller default batch size + bf16 vs. the
encoder candidates since it's ~10x their parameter count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable when run directly

from common.train import build_arg_parser, run  # noqa: E402

CHECKPOINT = "Qwen/Qwen2.5-1.5B"

if __name__ == "__main__":
    parser = build_arg_parser(CHECKPOINT)
    parser.set_defaults(train_batch_size=4, eval_batch_size=8, bf16=True)
    args = parser.parse_args()
    run(args)
