"""DistilBERT candidate -- low-cost/latency reference (see docs/PLAN.md -> Modelos candidatos)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable when run directly

from common.train import build_arg_parser, run  # noqa: E402

CHECKPOINT = "distilbert-base-uncased"

if __name__ == "__main__":
    args = build_arg_parser(CHECKPOINT).parse_args()
    run(args)
