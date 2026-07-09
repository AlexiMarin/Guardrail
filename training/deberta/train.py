"""DeBERTa-v3-base candidate -- the stronger encoder, best accuracy expected among the two."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `common` imports when run directly

from common.train import build_arg_parser, run  # noqa: E402

CHECKPOINT = "microsoft/deberta-v3-base"

if __name__ == "__main__":
    args = build_arg_parser(CHECKPOINT).parse_args()
    run(args)
