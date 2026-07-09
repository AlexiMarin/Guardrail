"""
Download the benchmark datasets from HuggingFace and save them to S3.

Runs as a SageMaker Processing Job: writes parquet files locally, SageMaker syncs them to S3.
Dataset ids and licenses are documented in data/schema.md.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# The processing container doesn't ship `datasets` and can't install a requirements.txt,
# so we grab it at runtime.
try:
    from datasets import load_dataset
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "datasets", "pyarrow"])
    from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# repo id + which configs to pull (None = the default config)
SOURCES = {
    # Attacks
    "jailbreakbench": {
        "repo_id": "JailbreakBench/JBB-Behaviors",
        "configs": ["behaviors", "judge_comparison"],
        "gated": False,
        "license": "mit",
    },
    "harmbench": {
        "repo_id": "walledai/HarmBench",
        # skip "copyright" — it's about copyright infringement, not jailbreaks
        "configs": ["standard", "contextual"],
        "gated": True,
        "license": "mit",
    },
    "hackaprompt": {
        "repo_id": "hackaprompt/hackaprompt-dataset",
        "configs": None,
        "gated": True,
        "license": "mit",
    },
    "gandalf": {
        "repo_id": "Lakera/gandalf_ignore_instructions",
        "configs": None,
        "gated": False,
        "license": "mit",
    },
    # Prompt injection
    "deepset_prompt_injections": {
        "repo_id": "deepset/prompt-injections",
        "configs": None,
        "gated": False,
        "license": "apache-2.0",
    },
    # Balanced guard-training mix
    "wildguardmix": {
        "repo_id": "allenai/wildguardmix",
        "configs": ["wildguardtrain", "wildguardtest"],
        "gated": True,
        "license": "odc-by",
    },
    # Benign-but-scary prompts, to catch false positives
    "xstest": {
        # use Paul/XSTest, not the gated walledai mirror
        "repo_id": "Paul/XSTest",
        "configs": None,
        "gated": False,
        "license": "cc-by-4.0",
    },
    "orbench": {
        "repo_id": "bench-llms/or-bench",
        "configs": ["or-bench-80k", "or-bench-hard-1k", "or-bench-toxic"],
        "gated": False,
        "license": "cc-by-4.0",
    },
    # Plain toxicity (not jailbreak) for contrast
    "toxicchat": {
        # non-commercial license — fine for a portfolio, not for a product
        "repo_id": "lmsys/toxic-chat",
        "configs": ["toxicchat0124"],
        "gated": False,
        "license": "cc-by-nc-4.0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output",
        help="Local directory the Processing Job syncs to S3 (one subfolder per source).",
    )
    return parser.parse_args()


def download_source(key: str, spec: dict, output_dir: Path, hf_token: str | None) -> None:
    if spec["gated"] and not hf_token:
        raise RuntimeError(
            f"{key} ({spec['repo_id']}) is gated on the Hub -- accept its terms on "
            "huggingface.co with your account, then set HF_TOKEN before running this job."
        )

    dest = output_dir / key
    dest.mkdir(parents=True, exist_ok=True)

    for config in spec["configs"] or [None]:
        logger.info("Downloading %s (config=%s, license=%s)", spec["repo_id"], config, spec["license"])
        dataset_dict = load_dataset(spec["repo_id"], config, token=hf_token)
        for split_name, split in dataset_dict.items():
            out_name = f"{config}_{split_name}" if config else split_name
            out_path = dest / f"{out_name}.parquet"
            split.to_parquet(str(out_path))
            logger.info("  wrote %s (%d rows) -> %s", split_name, len(split), out_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    hf_token = os.environ.get("HF_TOKEN")

    for key, spec in SOURCES.items():
        download_source(key, spec, output_dir, hf_token)


if __name__ == "__main__":
    main()
