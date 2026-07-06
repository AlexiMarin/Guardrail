"""
Build the training dataset from the raw sources downloaded by data/download.py.

Steps (see docs/PLAN.md -> Datasets and data/schema.md):
  1. Load each raw source from the local input dir (synced from S3 by the Processing Job).
  2. Map each source to the multiclass schema (benign / prompt_injection / jailbreak).
  3. Deduplicate ACROSS sources (AdvBench overlaps HarmBench & JailbreakBench).
  4. Top up mundane benign if the mix is too skewed toward "hard" benign (XSTest/OR-Bench).
  5. Hold out HackAPrompt entirely as the UNSEEN attack style, to measure generalization.
  6. Split the remainder into train / val / test.
  7. Write curated splits to the local output dir (synced to S3 by the Processing Job).

Runs as a SageMaker Processing Job.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS = {"benign": 0, "prompt_injection": 1, "jailbreak": 2}

# Held out entirely from train/val/test -- a stylistically distinct attack (competition-style
# injection-to-leak-a-secret) used only to measure generalization to unseen attack styles.
UNSEEN_ATTACK_SOURCES = {"hackaprompt"}

# Top-up source for mundane benign if the mix skews too hard (see docs/PLAN.md "Negativos mundanos").
# CC-BY-NC-4.0 (generated with text-davinci-003 outputs) -- fine for this portfolio/demo, not resale.
MUNDANE_BENIGN_REPO_ID = "tatsu-lab/alpaca"
TARGET_MUNDANE_BENIGN_FRACTION = 0.6
MUNDANE_BENIGN_SOURCES = {"wildguardmix", "toxicchat"}


def _read_all(input_dir: Path, source: str, pattern: str = "*.parquet") -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted((input_dir / source).glob(pattern))]
    if not frames:
        raise FileNotFoundError(f"No parquet files found for source={source!r} under {input_dir / source}")
    return pd.concat(frames, ignore_index=True)


def load_jailbreakbench(input_dir: Path) -> pd.DataFrame:
    src = input_dir / "jailbreakbench"
    harmful = pd.read_parquet(src / "behaviors_harmful.parquet")
    benign = pd.read_parquet(src / "behaviors_benign.parquet")
    harmful = harmful.rename(columns={"Goal": "text"})[["text"]].assign(label=LABELS["jailbreak"])
    benign = benign.rename(columns={"Goal": "text"})[["text"]].assign(label=LABELS["benign"])
    return pd.concat([harmful, benign], ignore_index=True)


def load_harmbench(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "harmbench")
    text_col = next((c for c in ("prompt", "text", "Behavior", "behavior") if c in df.columns), df.columns[0])
    # Direct harmful requests without adversarial framing -> folded into `jailbreak`
    # (see the "Open labeling decision" this closes in data/schema.md).
    return df.rename(columns={text_col: "text"})[["text"]].assign(label=LABELS["jailbreak"])


def load_hackaprompt(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "hackaprompt")
    text_col = "user_input" if "user_input" in df.columns else df.columns[0]
    # data/schema.md assigns HackAPrompt to `jailbreak` (its typical-sources column).
    return df.rename(columns={text_col: "text"})[["text"]].assign(label=LABELS["jailbreak"])


def load_gandalf(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "gandalf")
    # Every row is a password-extraction attempt against the Gandalf game -- no benign rows here.
    return df[["text"]].assign(label=LABELS["prompt_injection"])


def load_deepset(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "deepset_prompt_injections")
    df["label"] = df["label"].map({1: LABELS["prompt_injection"], 0: LABELS["benign"]})
    return df[["text", "label"]]


def load_wildguardmix(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "wildguardmix")
    df = df.rename(columns={"prompt": "text"})
    # General harm annotation, not injection-specific -- harmful folds into `jailbreak` (see harmbench note).
    df["label"] = df["prompt_harm_label"].map({"harmful": LABELS["jailbreak"], "unharmful": LABELS["benign"]})
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


def load_xstest(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "xstest")
    df = df.rename(columns={"prompt": "text"})
    # "safe" = risky-sounding-but-benign (the false-positive control this dataset exists for).
    # "unsafe" = genuinely unsafe direct requests -> folded into `jailbreak`.
    df["label"] = df["label"].map({"safe": LABELS["benign"], "unsafe": LABELS["jailbreak"]})
    return df[["text", "label"]]


def load_orbench(input_dir: Path) -> pd.DataFrame:
    label_by_config = {
        "or-bench-80k": LABELS["benign"],
        "or-bench-hard-1k": LABELS["benign"],
        "or-bench-toxic": LABELS["jailbreak"],  # genuinely harmful contrast set, not benign
    }
    frames = []
    for config, label in label_by_config.items():
        df = _read_all(input_dir, "orbench", pattern=f"{config}_*.parquet")
        frames.append(df.rename(columns={"prompt": "text"})[["text"]].assign(label=label))
    return pd.concat(frames, ignore_index=True)


def load_toxicchat(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "toxicchat")
    df = df.rename(columns={"user_input": "text"})
    # Toxicity alone isn't this guardrail's concern -- only the native `jailbreaking` flag sets the label.
    df["label"] = df["jailbreaking"].map({1: LABELS["jailbreak"], 0: LABELS["benign"]})
    return df[["text", "label"]]


LOADERS = {
    "jailbreakbench": load_jailbreakbench,
    "harmbench": load_harmbench,
    "hackaprompt": load_hackaprompt,
    "gandalf": load_gandalf,
    "deepset_prompt_injections": load_deepset,
    "wildguardmix": load_wildguardmix,
    "xstest": load_xstest,
    "orbench": load_orbench,
    "toxicchat": load_toxicchat,
}


def build_all_sources(input_dir: Path) -> pd.DataFrame:
    frames = []
    for source, loader in LOADERS.items():
        df = loader(input_dir)
        df = df.assign(source=source)
        logger.info("%s: %d rows, label counts=%s", source, len(df), df["label"].value_counts().to_dict())
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    key = df["text"].str.strip().str.lower()
    before = len(df)
    df = df[~key.duplicated(keep="first")]
    logger.info(
        "Deduplicated %d -> %d rows (%d dropped; catches e.g. AdvBench overlap between HarmBench/JailbreakBench)",
        before, len(df), before - len(df),
    )
    return df.reset_index(drop=True)


def top_up_mundane_benign(df: pd.DataFrame, target_fraction: float, seed: int, hf_token: str | None) -> pd.DataFrame:
    benign = df[df["label"] == LABELS["benign"]]
    if len(benign) == 0:
        return df

    mundane_fraction = benign["source"].isin(MUNDANE_BENIGN_SOURCES).mean()
    logger.info("Mundane benign fraction before top-up: %.2f (target %.2f)", mundane_fraction, target_fraction)
    if mundane_fraction >= target_fraction:
        return df

    n_mundane = int(benign["source"].isin(MUNDANE_BENIGN_SOURCES).sum())
    n_needed = max(int(target_fraction * len(benign)) - n_mundane, 0)
    if n_needed == 0:
        return df

    logger.info("Topping up %d mundane benign rows from %s", n_needed, MUNDANE_BENIGN_REPO_ID)
    alpaca = load_dataset(MUNDANE_BENIGN_REPO_ID, token=hf_token)["train"].to_pandas()
    alpaca = alpaca.sample(n=min(n_needed, len(alpaca)), random_state=seed)
    alpaca["text"] = (alpaca["instruction"].fillna("") + " " + alpaca["input"].fillna("")).str.strip()
    alpaca = alpaca[["text"]].assign(label=LABELS["benign"], source="alpaca")
    return pd.concat([df, alpaca], ignore_index=True)


def split_dataset(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> dict[str, pd.DataFrame]:
    unseen_mask = df["source"].isin(UNSEEN_ATTACK_SOURCES)
    unseen_attacks = df[unseen_mask].reset_index(drop=True)
    pool = df[~unseen_mask].reset_index(drop=True)

    # Match a benign contrast sample into the unseen set so it's usable for FPR too, not just recall.
    benign_pool = pool[pool["label"] == LABELS["benign"]]
    n_contrast = min(len(unseen_attacks), len(benign_pool))
    contrast = benign_pool.sample(n=n_contrast, random_state=seed)
    unseen_attacks = pd.concat([unseen_attacks, contrast], ignore_index=True)
    pool = pool.drop(contrast.index).reset_index(drop=True)

    train_val, test = train_test_split(pool, test_size=test_size, stratify=pool["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size), stratify=train_val["label"], random_state=seed
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "unseen_attacks": unseen_attacks.reset_index(drop=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--mundane-benign-target", type=float, default=TARGET_MUNDANE_BENIGN_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    hf_token = os.environ.get("HF_TOKEN")

    df = build_all_sources(input_dir)
    logger.info("Loaded %d rows across %d sources", len(df), df["source"].nunique())

    df = dedupe(df)
    df = top_up_mundane_benign(df, args.mundane_benign_target, args.seed, hf_token)

    splits = split_dataset(df, args.val_size, args.test_size, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        out_path = output_dir / f"{name}.parquet"
        split_df[["text", "label", "source"]].to_parquet(out_path)
        logger.info(
            "%s: %d rows, label counts=%s -> %s",
            name, len(split_df), split_df["label"].value_counts().to_dict(), out_path,
        )


if __name__ == "__main__":
    main()
