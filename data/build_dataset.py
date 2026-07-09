"""
Turn the raw datasets from download.py into train/val/test splits.

Loads each source, maps it to our 3 labels, dedupes, tops up plain-benign examples, holds out
HackAPrompt as an unseen-attack test, then splits the rest. Runs as a SageMaker Processing Job.
Label decisions live in data/schema.md.
"""

import argparse
import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS = {"benign": 0, "prompt_injection": 1, "jailbreak": 2}

# HackAPrompt is held out completely — it's our "unseen attack style" for measuring generalization.
UNSEEN_ATTACK_SOURCES = {"hackaprompt"}

# Extra plain-benign examples, added only if our benign mix leans too much on the scary-sounding
# kind (XSTest/OR-Bench). Alpaca, non-commercial license — fine for a portfolio.
MUNDANE_BENIGN_REPO_ID = "tatsu-lab/alpaca"
# direct parquet link (from HF's datasets-server); fetched with plain HTTP, see top_up_mundane_benign
MUNDANE_BENIGN_PARQUET_URL = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
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
    # plain harmful requests, no disguise — we still label them jailbreak (see schema.md)
    return df.rename(columns={text_col: "text"})[["text"]].assign(label=LABELS["jailbreak"])


def load_hackaprompt(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "hackaprompt")
    text_col = "user_input" if "user_input" in df.columns else df.columns[0]
    # labeled jailbreak per schema.md
    return df.rename(columns={text_col: "text"})[["text"]].assign(label=LABELS["jailbreak"])


def load_gandalf(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "gandalf")
    # every row is a password-extraction attempt, all injection
    return df[["text"]].assign(label=LABELS["prompt_injection"])


def load_deepset(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "deepset_prompt_injections")
    df["label"] = df["label"].map({1: LABELS["prompt_injection"], 0: LABELS["benign"]})
    return df[["text", "label"]]


def load_wildguardmix(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "wildguardmix")
    df = df.rename(columns={"prompt": "text"})
    # harmful here is a general flag, folds into jailbreak like harmbench
    df["label"] = df["prompt_harm_label"].map({"harmful": LABELS["jailbreak"], "unharmful": LABELS["benign"]})
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


def load_xstest(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "xstest")
    df = df.rename(columns={"prompt": "text"})
    # safe = scary-sounding but actually benign (the whole point of this set); unsafe = real, folds into jailbreak
    df["label"] = df["label"].map({"safe": LABELS["benign"], "unsafe": LABELS["jailbreak"]})
    return df[["text", "label"]]


def load_orbench(input_dir: Path) -> pd.DataFrame:
    label_by_config = {
        "or-bench-80k": LABELS["benign"],
        "or-bench-hard-1k": LABELS["benign"],
        "or-bench-toxic": LABELS["jailbreak"],  # actually harmful, not benign
    }
    frames = []
    for config, label in label_by_config.items():
        df = _read_all(input_dir, "orbench", pattern=f"{config}_*.parquet")
        frames.append(df.rename(columns={"prompt": "text"})[["text"]].assign(label=label))
    return pd.concat(frames, ignore_index=True)


def load_toxicchat(input_dir: Path) -> pd.DataFrame:
    df = _read_all(input_dir, "toxicchat")
    df = df.rename(columns={"user_input": "text"})
    # we only care about the jailbreaking flag here, not plain toxicity
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
    # catches the same prompts appearing in more than one source (e.g. AdvBench in both HarmBench and JailbreakBench)
    logger.info("Deduplicated %d -> %d rows (%d dropped)", before, len(df), before - len(df))
    return df.reset_index(drop=True)


def top_up_mundane_benign(df: pd.DataFrame, target_fraction: float, seed: int) -> pd.DataFrame:
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
    # Grab the parquet over plain HTTP instead of datasets.load_dataset -- installing datasets/pyarrow
    # in the container kept breaking, while pd.read_parquet on the preinstalled pyarrow just works.
    # Alpaca is public, so no token needed.
    with urllib.request.urlopen(MUNDANE_BENIGN_PARQUET_URL) as response:
        alpaca = pd.read_parquet(io.BytesIO(response.read()))
    alpaca = alpaca.sample(n=min(n_needed, len(alpaca)), random_state=seed)
    alpaca["text"] = (alpaca["instruction"].fillna("") + " " + alpaca["input"].fillna("")).str.strip()
    alpaca = alpaca[["text"]].assign(label=LABELS["benign"], source="alpaca")
    return pd.concat([df, alpaca], ignore_index=True)


def split_dataset(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> dict[str, pd.DataFrame]:
    unseen_mask = df["source"].isin(UNSEEN_ATTACK_SOURCES)
    unseen_attacks = df[unseen_mask].reset_index(drop=True)
    pool = df[~unseen_mask].reset_index(drop=True)

    # Add some benign rows to the unseen set so we can measure false positives there too, not just
    # recall. Cap at 20% of the benign pool -- HackAPrompt is huge, so a 1:1 match would eat all our
    # benign data and leave none for train/val/test (which is exactly what happened the first time).
    benign_pool = pool[pool["label"] == LABELS["benign"]]
    n_contrast = min(len(unseen_attacks), len(benign_pool) // 5)
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

    df = build_all_sources(input_dir)
    logger.info("Loaded %d rows across %d sources", len(df), df["source"].nunique())

    df = dedupe(df)
    df = top_up_mundane_benign(df, args.mundane_benign_target, args.seed)

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
