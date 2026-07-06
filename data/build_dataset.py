"""
Build the training dataset from the raw sources in S3.

Steps (see docs/PLAN.md -> Datasets and data/schema.md):
  1. Load raw sources from S3.
  2. Map each source to the multiclass schema (benign / prompt_injection / jailbreak).
  3. Deduplicate ACROSS sources (AdvBench overlaps HarmBench & JailbreakBench).
  4. Split train / val / test + a held-out set of UNSEEN attack styles for generalization.
  5. Balance the benign mix (mostly mundane, minority hard). Top up with Alpaca/Dolly only if scarce.
  6. Write curated splits back to s3://<bucket>/curated/.

Runs as a SageMaker Processing Job.
"""

LABELS = {"benign": 0, "prompt_injection": 1, "jailbreak": 2}


def main() -> None:
    raise NotImplementedError("TODO: implement label mapping, dedupe, split -> S3")


if __name__ == "__main__":
    main()
