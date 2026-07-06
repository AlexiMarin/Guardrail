"""
Download the guardrail datasets from the Hugging Face Hub and write them to S3.

Runs as an ephemeral SageMaker Processing Job — data never touches a local machine.
See docs/PLAN.md ("Flujo de datos 100% en la nube") for the rationale.

TODO:
  - Confirm exact HF repo ids + licenses for each source.
  - Handle gated datasets (HF token via env var / Secrets Manager).
  - Write raw copies to s3://<bucket>/raw/<source>/  (SSE-KMS in a sensitive-data setup).
"""

# key -> Hugging Face repo id. TODO: fill in verified ids + configs.
SOURCES: dict[str, str] = {
    # Attacks
    "jailbreakbench": "",
    "harmbench": "",
    "hackaprompt": "",
    "gandalf": "",
    # Prompt injection
    "deepset_prompt_injections": "",
    # Balanced guard-training mix
    "wildguardmix": "",
    # Hard / confusing benign (false-positive control)
    "xstest": "",
    "orbench": "",
    # Contrast: ordinary toxicity (not jailbreak)
    "toxicchat": "",
}


def main() -> None:
    raise NotImplementedError("TODO: download each source and upload raw to S3")


if __name__ == "__main__":
    main()
