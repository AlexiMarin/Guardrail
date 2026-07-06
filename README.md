# Guardrail

A specialized **jailbreak / prompt-injection classifier** that screens prompts before they reach an LLM —
faster and cheaper than an LLM-as-judge guardrail, deployed as a full MLOps pipeline on AWS SageMaker.

> **Design principle:** each guardrail dimension needs a classifier trained on the *right* signal. This
> project targets the **jailbreak / prompt-injection** dimension — one piece of the specialized-classifier
> layer that sits in front of an LLM in production.

## What this project demonstrates

- **Applied ML:** fine-tuning and comparing 3 model architectures for a multiclass safety task.
- **MLOps / DevOps:** an end-to-end SageMaker Pipeline (data prep → train → evaluate → registry →
  conditional deploy) with a quality gate, Terraform IaC, and CI/CD.
- **Secure data flow:** a cloud-only pipeline (SageMaker Studio + ephemeral Processing Jobs, no local data)
  designed to also work with sensitive / regulated data.

## Approach (technical report)

- **Hypothesis** — specialized **classifier** models work better than general-purpose **LLMs** as
  guardrails for jailbreak / prompt-injection detection. "Better" is measured across recall,
  false-positive rate, latency, cost, and generalization to unseen attacks — the experiment tests
  whether this holds.
- **Method** — 3 candidate classifiers (DistilBERT, DeBERTa-v3-base, Qwen2.5-1.5B) fine-tuned on 9 public
  jailbreak / prompt-injection benchmarks; multiclass labels (`benign` / `prompt_injection` / `jailbreak`);
  held-out set of *unseen* attacks to measure generalization. Compared against a **baseline: a
  general-purpose LLM prompted as a guardrail** (the "LLM guardrail" the hypothesis is tested against).
- **Results** — _TODO: metrics table (recall per attack class, false-positive rate, latency, cost)._
- **Conclusion** — _TODO: deployed model + trade-off + honest limitations._

## Repo layout

```
data/         # download + build dataset (run as SageMaker Processing Jobs; output to S3)
training/     # per-model training code (distilbert / deberta / qwen) + shared code
evaluation/   # metrics per class, unseen-attacks eval, quality gate
pipelines/    # SageMaker Pipeline definition
infra/        # Terraform (SageMaker, endpoint, API Gateway + Lambda, IAM, monitoring)
inference/    # Lambda handler for the public demo
.github/      # CI/CD workflows (infra + ML)
```