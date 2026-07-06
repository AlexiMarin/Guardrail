"""
SageMaker Pipeline definition for the guardrail project.

Flow (see docs/PLAN.md -> Arquitectura MLOps):
  prep data -> train (DistilBERT | DeBERTa | Qwen, in parallel) -> evaluate
  -> register in Model Registry -> conditional deploy (only if quality gate passes)

Triggered by the ml.yml CI workflow (workflow_dispatch or code/data change).
"""


def build_pipeline():
    raise NotImplementedError("TODO: define ProcessingStep, TrainingSteps, ConditionStep, RegisterModel")


if __name__ == "__main__":
    build_pipeline()
