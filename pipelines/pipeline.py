"""
SageMaker Pipeline for the guardrail project.

Flow: prep data -> train the 3 models -> evaluate -> register -> deploy only if the gate passes.
Triggered by the ml.yml CI workflow.
"""


def build_pipeline():
    raise NotImplementedError("TODO: define ProcessingStep, TrainingSteps, ConditionStep, RegisterModel")


if __name__ == "__main__":
    build_pipeline()
