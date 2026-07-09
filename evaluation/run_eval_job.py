"""
Launch a SageMaker job that runs evaluation/evaluate.py on one trained model.

Runs as a *training* job, not a processing job -- this account's processing quota for g5 is 0 but
its training quota is 1 (they're tracked separately), and a training job runs any code fine.

All 3 models run on the same instance type so the latency numbers are comparable.

Usage:
    python3 evaluation/run_eval_job.py --model deberta \
        --role-arn arn:aws:iam::<account>:role/service-role/<SageMakerExecutionRole> --profile <profile>
    python3 evaluation/run_eval_job.py --model all --role-arn ... --profile ...
"""

import argparse

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

BUCKET = "sagemaker-us-east-2-306767070740"

# the model.tar.gz from each training run. Batch is smaller for the bigger models to fit in 24GB.
MODELS = {
    "distilbert": {
        "s3": f"s3://{BUCKET}/guardrail/models/distilbert/guardrail-train-distilbert-2026-07-06-21-26-59-335/output/model.tar.gz",
        "batch": 128,
    },
    "deberta": {
        "s3": f"s3://{BUCKET}/guardrail/models/deberta/guardrail-train-deberta-2026-07-06-22-26-04-081/output/model.tar.gz",
        "batch": 64,
    },
    "qwen": {
        "s3": f"s3://{BUCKET}/guardrail/models/qwen/guardrail-train-qwen-2026-07-07-17-09-42-852/output/model.tar.gz",
        "batch": 32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[*sorted(MODELS), "all"])
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--instance-type", default="ml.g5.xlarge")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    return parser.parse_args()


def run_one(model_name: str, args, sm_session) -> None:
    cfg = MODELS[model_name]
    estimator = HuggingFace(
        entry_point="evaluate.py",
        source_dir="evaluation",
        transformers_version="4.56.2",
        pytorch_version="2.8.0",
        py_version="py312",
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        max_run=args.max_runtime_seconds,
        base_job_name=f"guardrail-eval-{model_name}",
        output_path=f"s3://{BUCKET}/guardrail/eval/{model_name}",
        sagemaker_session=sm_session,
        hyperparameters={
            # evaluate.py gets these as --key value args
            "model-name": model_name,
            "model-dir": "/opt/ml/input/data/model",
            "data-dir": "/opt/ml/input/data/data",
            "output-dir": "/opt/ml/model",  # ends up in the job's output tarball
            "batch-size": cfg["batch"],
        },
    )
    print(f"Evaluating {model_name} on {args.instance_type} (batch {cfg['batch']})")
    # channels land at /opt/ml/input/data/<channel>/; evaluate.py unpacks the model tarball itself
    estimator.fit(
        {"model": cfg["s3"], "data": f"s3://{BUCKET}/guardrail/curated"},
        wait=True,
        logs=True,
    )


def main() -> None:
    args = parse_args()
    boto_session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)

    targets = sorted(MODELS) if args.model == "all" else [args.model]
    for name in targets:
        run_one(name, args, sm_session)


if __name__ == "__main__":
    main()
