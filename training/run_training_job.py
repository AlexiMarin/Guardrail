"""
Launch a SageMaker Training Job for one of the 3 candidate models. Shared across all 3 so the
instance, data channels, and safety nets stay the same (same Ctrl+C-stops-remote-job as the data
jobs).

All 3 train on GPU via the HuggingFace DLC on ml.g5.xlarge. We wanted g4dn but this account's quota
for it was 0 and the increase got denied (new account) -- g5/g6 already had a quota of 1, so we use
those. There's also a CPU fallback (`pytorch_cpu`) in case GPU quota disappears again.

Usage:
    python3 training/run_training_job.py --model distilbert \
        --role-arn arn:aws:iam::<account>:role/service-role/<SageMakerExecutionRole> \
        --profile <your-aws-cli-profile>
"""

import argparse

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.pytorch import PyTorch

# entry_point is relative to source_dir="training/"
MODEL_CONFIGS = {
    "distilbert": {
        "estimator": "huggingface_gpu",
        "entry_point": "distilbert/train.py",
        "instance_type": "ml.g5.xlarge",
    },
    "deberta": {
        "estimator": "huggingface_gpu",
        "entry_point": "deberta/train.py",
        "instance_type": "ml.g5.xlarge",
    },
    "qwen": {
        "estimator": "huggingface_gpu",
        "entry_point": "qwen/train.py",
        "instance_type": "ml.g5.xlarge",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--profile", default=None, help="Local AWS CLI profile to use.")
    parser.add_argument("--bucket", default=None, help="Defaults to the SageMaker session's default bucket.")
    parser.add_argument("--instance-type", default=None, help="Overrides the model's default instance type.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Subsample for a fast smoke test, not the real run."
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None, help="Subsample for a fast smoke test, not the real run."
    )
    parser.add_argument("--max-runtime-seconds", type=int, default=3600, help="Hard ceiling; see module docstring.")
    return parser.parse_args()


def build_estimator(args: argparse.Namespace, config: dict, sm_session, output_path: str):
    hyperparameters = dict(config.get("hyperparameters", {}))
    if args.max_train_samples is not None:
        hyperparameters["max-train-samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        hyperparameters["max-eval-samples"] = args.max_eval_samples
    if args.epochs is not None:
        hyperparameters["epochs"] = args.epochs
    if args.train_batch_size is not None:
        hyperparameters["train-batch-size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        hyperparameters["eval-batch-size"] = args.eval_batch_size

    common_kwargs = dict(
        entry_point=config["entry_point"],
        source_dir="training",
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type or config["instance_type"],
        hyperparameters=hyperparameters,
        output_path=output_path,
        max_run=args.max_runtime_seconds,
        base_job_name=f"guardrail-train-{args.model}",
        sagemaker_session=sm_session,
    )

    if config["estimator"] == "pytorch_cpu":
        return PyTorch(framework_version="2.8.0", py_version="py312", **common_kwargs)
    elif config["estimator"] == "huggingface_gpu":
        return HuggingFace(
            transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312", **common_kwargs
        )
    raise ValueError(f"Unknown estimator type: {config['estimator']!r}")


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.model]

    boto_session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    bucket = args.bucket or sm_session.default_bucket()

    curated_s3 = f"s3://{bucket}/guardrail/curated"
    output_path = f"s3://{bucket}/guardrail/models/{args.model}"
    estimator = build_estimator(args, config, sm_session, output_path)

    print(f"Training {args.model} ({config['estimator']}) -- reading {curated_s3}, writing model to {output_path}")
    print(f"Instance: {args.instance_type or config['instance_type']}  Hard runtime ceiling: {args.max_runtime_seconds}s")

    try:
        # both channels point at the same curated/ prefix -- load_split() picks the right
        # {split}.parquet from whichever dir it gets, and it's only ~20MB to download twice
        estimator.fit({"train": curated_s3, "val": curated_s3}, wait=True, logs=True)
    except KeyboardInterrupt:
        job_name = estimator.latest_training_job.job_name
        print(f"\nInterrupted locally -- stopping remote job {job_name} to avoid runaway billing...")
        sm_session.sagemaker_client.stop_training_job(TrainingJobName=job_name)
        raise


if __name__ == "__main__":
    main()
