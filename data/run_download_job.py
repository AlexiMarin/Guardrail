"""
Launch data/download.py as a SageMaker Processing Job from your laptop.

The job cleans up after itself -- SageMaker kills the instance as soon as it finishes. Two safety
nets in case it hangs: --max-runtime-seconds force-kills it, and Ctrl+C here stops the remote job
too (otherwise a local cancel leaves it running and billing).

Gated sources (HarmBench, HackAPrompt, WildGuardMix -- see data/schema.md) need an HF account that
already accepted their terms, plus HF_TOKEN in your shell (forwarded to the container, never saved).

Usage:
    export HF_TOKEN=hf_...
    python3 data/run_download_job.py \
        --role-arn arn:aws:iam::<account>:role/service-role/<SageMakerExecutionRole> \
        --profile <your-aws-cli-profile>
"""

import argparse
import os

import boto3
import sagemaker
from sagemaker.processing import ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--profile", default=None, help="Local AWS CLI profile to use.")
    parser.add_argument("--bucket", default=None, help="Defaults to the SageMaker session's default bucket.")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--max-runtime-seconds", type=int, default=3600, help="Hard ceiling; see module docstring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boto_session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)
    bucket = args.bucket or sm_session.default_bucket()

    env = {}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]

    # Plain CPU container -- this job only downloads and converts data, no GPU needed. download.py
    # installs its own deps at runtime because this processor only takes a single script, no requirements.txt.
    processor = SKLearnProcessor(
        framework_version="1.4-2",
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        max_runtime_in_seconds=args.max_runtime_seconds,
        base_job_name="guardrail-download",
        sagemaker_session=sm_session,
        env=env,
    )

    destination = f"s3://{bucket}/guardrail/raw"
    print(f"Launching processing job -> writes to {destination}")
    print(f"Hard runtime ceiling: {args.max_runtime_seconds}s")

    try:
        processor.run(
            code="data/download.py",
            outputs=[ProcessingOutput(output_name="raw", source="/opt/ml/processing/output", destination=destination)],
            arguments=["--output-dir", "/opt/ml/processing/output"],
            wait=True,
            logs=True,
        )
    except KeyboardInterrupt:
        job_name = processor.jobs[-1].job_name
        print(f"\nInterrupted locally -- stopping remote job {job_name} to avoid runaway billing...")
        sm_session.sagemaker_client.stop_processing_job(ProcessingJobName=job_name)
        raise


if __name__ == "__main__":
    main()
