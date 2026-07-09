"""
Launch data/build_dataset.py as a SageMaker Processing Job (same setup as run_download_job.py:
runtime ceiling + Ctrl+C handler as safety nets).

Reads the raw sources from the download job (s3://<bucket>/guardrail/raw/) and writes the labeled,
deduped, split dataset to s3://<bucket>/guardrail/curated/.

Usage:
    python3 data/run_build_dataset_job.py \
        --role-arn arn:aws:iam::<account>:role/service-role/<SageMakerExecutionRole> \
        --profile <your-aws-cli-profile>
"""

import argparse
import os

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
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

    # Same CPU container as the download job -- just pandas/sklearn work, no GPU.
    processor = SKLearnProcessor(
        framework_version="1.4-2",
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        max_runtime_in_seconds=args.max_runtime_seconds,
        base_job_name="guardrail-build-dataset",
        sagemaker_session=sm_session,
        env=env,
    )

    raw_source = f"s3://{bucket}/guardrail/raw"
    curated_destination = f"s3://{bucket}/guardrail/curated"
    print(f"Reading raw sources from {raw_source}")
    print(f"Writing curated splits to {curated_destination}")
    print(f"Hard runtime ceiling: {args.max_runtime_seconds}s")

    try:
        processor.run(
            code="data/build_dataset.py",
            inputs=[ProcessingInput(source=raw_source, destination="/opt/ml/processing/input")],
            outputs=[
                ProcessingOutput(
                    output_name="curated", source="/opt/ml/processing/output", destination=curated_destination
                )
            ],
            arguments=["--input-dir", "/opt/ml/processing/input", "--output-dir", "/opt/ml/processing/output"],
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
