"""
Launch inference/quantize.py as a SageMaker Processing Job (same setup as the data jobs: runtime
ceiling + Ctrl+C handler).

Runs on a CPU instance -- quantization and CPU-latency measurement don't need a GPU, and x86 CPU
matches Lambda so the latency number is representative. Reads the trained DistilBERT + test set from
S3, writes the quantized model + tokenizer + metrics to s3://<bucket>/guardrail/models/distilbert-onnx-int8/.

Usage:
    python3 inference/run_quantize_job.py \
        --role-arn arn:aws:iam::<account>:role/service-role/<SageMakerExecutionRole> \
        --profile <your-aws-cli-profile>
"""

import argparse

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor

BUCKET = "sagemaker-us-east-2-306767070740"
# the trained DistilBERT (see docs/models.md)
MODEL_S3 = f"s3://{BUCKET}/guardrail/models/distilbert/guardrail-train-distilbert-2026-07-06-21-26-59-335/output/model.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--profile", default=None, help="Local AWS CLI profile to use.")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--max-runtime-seconds", type=int, default=3600, help="Hard ceiling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boto_session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    sm_session = sagemaker.Session(boto_session=boto_session)

    processor = PyTorchProcessor(
        framework_version="2.8.0",
        py_version="py312",
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        max_runtime_in_seconds=args.max_runtime_seconds,
        base_job_name="guardrail-quantize",
        sagemaker_session=sm_session,
    )

    destination = f"s3://{BUCKET}/guardrail/models/distilbert-onnx-int8"
    print(f"Quantizing DistilBERT on {args.instance_type} -> writes to {destination}")

    try:
        processor.run(
            code="quantize.py",
            source_dir="inference",  # requirements.txt here (onnx/onnxruntime) gets pip-installed
            inputs=[
                ProcessingInput(source=MODEL_S3, destination="/opt/ml/processing/model"),
                ProcessingInput(source=f"s3://{BUCKET}/guardrail/curated", destination="/opt/ml/processing/input"),
            ],
            outputs=[ProcessingOutput(output_name="quantized", source="/opt/ml/processing/output", destination=destination)],
            arguments=["--model-dir", "/opt/ml/processing/model", "--data-dir", "/opt/ml/processing/input",
                       "--output-dir", "/opt/ml/processing/output"],
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
