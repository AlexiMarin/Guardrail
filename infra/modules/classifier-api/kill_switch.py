"""
Budget kill switch. Fires when the AWS Budget alert hits its threshold (via SNS) and sets the
classifier's reserved concurrency to 0, which throttles all new invocations (~free). The real hard
cost cap is the classifier's reserved concurrency; this is the extra brake if that spend still adds
up over days. Re-enable by re-running `terraform apply`.
"""

import os

import boto3


def handler(event, context):
    fn = os.environ["TARGET_FUNCTION"]
    boto3.client("lambda").put_function_concurrency(FunctionName=fn, ReservedConcurrentExecutions=0)
    print(f"Budget kill switch fired -- set {fn} reserved concurrency to 0")
