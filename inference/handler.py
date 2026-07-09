"""
Lambda handler for the public demo.

Sits between API Gateway and the SageMaker endpoint so the endpoint isn't exposed directly.

Request:  {"prompt": "..."}
Response: {"label": "benign|prompt_injection|jailbreak", "scores": {...}}
"""


def handler(event, context):
    raise NotImplementedError("TODO: parse prompt, invoke SageMaker endpoint, return label + scores")
