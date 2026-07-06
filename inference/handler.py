"""
Lambda handler for the public demo.

Sits behind API Gateway and in front of the SageMaker endpoint (the endpoint is never
exposed directly). Applies throttling / API key at the gateway.

Request:  {"prompt": "..."}
Response: {"label": "benign|prompt_injection|jailbreak", "scores": {...}}
"""


def handler(event, context):
    raise NotImplementedError("TODO: parse prompt, invoke SageMaker endpoint, return label + scores")
