# Lambda container image for the guardrail classifier. The quantized ONNX model + tokenizer are baked
# in, so the model loads from local disk at cold start (no S3 round-trip per container).
FROM public.ecr.aws/lambda/python:3.12

COPY lambda-requirements.txt ./
RUN pip install --no-cache-dir -r lambda-requirements.txt

COPY handler.py ./
# model/ holds model_int8.onnx + tokenizer.json, pulled from S3 by deploy.sh before the build
COPY model/ ./model/

CMD ["handler.handler"]
