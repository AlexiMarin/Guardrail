# Guardrail deploy commands -- wraps Docker + Terraform for the classifier Lambda.
# GitHub Actions (later) reuses these same targets, so this is the single source of commands.
#
# One-time setup:
#   cp infra/deploy.env.example      infra/deploy.env        # AWS_PROFILE, MODEL_S3 (gitignored)
#   cp infra/backend.hcl.example     infra/backend.hcl       # state bucket (gitignored)
#   cp infra/terraform.tfvars.example infra/terraform.tfvars # your values (gitignored)
# Then:
#   make deploy

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

-include infra/deploy.env   # optional, gitignored: AWS_PROFILE, MODEL_S3, AWS_REGION

AWS_REGION ?= us-east-2
export AWS_PROFILE AWS_REGION

INFRA := infra
INFERENCE := inference
TF := terraform -chdir=$(INFRA)

.DEFAULT_GOAL := help
.ONESHELL:
.PHONY: help model init ecr image apply deploy plan destroy url clean

help:
	@echo "Targets:"
	@echo "  model    - download the quantized model from S3 into inference/model/"
	@echo "  deploy   - full deploy: model -> build image -> push -> terraform apply -> print URL"
	@echo "  plan     - terraform plan"
	@echo "  destroy  - tear everything down"
	@echo "  url      - print the public endpoint URL"
	@echo "  clean    - remove local build artifacts"

model:
	test -n "$(MODEL_S3)" || { echo "set MODEL_S3 (s3:// path to the quantized model dir)"; exit 1; }
	mkdir -p $(INFERENCE)/model
	aws s3 cp "$(MODEL_S3)/model_int8.onnx" $(INFERENCE)/model/ --region $(AWS_REGION)
	aws s3 cp "$(MODEL_S3)/tokenizer.json"  $(INFERENCE)/model/ --region $(AWS_REGION)

init:
	$(TF) init -backend-config=backend.hcl

# ECR has to exist before we can push the image the Lambda references
ecr: init
	$(TF) apply -target=module.classifier_api.aws_ecr_repository.classifier -auto-approve

# chained with && so ECR_URL survives in one shell (macOS make 3.81 ignores .ONESHELL).
# --provenance=false: buildx otherwise pushes an OCI image index that Lambda rejects.
image: ecr
	ECR_URL=$$($(TF) output -raw ecr_repository_url) && \
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $${ECR_URL%%/*} && \
	docker buildx build --platform linux/amd64 --provenance=false --push -t $$ECR_URL:latest $(INFERENCE)

apply:
	ECR_URL=$$($(TF) output -raw ecr_repository_url) && \
	$(TF) apply -auto-approve -var "image_uri=$$ECR_URL:latest"

deploy: model image apply url

plan:
	$(TF) plan

destroy:
	$(TF) destroy

url:
	@$(TF) output -raw classify_url; echo

clean:
	rm -rf $(INFERENCE)/model $(INFRA)/.build $(INFERENCE)/__pycache__
