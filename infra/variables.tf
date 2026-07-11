variable "aws_region" {
  type    = string
  default = "us-east-2"
}

variable "project" {
  type    = string
  default = "guardrail"
}

variable "environment" {
  type    = string
  default = "demo"
}

variable "function_name" {
  type    = string
  default = "guardrail-classifier"
}

variable "image_uri" {
  type        = string
  default     = ""
  description = "ECR image URI for the Lambda. Set by deploy.sh after the image is pushed."
}

variable "allowed_origins" {
  type        = list(string)
  default     = ["*"]
  description = "CORS allow-list. Set to your portfolio domain(s) in terraform.tfvars."
}

variable "turnstile_secret" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Cloudflare Turnstile secret key. If empty, Turnstile checks are skipped."
}

variable "budget_limit_usd" {
  type    = number
  default = 30
}

variable "alert_email" {
  type    = string
  default = ""
}
