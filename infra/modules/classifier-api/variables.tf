# Inputs for the classifier-api module: a Lambda (container image) running the quantized guardrail
# classifier behind an HTTP API, plus the cost-control pieces (throttling, reserved concurrency,
# budget + kill switch). Reusable -- a second env would just call it again with different values.

variable "function_name" {
  type = string
}

variable "image_uri" {
  type        = string
  default     = ""
  description = "ECR image URI for the Lambda. Set by deploy.sh after the image is pushed."
}

variable "lambda_memory_mb" {
  type    = number
  default = 1536 # more memory = more CPU = lower latency; 1536 is a good spot for this model
}

variable "lambda_timeout_s" {
  type    = number
  default = 10
}

variable "reserved_concurrency" {
  type        = number
  default     = 2
  description = "Hard cap on parallel executions -- bounds max spend even under attack. THE cost lock."
}

variable "throttle_rate_limit" {
  type    = number
  default = 5 # steady-state requests/sec at the API Gateway
}

variable "throttle_burst_limit" {
  type    = number
  default = 10
}

variable "allowed_origins" {
  type        = list(string)
  default     = ["*"]
  description = "CORS allow-list. Set to your portfolio domain(s)."
}

variable "turnstile_secret" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Cloudflare Turnstile secret key. If empty, Turnstile checks are skipped."
}

variable "budget_limit_usd" {
  type        = number
  default     = 30
  description = "Monthly budget. At 100% the kill switch sets Lambda concurrency to 0."
}

variable "alert_email" {
  type        = string
  default     = ""
  description = "Email for budget alerts. If empty, no email subscription is created."
}
