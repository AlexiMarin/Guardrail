terraform {
  required_version = ">= 1.10" 

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {}
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      project     = var.project
      environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

module "classifier_api" {
  source = "./modules/classifier-api"

  function_name    = var.function_name
  image_uri        = var.image_uri
  allowed_origins  = var.allowed_origins
  turnstile_secret = var.turnstile_secret
  budget_limit_usd = var.budget_limit_usd
  alert_email      = var.alert_email
}
