output "classify_url" {
  value       = module.classifier_api.classify_url
  description = "Public endpoint. POST {\"prompt\": \"...\"} here."
}

output "ecr_repository_url" {
  value       = module.classifier_api.ecr_repository_url
  description = "Push the Lambda image here (deploy.sh does this)."
}
