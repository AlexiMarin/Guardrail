output "classify_url" {
  value       = "${aws_apigatewayv2_api.http.api_endpoint}/classify"
  description = "Public endpoint. POST {\"prompt\": \"...\"} here."
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.classifier.repository_url
  description = "Push the Lambda image here (deploy.sh does this)."
}
