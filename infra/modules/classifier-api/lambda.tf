resource "aws_lambda_function" "classifier" {
  function_name = var.function_name
  role          = aws_iam_role.classifier.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  architectures = ["x86_64"] # matches the CPU the model was quantized/benchmarked on

  memory_size = var.lambda_memory_mb
  timeout     = var.lambda_timeout_s

  # the cost lock: never more than this many parallel executions, no matter the traffic
  reserved_concurrent_executions = var.reserved_concurrency

  environment {
    variables = {
      TURNSTILE_SECRET = var.turnstile_secret
    }
  }

  depends_on = [aws_cloudwatch_log_group.classifier]
}
