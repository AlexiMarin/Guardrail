resource "aws_cloudwatch_log_group" "classifier" {
  name              = "/aws/lambda/${var.function_name}"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "killswitch" {
  name              = "/aws/lambda/${var.function_name}-killswitch"
  retention_in_days = 14
}

# alert if the function starts erroring (e.g. bad deploy)
resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${var.function_name}-errors"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = var.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  alarm_actions       = [aws_sns_topic.budget.arn]
}
