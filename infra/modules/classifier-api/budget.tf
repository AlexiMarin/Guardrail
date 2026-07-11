# Cost alerting + automatic shutoff. The structural cap is the classifier's reserved concurrency
# (lambda.tf); this adds a monthly budget that emails you and, at 100%, disables the endpoint.

resource "aws_sns_topic" "budget" {
  name = "${var.function_name}-budget"
}

# let AWS Budgets and CloudWatch alarms publish to the topic (without this, nothing fires)
data "aws_iam_policy_document" "sns" {
  statement {
    actions   = ["SNS:Publish"]
    resources = [aws_sns_topic.budget.arn]
    principals {
      type        = "Service"
      identifiers = ["budgets.amazonaws.com", "cloudwatch.amazonaws.com"]
    }
  }
}

resource "aws_sns_topic_policy" "budget" {
  arn    = aws_sns_topic.budget.arn
  policy = data.aws_iam_policy_document.sns.json
}

# optional email alert (only if an address is provided)
resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.budget.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_budgets_budget" "monthly" {
  name         = "${var.function_name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # warn at 80% of forecast, fire the kill switch at 100% of actual spend
  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "FORECASTED"
    subscriber_sns_topic_arns = [aws_sns_topic.budget.arn]
  }

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_sns_topic_arns = [aws_sns_topic.budget.arn]
  }
}

# --- kill-switch Lambda, triggered by the budget SNS topic ---
data "archive_file" "killswitch" {
  type        = "zip"
  source_file = "${path.module}/kill_switch.py"
  output_path = "${path.module}/.build/kill_switch.zip"
}

resource "aws_lambda_function" "killswitch" {
  function_name    = "${var.function_name}-killswitch"
  role             = aws_iam_role.killswitch.arn
  runtime          = "python3.12"
  handler          = "kill_switch.handler"
  filename         = data.archive_file.killswitch.output_path
  source_code_hash = data.archive_file.killswitch.output_base64sha256
  timeout          = 30

  environment {
    variables = {
      TARGET_FUNCTION = var.function_name
    }
  }

  depends_on = [aws_cloudwatch_log_group.killswitch]
}

resource "aws_sns_topic_subscription" "killswitch" {
  topic_arn = aws_sns_topic.budget.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.killswitch.arn
}

resource "aws_lambda_permission" "sns_killswitch" {
  statement_id  = "AllowSNSInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.killswitch.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.budget.arn
}
