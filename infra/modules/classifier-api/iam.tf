data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# --- classifier Lambda role: just needs to write logs (inference is pure local compute) ---
resource "aws_iam_role" "classifier" {
  name               = "${var.function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "classifier_logs" {
  role       = aws_iam_role.classifier.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- kill-switch Lambda role: write logs + set the classifier's concurrency to 0 ---
resource "aws_iam_role" "killswitch" {
  name               = "${var.function_name}-killswitch-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "killswitch_logs" {
  role       = aws_iam_role.killswitch.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "killswitch" {
  statement {
    actions   = ["lambda:PutFunctionConcurrency"]
    resources = [aws_lambda_function.classifier.arn]
  }
}

resource "aws_iam_role_policy" "killswitch" {
  name   = "${var.function_name}-killswitch"
  role   = aws_iam_role.killswitch.id
  policy = data.aws_iam_policy_document.killswitch.json
}
