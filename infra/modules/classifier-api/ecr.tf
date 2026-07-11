resource "aws_ecr_repository" "classifier" {
  name         = var.function_name
  force_delete = true # it's a demo; let `terraform destroy` clean up images too

  image_scanning_configuration {
    scan_on_push = true
  }
}
