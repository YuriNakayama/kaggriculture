variable "aws_region" {
  description = "AWS region for dev resources."
  type        = string
  default     = "ap-northeast-1"
}

variable "dvc_bucket_name" {
  description = "Globally-unique S3 bucket name for the DVC remote (e.g. kaggriculture-dvc-<account_id>)."
  type        = string
}

variable "resource_prefix" {
  description = "Prefix applied to IAM resources."
  type        = string
  default     = "kaggriculture-dev"
}

variable "github_repo" {
  description = "owner/repo allowed to assume the GitHub Actions OIDC role."
  type        = string
  default     = "YuriNakayama/kaggriculture"
}
