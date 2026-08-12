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

variable "github_owner_id" {
  description = "Numeric GitHub owner id. `gh api users/<owner> --jq .id`"
  type        = string
}

variable "github_repo_id" {
  description = "Numeric GitHub repository id. `gh api repos/<owner>/<repo> --jq .id`"
  type        = string
}

variable "amplify_github_access_token" {
  description = "GitHub PAT for Amplify (repo read + webhook). tfvars only — never commit."
  type        = string
  sensitive   = true
}

variable "playable_domain" {
  description = "Apex domain with an existing Route53 hosted zone for the playable app."
  type        = string
  default     = "avifauna.click"
}

variable "playable_subdomain" {
  description = "Subdomain prefix for the playable app."
  type        = string
  default     = "goose"
}
