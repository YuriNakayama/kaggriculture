variable "prefix" {
  description = "Resource name prefix applied to the IAM role (e.g. kaggriculture-dev)."
  type        = string
}

variable "github_repo" {
  description = "GitHub repository allowed to assume this role, in `owner/repo` form."
  type        = string
  default     = "YuriNakayama/kaggriculture"
}



variable "dvc_bucket_arn" {
  description = "DVC remote S3 bucket ARN. The role gets read access (dvc pull) and scoped write access (dvc push, scrape logs)."
  type        = string
}
