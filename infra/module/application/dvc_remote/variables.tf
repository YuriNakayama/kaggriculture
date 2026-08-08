variable "bucket_name" {
  description = "Name of the S3 bucket used as DVC remote."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix (e.g. kaggriculture-dev) applied to IAM resources."
  type        = string
}
