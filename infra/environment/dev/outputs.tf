output "dvc_bucket_name" {
  description = "Use as: dev/dvc remote modify s3 url s3://<bucket>/remote"
  value       = module.dvc_remote.bucket_name
}

output "dvc_iam_user_name" {
  description = "IAM user dedicated to DVC remote access (for local dev)."
  value       = module.dvc_remote.iam_user_name
}

output "dvc_iam_access_key_id" {
  description = "Access key ID. Add to ~/.aws/credentials under profile `kaggriculture`."
  value       = module.dvc_remote.iam_access_key_id
  sensitive   = true
}

output "dvc_iam_secret_access_key" {
  description = "Secret access key. Add to ~/.aws/credentials under profile `kaggriculture`."
  value       = module.dvc_remote.iam_secret_access_key
  sensitive   = true
}

output "github_actions_role_arn" {
  description = "Set this as the repository secret AWS_ROLE_ARN (used by scrape-kaggle.yml)."
  value       = module.github_actions_oidc.role_arn
}
