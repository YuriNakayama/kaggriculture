provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Kaggriculture"
      Environment = "dev"
      ManagedBy   = "Terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

# S3 bucket backing the DVC remote: versioned, SSE-encrypted, public access
# fully blocked. This is where replays and datasets live.
module "dvc_remote" {
  source      = "../../module/application/dvc_remote"
  bucket_name = var.dvc_bucket_name
  prefix      = var.resource_prefix
}

# OIDC role assumed by .github/workflows/scrape-kaggle.yml so the scheduled
# scrape can write to the DVC bucket without any long-lived AWS credentials
# stored in this (public) repository.
module "github_actions_oidc" {
  source      = "../../module/application/github_actions_oidc"
  prefix      = var.resource_prefix
  github_repo = var.github_repo

  # Required by the immutable-subject claim GitHub now issues; see the trust
  # policy comment in the module.
  github_owner_id = var.github_owner_id
  github_repo_id  = var.github_repo_id

  # Built from the variable rather than module.dvc_remote so this module can be
  # applied with -target without dragging the bucket resources along.
  dvc_bucket_arn = "arn:aws:s3:::${var.dvc_bucket_name}"
}

# Amplify Hosting for the playable frontend (frontend/ + amplify.yml), served
# at https://<playable_subdomain>.<playable_domain>. Static hosting only.
module "amplify_hosting" {
  source              = "../../module/application/amplify_hosting"
  prefix              = var.resource_prefix
  github_repo         = var.github_repo
  github_access_token = var.amplify_github_access_token
  branch_name         = "main"
  domain_name         = var.playable_domain
  subdomain_prefix    = var.playable_subdomain
}
