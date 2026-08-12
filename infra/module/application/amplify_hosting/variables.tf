variable "prefix" {
  description = "Prefix applied to resource names."
  type        = string
}

variable "github_repo" {
  description = "owner/repo of the repository Amplify builds from."
  type        = string
}

variable "github_access_token" {
  description = <<-EOT
    GitHub personal access token Amplify uses to register its webhook and
    read the repository (classic PAT with `repo` scope, or fine-grained with
    read + webhook admin on this repo). Stored in Amplify, not in state
    outputs. Lives in terraform.tfvars — never commit it.
  EOT
  type        = string
  sensitive   = true
}

variable "branch_name" {
  description = "Branch Amplify auto-builds and serves."
  type        = string
  default     = "main"
}

variable "domain_name" {
  description = "Apex domain with an existing Route53 hosted zone (e.g. avifauna.click)."
  type        = string
}

variable "subdomain_prefix" {
  description = "Subdomain serving the app (e.g. goose -> goose.<domain_name>)."
  type        = string
  default     = "goose"
}
