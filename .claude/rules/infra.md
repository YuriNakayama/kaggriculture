---
paths:
  - "infra/**"
---

# Infrastructure Rules (`infra/**`)

Terraform-managed AWS resources. Kept deliberately small: this repository needs an S3 bucket for the DVC remote and an OIDC role so GitHub Actions can write to it without long-lived credentials.

```
infra/
  environment/dev/          Root module — the only place you run terraform
  module/application/
    dvc_remote/             S3 bucket (versioning + SSE + public-access block)
    github_actions_oidc/    OIDC provider + IAM role for GitHub Actions
```

## Hard rules

> **`terraform apply` and `terraform destroy` are denied in `.claude/settings.json`.**
> Claude runs `fmt`, `validate`, and `plan` only. Applying infrastructure is a human decision.

> **This repository is public.** Never commit `terraform.tfstate` (it contains resource attributes and sometimes secrets), `terraform.tfvars` (account IDs, bucket names), or `.terraform/`. All three are gitignored — verify before every commit.

## Conventions

- **No hardcoded account IDs, ARNs, or bucket names in `.tf` bodies.** Everything parameterised through `variables.tf`; real values live in `terraform.tfvars`, which is gitignored. `terraform.tfvars.example` documents the shape with placeholders.
- Region is `ap-northeast-1`.
- Run terraform from `infra/environment/dev/` — that is the root module.
- After editing, always `terraform fmt` then `terraform validate`.

## OIDC, not access keys

GitHub Actions authenticates by assuming an IAM role via OIDC. The workflow declares:

```yaml
permissions:
  contents: read
  id-token: write        # required for OIDC
```

and uses `aws-actions/configure-aws-credentials@v4` with `role-to-assume: ${{ secrets.AWS_ROLE_ARN }}`. There are **no** `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` secrets in this repo, and none should be added — the OIDC trust policy scopes access to this specific repository.

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `KAGGRICULTURE_DVC_BUCKET` | `gpu/runpod`, `gpu/kaggle` | DVC/S3 bucket name. **Never hardcoded** — the bucket name embeds the AWS account id, and this repository is public. Set it from `terraform output -raw dvc_bucket_name`. |

## Repository secrets required

| Secret | Used by | Purpose |
|---|---|---|
| `AWS_ROLE_ARN` | `scrape-kaggle.yml` | Role to assume for DVC S3 access |
| `KAGGLE_USERNAME` | `scrape-kaggle.yml` | Kaggle API auth |
| `KAGGLE_KEY` | `scrape-kaggle.yml` | Kaggle API auth |

## Workflow-side conventions

Long-running scheduled workflows that write to the DVC remote must serialise:

```yaml
concurrency:
  group: scrape-kaggle
  cancel-in-progress: false   # never cancel mid-write; queue instead
```

`cancel-in-progress: false` is deliberate — cancelling a job partway through a DVC push can leave the remote and the `.dvc` files inconsistent.
