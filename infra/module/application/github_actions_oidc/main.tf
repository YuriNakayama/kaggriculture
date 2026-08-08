# GitHub Actions OIDC role for `${var.github_repo}`.
#
# .github/workflows/scrape-kaggle.yml assumes this role via
# aws-actions/configure-aws-credentials@v4 to read and write the DVC remote.
# The trust policy is scoped to this repository, so no long-lived AWS keys are
# needed — which matters because the repository is public.
#
# The OIDC provider is shared across the AWS account and is referenced as a
# `data` resource rather than created here.

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${var.prefix}-github-actions"
  description        = "Assumed by GitHub Actions in ${var.github_repo} via OIDC."
  assume_role_policy = data.aws_iam_policy_document.trust.json
}

# Read access to the DVC remote. `dev/scrape` runs `dvc pull` first so it can
# diff against what has already been fetched.
data "aws_iam_policy_document" "dvc_read" {
  statement {
    sid       = "ListDvcBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [var.dvc_bucket_arn]
  }

  statement {
    sid       = "GetDvcObjects"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${var.dvc_bucket_arn}/remote/*"]
  }
}

resource "aws_iam_role_policy" "dvc_read" {
  name   = "dvc-read"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.dvc_read.json
}

# Write access for the Kaggle episode scraper workflow (scrape-kaggle.yml).
# `dev/scrape` runs `dvc push`, so the OIDC role needs PutObject on the
# append-only `remote/*` DVC area, plus PutObject on a separate `scrape_logs/*`
# prefix where the scraper persists its run log (so a failed run's log survives
# even when the GitHub Actions job is cancelled / times out).
data "aws_iam_policy_document" "scrape_write" {
  statement {
    sid       = "PutDvcObjects"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["${var.dvc_bucket_arn}/remote/*"]
  }

  statement {
    sid    = "PutGetScrapeLogs"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
    ]
    resources = ["${var.dvc_bucket_arn}/scrape_logs/*"]
  }
}

resource "aws_iam_role_policy" "scrape_write" {
  name   = "scrape-write"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.scrape_write.json
}
