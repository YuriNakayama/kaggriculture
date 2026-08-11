# Amplify Hosting for the playable frontend (static SPA, no backend).
#
# The build itself is defined by amplify.yml at the repository root — it
# collects the Python agent cases, builds the visualizers with pnpm, and
# publishes frontend/playable/dist. Costs are static-hosting only.

resource "aws_amplify_app" "this" {
  name         = "${var.prefix}-playable"
  repository   = "https://github.com/${var.github_repo}"
  access_token = var.github_access_token
  platform     = "WEB"

  enable_branch_auto_build = true

  # SPA rewrite: anything without a file extension falls through to
  # index.html. Extensioned assets (js/wasm/py/json/replay page…) are served
  # as-is — .py must stay listed or Pyodide agent files would 404 into HTML.
  custom_rule {
    source = "</^[^.]+$|\\.(?!(css|gif|ico|jpg|jpeg|js|mjs|png|txt|svg|webp|woff|woff2|ttf|map|json|wasm|py|html)$)([^.]+$)/>"
    status = "200"
    target = "/index.html"
  }
}

resource "aws_amplify_branch" "main" {
  app_id            = aws_amplify_app.this.id
  branch_name       = var.branch_name
  enable_auto_build = true
  stage             = "PRODUCTION"
}

resource "aws_amplify_domain_association" "this" {
  app_id      = aws_amplify_app.this.id
  domain_name = var.domain_name

  # Terraform creates the validation/CNAME records below; don't block apply
  # on certificate issuance.
  wait_for_verification = false

  sub_domain {
    branch_name = aws_amplify_branch.main.branch_name
    prefix      = var.subdomain_prefix
  }
}

# No explicit Route53 records: the hosted zone lives in the same account, so
# Amplify creates and manages the TLS-validation and subdomain CNAMEs itself
# when the domain association is created.
