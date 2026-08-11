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

data "aws_route53_zone" "this" {
  name = var.domain_name
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

# TLS certificate validation record ("<name> CNAME <value>").
locals {
  cert_parts = split(" ", aws_amplify_domain_association.this.certificate_verification_dns_record)
}

resource "aws_route53_record" "cert_verification" {
  zone_id = data.aws_route53_zone.this.zone_id
  name    = local.cert_parts[0]
  type    = local.cert_parts[1]
  records = [local.cert_parts[2]]
  ttl     = 300
}

# Subdomain -> CloudFront distribution records ("<prefix> CNAME <target>").
resource "aws_route53_record" "sub_domain" {
  for_each = {
    for sd in aws_amplify_domain_association.this.sub_domain :
    sd.prefix => split(" ", sd.dns_record)
  }

  zone_id = data.aws_route53_zone.this.zone_id
  name    = "${each.key}.${var.domain_name}"
  type    = each.value[1]
  records = [each.value[2]]
  ttl     = 300
}
