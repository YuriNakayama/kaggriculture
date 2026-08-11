output "app_id" {
  description = "Amplify app id."
  value       = aws_amplify_app.this.id
}

output "default_domain" {
  description = "Amplify default domain (…amplifyapp.com)."
  value       = aws_amplify_app.this.default_domain
}

output "app_url" {
  description = "Custom-domain URL of the playable app."
  value       = "https://${var.subdomain_prefix}.${var.domain_name}"
}
