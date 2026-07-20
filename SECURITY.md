# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
privately. **Do not disclose it publicly until we have had a chance to address it.**

To report a security issue, please contact the project maintainers via email
or open a draft security advisory on GitHub.

We will acknowledge receipt within 48 hours and provide an estimated timeline
for a fix. We appreciate your responsible disclosure.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |
| older   | :x:                |

## Best Practices

- All model inference runs on edge devices (ARM) without external network access
- API keys and secrets are configured via environment variables, never hardcoded
- Docker containers run with least-privilege user accounts
- Model files (.pt, .gguf) are verified via SHA-256 checksums before loading
