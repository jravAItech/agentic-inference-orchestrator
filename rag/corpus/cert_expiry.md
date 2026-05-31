# Runbook: TLS Certificate Expiry

Fault code: TLS-EXP-495
Symptom: Clients fail handshake; logs show certificate expired.
Root cause: cert not rotated before expiry; missing auto-renewal.
Remediation:
1. Confirm expiry: `openssl x509 -enddate -noout -in cert.pem`.
2. Issue/rotate cert via cert-manager or ACME.
3. Reload ingress; verify chain.
4. Enable auto-renewal to prevent recurrence.
