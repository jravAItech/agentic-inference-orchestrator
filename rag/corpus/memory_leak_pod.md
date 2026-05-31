# Runbook: Pod OOMKilled / Memory Leak

Fault code: K8S-OOM-137
Symptom: Pod restarts with exit code 137, rising memory before crash.
Root cause: unbounded cache or leaked references in the service.
Remediation:
1. Inspect `kubectl describe pod` for OOMKilled events.
2. Pull heap profile from the service's /debug/pprof endpoint.
3. Set memory limits/requests and a cache eviction policy.
4. Roll out fix; monitor RSS in Grafana for 24h.
