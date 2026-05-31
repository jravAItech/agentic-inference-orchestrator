# Runbook: Database Connection Pool Exhaustion

Fault code: DB-POOL-503
Symptom: API returns 503, logs show "connection pool exhausted".
Root cause: long-running queries holding connections; pool max too low.
Remediation:
1. Check active connections: `SELECT count(*) FROM pg_stat_activity;`
2. Identify long queries > 30s and terminate if safe.
3. Raise pool max in config (max_connections) and redeploy.
4. Add a statement_timeout to prevent recurrence.
