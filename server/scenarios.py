"""
Incident scenario definitions for the Incident Response Environment.

Each scenario defines:
- Alert details and severity
- Service topology (which services exist and their relationships)
- Ground truth root cause and valid remedies
- Clue map: which queries reveal which relevant information
- Log/metric/runbook data templates per service
"""

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Set


@dataclass
class ServiceInfo:
    """Information about a single service in the incident topology."""

    name: str
    description: str
    log_lines: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    runbook_entry: str = ""
    is_relevant: bool = False  # whether investigating this yields useful clues


@dataclass
class Scenario:
    """Complete definition of an incident scenario (task)."""

    # Task metadata
    task_name: str
    task_difficulty: str  # easy, medium, hard
    max_steps: int

    # Incident details
    alert_summary: str
    severity: str  # P1, P2, P3
    task_description: str

    # Service topology
    services: Dict[str, ServiceInfo] = field(default_factory=dict)

    # Ground truth
    root_cause: str = ""
    root_cause_keywords: List[str] = field(default_factory=list)
    valid_remedies: List[Dict[str, str]] = field(default_factory=list)

    # Clue tracking
    relevant_services: Set[str] = field(default_factory=set)
    total_clues: int = 0


def _variant_bucket(seed: int) -> int:
    """Map an arbitrary seed to one of three deterministic variant buckets."""
    return abs(int(seed)) % 3


def _apply_seed_variant(scenario: Scenario, seed: int) -> Scenario:
    """Apply light-weight deterministic variants for anti-hardcoding robustness."""
    bucket = _variant_bucket(seed)
    rng = random.Random(10_000 + seed + len(scenario.task_name) * 37)

    suffixes = ["SLO watchdog", "synthetic canary", "latency sentinel"]
    scenario.alert_summary = f"{scenario.alert_summary} [{suffixes[bucket]}]"
    scenario.task_description = f"{scenario.task_description} (Scenario variant {bucket + 1}/3)"

    # Small deterministic metric perturbations preserve semantics while reducing overfitting.
    metric_delta = [0.0, 0.02, -0.015][bucket]
    for service in scenario.services.values():
        if "error_rate" in service.metrics:
            v = float(service.metrics["error_rate"])
            service.metrics["error_rate"] = round(max(0.0, min(0.99, v + metric_delta)), 3)
        if "latency_p99_ms" in service.metrics:
            v = float(service.metrics["latency_p99_ms"])
            service.metrics["latency_p99_ms"] = round(max(1.0, v * (1.0 + metric_delta)), 2)

    # Add one distractor-style operator message to a non-relevant service.
    distractors = [
        "[INFO]  observer: deployment canary healthy in unrelated subsystem",
        "[WARN]  observer: transient network jitter observed in non-critical path",
        "[INFO]  observer: routine maintenance log rotation completed",
    ]
    non_relevant = [s for s in scenario.services.values() if not s.is_relevant]
    if non_relevant:
        target = non_relevant[rng.randrange(len(non_relevant))]
        target.log_lines.append(distractors[bucket])

    return scenario


def create_easy_scenario(seed: int = 42) -> Scenario:
    """Task 1: Single-Service Config Error

    Payment service is returning 500 errors because the STRIPE_API_KEY
    environment variable was misconfigured during the latest deployment.

    Difficulty: Easy
    - Only one service is the root cause
    - Logs clearly show the error
    - Runbook has a direct fix procedure
    """
    services = {
        "payment-service": ServiceInfo(
            name="payment-service",
            description="Handles payment processing via Stripe API",
            log_lines=[
                "[2026-04-01T10:00:01Z] [INFO]  payment-service: Service started, version 2.4.1",
                "[2026-04-01T10:00:05Z] [INFO]  payment-service: Loading configuration from environment",
                "[2026-04-01T10:01:12Z] [ERROR] payment-service: Stripe API call failed: InvalidAPIKey - The API key provided is not valid",
                "[2026-04-01T10:01:12Z] [ERROR] payment-service: POST /api/v1/charge returned 500 Internal Server Error",
                "[2026-04-01T10:01:13Z] [WARN]  payment-service: Retrying failed external call (attempt 2/3)",
                "[2026-04-01T10:01:14Z] [ERROR] payment-service: Stripe API call failed: InvalidAPIKey - The API key provided is not valid",
                "[2026-04-01T10:01:14Z] [ERROR] payment-service: All retry attempts exhausted for POST /api/v1/charge",
                "[2026-04-01T10:01:15Z] [ERROR] payment-service: Returning 500 to upstream caller",
                "[2026-04-01T10:02:01Z] [ERROR] payment-service: Stripe API call failed: InvalidAPIKey - The API key provided is not valid",
                "[2026-04-01T10:02:01Z] [WARN]  payment-service: Error rate exceeded 50% threshold (current: 67%)",
                "[2026-04-01T10:03:00Z] [INFO]  payment-service: Health check: DEGRADED - high error rate",
                "[2026-04-01T10:03:30Z] [ERROR] payment-service: ALERT TRIGGERED: 500 error rate > 50%",
            ],
            metrics={
                "cpu_percent": 23.5,
                "memory_percent": 45.2,
                "error_rate": 0.67,
                "latency_p99_ms": 120.0,
                "requests_per_second": 45.0,
                "active_connections": 12,
                "uptime_hours": 0.05,
            },
            runbook_entry=(
                "## payment-service Runbook\n\n"
                "### Config Rollback Procedure\n"
                "If the service is failing due to a misconfigured environment variable:\n"
                "1. Identify the problematic config key from error logs\n"
                "2. Run: `rollback_config payment-service` to revert to last known good config\n"
                "3. Verify: Check error rate returns to baseline within 2 minutes\n\n"
                "### Common Issues\n"
                "- **InvalidAPIKey errors**: Usually caused by a deployment that overwrote "
                "the STRIPE_API_KEY with a test/invalid value. Fix: rollback config.\n"
                "- **High latency**: Check downstream dependency (Stripe API status page)\n"
                "- **Memory leak**: Restart service with `restart_service payment-service`\n"
            ),
            is_relevant=True,
        ),
        "api-gateway": ServiceInfo(
            name="api-gateway",
            description="Routes incoming HTTP requests to backend services",
            log_lines=[
                "[2026-04-01T10:00:00Z] [INFO]  api-gateway: Gateway healthy, routing active",
                "[2026-04-01T10:01:12Z] [WARN]  api-gateway: Upstream payment-service returned 500 for request req-8831",
                "[2026-04-01T10:01:14Z] [WARN]  api-gateway: Upstream payment-service returned 500 for request req-8832",
                "[2026-04-01T10:02:01Z] [WARN]  api-gateway: Upstream payment-service returned 500 for request req-8840",
                "[2026-04-01T10:03:00Z] [INFO]  api-gateway: Circuit breaker for payment-service: HALF-OPEN",
            ],
            metrics={
                "cpu_percent": 15.1,
                "memory_percent": 32.0,
                "error_rate": 0.12,
                "latency_p99_ms": 250.0,
                "requests_per_second": 120.0,
                "active_connections": 85,
                "uptime_hours": 720.3,
            },
            runbook_entry=(
                "## api-gateway Runbook\n\n"
                "The API gateway routes traffic to backend services.\n"
                "If upstream services are failing, the gateway will log warnings "
                "but the root cause is in the upstream service, not the gateway.\n\n"
                "### Circuit Breaker\n"
                "The gateway has automatic circuit breakers. If an upstream returns "
                ">50% errors, the circuit breaker opens to shed load.\n"
            ),
            is_relevant=False,
        ),
        "database": ServiceInfo(
            name="database",
            description="PostgreSQL primary database",
            log_lines=[
                "[2026-04-01T10:00:00Z] [INFO]  database: PostgreSQL 15.4 running, accepting connections",
                "[2026-04-01T10:01:00Z] [INFO]  database: Active connections: 23/100",
                "[2026-04-01T10:02:00Z] [INFO]  database: Active connections: 24/100",
                "[2026-04-01T10:03:00Z] [INFO]  database: Checkpoint completed, WAL size: 128MB",
            ],
            metrics={
                "cpu_percent": 18.7,
                "memory_percent": 55.3,
                "error_rate": 0.0,
                "latency_p99_ms": 8.5,
                "active_connections": 24,
                "connection_pool_max": 100,
                "uptime_hours": 2160.0,
            },
            runbook_entry=(
                "## database Runbook\n\n"
                "PostgreSQL primary database. Rarely the source of issues unless "
                "connection pool is exhausted or slow queries are detected.\n\n"
                "### Troubleshooting\n"
                "- Check `active_connections` vs `connection_pool_max`\n"
                "- Look for slow query warnings in logs\n"
            ),
            is_relevant=False,
        ),
    }

    scenario = Scenario(
        task_name="easy_config_error",
        task_difficulty="easy",
        max_steps=10,
        alert_summary="🚨 ALERT: payment-service: 500 Internal Server Error rate > 50% (current: 67%)",
        severity="P2",
        task_description=(
            "The payment service is returning HTTP 500 errors at a rate above 50%. "
            "Investigate the cause by checking logs and metrics, identify the root cause, "
            "and apply the correct fix from the runbook."
        ),
        services=services,
        root_cause="misconfigured STRIPE_API_KEY environment variable after deployment",
        root_cause_keywords=["stripe", "api_key", "apikey", "invalid", "config", "misconfigured", "environment variable"],
        valid_remedies=[
            {"service": "payment-service", "remedy": "rollback_config"},
        ],
        relevant_services={"payment-service"},
        total_clues=3,  # logs show error, metrics show error_rate, runbook has fix
    )
    return _apply_seed_variant(scenario, seed)


def create_medium_scenario(seed: int = 42) -> Scenario:
    """Task 2: Cascading Database Failure

    A long-running query on db-primary is holding locks, causing connection
    pool exhaustion. This cascades to API timeouts and queue backlogs.

    Difficulty: Medium
    - Multiple services affected, must trace upstream
    - Root cause is NOT the first service that shows symptoms
    - Requires correlating metrics across services
    """
    services = {
        "api-gateway": ServiceInfo(
            name="api-gateway",
            description="Routes incoming HTTP requests to backend services",
            log_lines=[
                "[2026-04-01T14:00:00Z] [INFO]  api-gateway: Gateway healthy",
                "[2026-04-01T14:05:12Z] [WARN]  api-gateway: Upstream user-service response time > 3s for req-12001",
                "[2026-04-01T14:05:15Z] [WARN]  api-gateway: Upstream order-service response time > 5s for req-12002",
                "[2026-04-01T14:05:30Z] [ERROR] api-gateway: Upstream user-service timeout (10s) for req-12010",
                "[2026-04-01T14:06:00Z] [ERROR] api-gateway: Multiple upstream timeouts detected, overall latency P99 > 5s",
                "[2026-04-01T14:06:30Z] [WARN]  api-gateway: Request queue depth: 247 (threshold: 100)",
                "[2026-04-01T14:07:00Z] [ERROR] api-gateway: ALERT TRIGGERED: response latency > 5s, error rate rising",
            ],
            metrics={
                "cpu_percent": 45.2,
                "memory_percent": 52.1,
                "error_rate": 0.35,
                "latency_p99_ms": 8500.0,
                "requests_per_second": 180.0,
                "active_connections": 195,
                "queue_depth": 247,
                "uptime_hours": 720.0,
            },
            runbook_entry=(
                "## api-gateway Runbook\n\n"
                "If the gateway shows high latency, the root cause is usually "
                "in downstream services, not the gateway itself.\n\n"
                "### Diagnosis Steps\n"
                "1. Check which upstream services are timing out (look for WARN/ERROR logs)\n"
                "2. Investigate those services' dependencies (database, cache, etc.)\n"
                "3. The gateway has automatic backpressure — fixing the upstream fixes the gateway\n"
            ),
            is_relevant=True,
        ),
        "user-service": ServiceInfo(
            name="user-service",
            description="Handles user authentication and profile operations",
            log_lines=[
                "[2026-04-01T14:00:00Z] [INFO]  user-service: Service healthy",
                "[2026-04-01T14:05:10Z] [WARN]  user-service: Database query took 4.2s (threshold: 1s)",
                "[2026-04-01T14:05:12Z] [WARN]  user-service: Connection pool wait time > 2s",
                "[2026-04-01T14:05:30Z] [ERROR] user-service: Failed to acquire DB connection within timeout (10s)",
                "[2026-04-01T14:06:00Z] [ERROR] user-service: GET /api/users/profile returned 503 Service Unavailable",
                "[2026-04-01T14:06:15Z] [WARN]  user-service: DB connection pool exhausted (50/50 in use)",
            ],
            metrics={
                "cpu_percent": 32.0,
                "memory_percent": 48.5,
                "error_rate": 0.42,
                "latency_p99_ms": 6200.0,
                "requests_per_second": 60.0,
                "active_connections": 50,
                "db_pool_size": 50,
                "db_pool_used": 50,
                "uptime_hours": 168.0,
            },
            runbook_entry=(
                "## user-service Runbook\n\n"
                "### Database Connection Issues\n"
                "If DB connection pool is exhausted, check db-primary for:\n"
                "- Long-running queries holding locks\n"
                "- Connection count approaching max\n\n"
                "The user-service does NOT have independent DB issues — "
                "it shares the connection pool with db-primary.\n"
            ),
            is_relevant=True,
        ),
        "order-service": ServiceInfo(
            name="order-service",
            description="Manages order creation and processing",
            log_lines=[
                "[2026-04-01T14:00:00Z] [INFO]  order-service: Service healthy",
                "[2026-04-01T14:05:15Z] [WARN]  order-service: Database query took 5.8s (threshold: 1s)",
                "[2026-04-01T14:05:20Z] [ERROR] order-service: Failed to acquire DB connection within timeout (10s)",
                "[2026-04-01T14:06:00Z] [ERROR] order-service: POST /api/orders returned 503 Service Unavailable",
                "[2026-04-01T14:06:30Z] [WARN]  order-service: Message queue backlog growing: 1,245 pending messages",
            ],
            metrics={
                "cpu_percent": 28.5,
                "memory_percent": 41.0,
                "error_rate": 0.38,
                "latency_p99_ms": 7800.0,
                "requests_per_second": 40.0,
                "active_connections": 50,
                "db_pool_size": 50,
                "db_pool_used": 50,
                "queue_backlog": 1245,
                "uptime_hours": 168.0,
            },
            runbook_entry=(
                "## order-service Runbook\n\n"
                "### Database Issues\n"
                "Similar to user-service, relies on db-primary. "
                "If DB connections are exhausted, fix db-primary first.\n\n"
                "### Queue Backlog\n"
                "Queue backlog is a SYMPTOM of slow processing. "
                "Fix the database issue and the backlog will drain.\n"
            ),
            is_relevant=True,
        ),
        "db-primary": ServiceInfo(
            name="db-primary",
            description="PostgreSQL primary database serving all backend services",
            log_lines=[
                "[2026-04-01T14:00:00Z] [INFO]  db-primary: PostgreSQL 15.4 running",
                "[2026-04-01T14:03:00Z] [WARN]  db-primary: Slow query detected (duration: 45.2s): "
                "SELECT o.*, u.email FROM orders o JOIN users u ON o.user_id = u.id "
                "WHERE o.created_at > '2020-01-01' ORDER BY o.total DESC",
                "[2026-04-01T14:04:00Z] [WARN]  db-primary: Active connections: 95/100 (approaching limit)",
                "[2026-04-01T14:04:30Z] [WARN]  db-primary: Lock wait timeout detected: 3 transactions waiting",
                "[2026-04-01T14:05:00Z] [ERROR] db-primary: Connection pool EXHAUSTED: 100/100 connections in use",
                "[2026-04-01T14:05:30Z] [ERROR] db-primary: Rejecting new connections — pool full",
                "[2026-04-01T14:06:00Z] [WARN]  db-primary: Long-running query still active (PID 14523, duration: 225s)",
            ],
            metrics={
                "cpu_percent": 87.3,
                "memory_percent": 72.1,
                "error_rate": 0.02,
                "latency_p99_ms": 4200.0,
                "active_connections": 100,
                "connection_pool_max": 100,
                "lock_wait_count": 3,
                "slow_query_count": 1,
                "longest_query_seconds": 225.0,
                "uptime_hours": 2160.0,
            },
            runbook_entry=(
                "## db-primary Runbook\n\n"
                "### Long-Running Query / Lock Resolution\n"
                "If a slow query is holding locks and exhausting connections:\n"
                "1. Identify the query PID from logs (look for 'Long-running query' or 'Slow query')\n"
                "2. Run: `kill_query db-primary` to terminate the offending query\n"
                "3. Run: `scale_connections db-primary` to temporarily increase pool size\n"
                "4. Verify: Connection count drops, downstream services recover\n\n"
                "### Common Root Causes\n"
                "- Unoptimized queries scanning large tables without proper indexes\n"
                "- Missing WHERE clause on historical data queries\n"
                "- Deadlocks from concurrent transactions\n"
            ),
            is_relevant=True,
        ),
        "cache-redis": ServiceInfo(
            name="cache-redis",
            description="Redis cache for frequently accessed data",
            log_lines=[
                "[2026-04-01T14:00:00Z] [INFO]  cache-redis: Redis 7.2 running, memory usage: 2.1GB/8GB",
                "[2026-04-01T14:05:00Z] [WARN]  cache-redis: Cache miss rate elevated: 45% (baseline: 15%)",
                "[2026-04-01T14:06:00Z] [INFO]  cache-redis: Eviction count increased — memory pressure moderate",
            ],
            metrics={
                "cpu_percent": 12.0,
                "memory_percent": 26.3,
                "cache_hit_rate": 0.55,
                "cache_miss_rate": 0.45,
                "evictions_per_second": 8.0,
                "connected_clients": 90,
                "uptime_hours": 2160.0,
            },
            runbook_entry=(
                "## cache-redis Runbook\n\n"
                "Cache miss rate spikes are usually a SYMPTOM of database issues, "
                "not a cache problem. When the DB is slow, cache population fails "
                "and miss rate rises.\n\n"
                "Fix the database issue and cache will repopulate naturally.\n"
            ),
            is_relevant=False,
        ),
    }

    scenario = Scenario(
        task_name="medium_cascading_db",
        task_difficulty="medium",
        max_steps=15,
        alert_summary=(
            "🚨 ALERT: api-gateway: response latency P99 > 5s (current: 8.5s), "
            "error rate rising (35%). Multiple services affected."
        ),
        severity="P1",
        task_description=(
            "The API gateway is showing high latency and rising error rates. "
            "Multiple backend services appear affected. Investigate the root cause "
            "by tracing the issue through the service dependency chain, identify "
            "the root cause, and apply the correct remediation."
        ),
        services=services,
        root_cause=(
            "Long-running unoptimized SQL query on db-primary holding locks "
            "and exhausting the connection pool, causing cascading failures "
            "in user-service and order-service"
        ),
        root_cause_keywords=[
            "slow query", "long-running", "query", "db-primary", "database",
            "connection pool", "exhausted", "lock", "sql",
        ],
        valid_remedies=[
            {"service": "db-primary", "remedy": "kill_query"},
            {"service": "db-primary", "remedy": "scale_connections"},
        ],
        relevant_services={"api-gateway", "user-service", "order-service", "db-primary"},
        total_clues=5,
    )
    return _apply_seed_variant(scenario, seed)


def create_hard_scenario(seed: int = 42) -> Scenario:
    """Task 3: Intermittent Auth Failure During Key Rotation

    Random 401 Unauthorized errors on the auth service during peak hours.
    Caused by a race condition between the key-rotation-service rotating
    JWT signing keys and the token-cache not invalidating old tokens
    fast enough.

    Difficulty: Hard
    - Intermittent (not constant) — harder to diagnose
    - Requires correlating timestamps across multiple services
    - Root cause involves interaction between two services
    - Must read runbook to find the specific hotfix
    """
    services = {
        "auth-service": ServiceInfo(
            name="auth-service",
            description="Handles authentication, JWT token issuance and validation",
            log_lines=[
                "[2026-04-01T18:00:00Z] [INFO]  auth-service: Service healthy, JWT validation active",
                "[2026-04-01T18:15:32Z] [WARN]  auth-service: Token validation failed for user usr-4421: signature mismatch",
                "[2026-04-01T18:15:33Z] [INFO]  auth-service: Token validation succeeded for user usr-4422",
                "[2026-04-01T18:15:34Z] [WARN]  auth-service: Token validation failed for user usr-4423: signature mismatch",
                "[2026-04-01T18:15:34Z] [INFO]  auth-service: Token validation succeeded for user usr-4424",
                "[2026-04-01T18:15:35Z] [INFO]  auth-service: Token validation succeeded for user usr-4425",
                "[2026-04-01T18:15:36Z] [WARN]  auth-service: Token validation failed for user usr-4426: key ID not found in active key set",
                "[2026-04-01T18:30:00Z] [WARN]  auth-service: Intermittent 401 rate: 18% (baseline: 0.5%)",
                "[2026-04-01T18:30:01Z] [INFO]  auth-service: Note: failures correlate with tokens issued before 18:15:00Z",
                "[2026-04-01T18:45:00Z] [WARN]  auth-service: 401 rate decreased to 8% as old tokens expire naturally",
                "[2026-04-01T19:00:00Z] [ERROR] auth-service: ALERT: intermittent 401 errors during peak hours",
            ],
            metrics={
                "cpu_percent": 35.2,
                "memory_percent": 41.0,
                "error_rate": 0.18,
                "latency_p99_ms": 95.0,
                "requests_per_second": 250.0,
                "active_connections": 120,
                "auth_success_rate": 0.82,
                "auth_failure_rate": 0.18,
                "uptime_hours": 720.0,
            },
            runbook_entry=(
                "## auth-service Runbook\n\n"
                "### Known Issue: Token Validation Race Condition (HIGH PRIORITY)\n"
                "**Symptoms**: Intermittent 401 errors that correlate with key rotation events.\n"
                "**Root Cause**: When key-rotation-service rotates JWT signing keys, "
                "the token-cache may still serve tokens signed with the OLD key. "
                "The auth-service then fails to validate these tokens because it only "
                "has the NEW key in its active set.\n\n"
                "**Fix**: Apply the race condition hotfix:\n"
                "  `apply_hotfix auth-service token-cache-race-fix`\n\n"
                "This hotfix adds a grace period that keeps the previous signing key "
                "active for 10 minutes after rotation, allowing cached tokens to expire "
                "naturally.\n\n"
                "### Other Common Issues\n"
                "- **All tokens failing**: Check if auth-service can reach the key store\n"
                "- **High latency**: Check token-cache health\n"
            ),
            is_relevant=True,
        ),
        "token-cache": ServiceInfo(
            name="token-cache",
            description="Caches validated JWT tokens to reduce auth-service load",
            log_lines=[
                "[2026-04-01T18:00:00Z] [INFO]  token-cache: Cache healthy, hit rate: 92%",
                "[2026-04-01T18:15:00Z] [INFO]  token-cache: Key rotation event received, invalidating affected entries",
                "[2026-04-01T18:15:01Z] [WARN]  token-cache: Bulk invalidation in progress — 12,450 entries to process",
                "[2026-04-01T18:15:15Z] [WARN]  token-cache: Invalidation still in progress — 8,200 entries remaining",
                "[2026-04-01T18:15:30Z] [WARN]  token-cache: Invalidation still in progress — 4,100 entries remaining",
                "[2026-04-01T18:16:00Z] [INFO]  token-cache: Invalidation complete",
                "[2026-04-01T18:16:01Z] [WARN]  token-cache: During invalidation window (60s), some stale tokens may have been served",
                "[2026-04-01T18:30:00Z] [INFO]  token-cache: Hit rate recovering: 78%",
            ],
            metrics={
                "cpu_percent": 18.0,
                "memory_percent": 35.5,
                "cache_hit_rate": 0.78,
                "cache_miss_rate": 0.22,
                "stale_entries_served": 342,
                "invalidation_lag_seconds": 60.0,
                "ttl_seconds": 300,
                "uptime_hours": 720.0,
            },
            runbook_entry=(
                "## token-cache Runbook\n\n"
                "### Cache Invalidation Lag\n"
                "During key rotation events, the cache takes ~60 seconds to fully "
                "invalidate old tokens. This is a known limitation.\n\n"
                "The fix is in auth-service (apply the race condition hotfix), "
                "not in the cache itself.\n"
            ),
            is_relevant=True,
        ),
        "key-rotation-service": ServiceInfo(
            name="key-rotation-service",
            description="Rotates JWT signing keys on a scheduled basis",
            log_lines=[
                "[2026-04-01T18:00:00Z] [INFO]  key-rotation-service: Service healthy",
                "[2026-04-01T18:15:00Z] [INFO]  key-rotation-service: Scheduled key rotation triggered",
                "[2026-04-01T18:15:00Z] [INFO]  key-rotation-service: New signing key generated: key-id-2026-04-01-v2",
                "[2026-04-01T18:15:01Z] [INFO]  key-rotation-service: Old key deactivated: key-id-2026-03-31-v1",
                "[2026-04-01T18:15:01Z] [INFO]  key-rotation-service: Notified auth-service and token-cache of rotation",
                "[2026-04-01T18:15:02Z] [INFO]  key-rotation-service: Key rotation complete",
            ],
            metrics={
                "cpu_percent": 5.0,
                "memory_percent": 12.0,
                "last_rotation_timestamp": 1743530100.0,
                "rotation_count_24h": 1,
                "uptime_hours": 720.0,
            },
            runbook_entry=(
                "## key-rotation-service Runbook\n\n"
                "Rotates JWT signing keys daily at peak hours (configurable).\n"
                "The rotation itself is working correctly — the issue is in how "
                "downstream services handle the transition.\n\n"
                "See auth-service runbook for the known race condition fix.\n"
            ),
            is_relevant=True,
        ),
        "session-store": ServiceInfo(
            name="session-store",
            description="Stores user session data in Redis",
            log_lines=[
                "[2026-04-01T18:00:00Z] [INFO]  session-store: Healthy, 15,234 active sessions",
                "[2026-04-01T18:15:00Z] [INFO]  session-store: Session count stable",
                "[2026-04-01T18:30:00Z] [WARN]  session-store: Slight increase in session creation rate (users re-authenticating)",
            ],
            metrics={
                "cpu_percent": 8.0,
                "memory_percent": 22.0,
                "active_sessions": 15234,
                "sessions_created_per_minute": 45.0,
                "uptime_hours": 720.0,
            },
            runbook_entry=(
                "## session-store Runbook\n\n"
                "Stores session data. Increased session creation might indicate "
                "auth issues forcing users to re-login. This is a SYMPTOM.\n"
            ),
            is_relevant=False,
        ),
        "api-gateway": ServiceInfo(
            name="api-gateway",
            description="Routes incoming HTTP requests to backend services",
            log_lines=[
                "[2026-04-01T18:00:00Z] [INFO]  api-gateway: Gateway healthy",
                "[2026-04-01T18:15:35Z] [WARN]  api-gateway: Upstream auth-service returned 401 for req-22001",
                "[2026-04-01T18:15:36Z] [INFO]  api-gateway: Request req-22002 authenticated successfully",
                "[2026-04-01T18:30:00Z] [INFO]  api-gateway: 401 rate from auth-service: 18%",
            ],
            metrics={
                "cpu_percent": 20.0,
                "memory_percent": 30.0,
                "error_rate": 0.05,
                "latency_p99_ms": 150.0,
                "requests_per_second": 300.0,
                "uptime_hours": 720.0,
            },
            runbook_entry=(
                "## api-gateway Runbook\n\n"
                "If seeing 401s from auth-service, the issue is in auth — not the gateway.\n"
            ),
            is_relevant=False,
        ),
        "load-balancer": ServiceInfo(
            name="load-balancer",
            description="Distributes traffic across service instances",
            log_lines=[
                "[2026-04-01T18:00:00Z] [INFO]  load-balancer: All backends healthy",
                "[2026-04-01T18:15:00Z] [INFO]  load-balancer: Peak traffic period detected — RPS: 300",
                "[2026-04-01T18:30:00Z] [INFO]  load-balancer: Traffic distribution normal",
            ],
            metrics={
                "cpu_percent": 10.0,
                "memory_percent": 15.0,
                "requests_per_second": 300.0,
                "backend_health_checks_passed": 6,
                "backend_health_checks_failed": 0,
                "uptime_hours": 8760.0,
            },
            runbook_entry=(
                "## load-balancer Runbook\n\n"
                "Load balancer is healthy. If errors are occurring, "
                "they are in the backend services, not the LB.\n"
            ),
            is_relevant=False,
        ),
        "user-db": ServiceInfo(
            name="user-db",
            description="User data storage (PostgreSQL)",
            log_lines=[
                "[2026-04-01T18:00:00Z] [INFO]  user-db: PostgreSQL healthy, connections: 30/100",
                "[2026-04-01T18:15:00Z] [INFO]  user-db: Normal operations",
            ],
            metrics={
                "cpu_percent": 15.0,
                "memory_percent": 40.0,
                "active_connections": 30,
                "connection_pool_max": 100,
                "uptime_hours": 2160.0,
            },
            runbook_entry="## user-db Runbook\n\nUser database is healthy. No known issues.\n",
            is_relevant=False,
        ),
    }

    scenario = Scenario(
        task_name="hard_intermittent_auth",
        task_difficulty="hard",
        max_steps=20,
        alert_summary=(
            "🚨 ALERT: auth-service: intermittent 401 Unauthorized errors during peak hours. "
            "Auth failure rate: 18% (baseline: 0.5%). Pattern: sporadic, not affecting all users."
        ),
        severity="P1",
        task_description=(
            "The auth service is returning intermittent 401 errors during peak hours. "
            "Not all requests are failing — the pattern is sporadic. "
            "This requires investigating multiple services, correlating timestamps, "
            "and identifying the interaction between services that causes the failures. "
            "Find the root cause and apply the specific fix from the runbook."
        ),
        services=services,
        root_cause=(
            "Race condition between key-rotation-service rotating JWT signing keys "
            "and token-cache not invalidating stale tokens fast enough. "
            "During the 60-second invalidation window, auth-service receives tokens "
            "signed with the old (now deactivated) key and rejects them."
        ),
        root_cause_keywords=[
            "race condition", "key rotation", "token", "cache", "invalidation",
            "stale", "signing key", "jwt", "old key", "grace period",
        ],
        valid_remedies=[
            {"service": "auth-service", "remedy": "apply_hotfix token-cache-race-fix"},
            {"service": "auth-service", "remedy": "token-cache-race-fix"},
        ],
        relevant_services={"auth-service", "token-cache", "key-rotation-service"},
        total_clues=6,
    )
    return _apply_seed_variant(scenario, seed)


# Registry of all available scenarios
SCENARIOS = {
    "easy_config_error": create_easy_scenario,
    "medium_cascading_db": create_medium_scenario,
    "hard_intermittent_auth": create_hard_scenario,
}


def get_scenario(task_name: str, seed: int = 42) -> Scenario:
    """Get a scenario by task name."""
    if task_name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    return SCENARIOS[task_name](seed=seed)


def list_tasks() -> List[str]:
    """Return all available task names."""
    return list(SCENARIOS.keys())
