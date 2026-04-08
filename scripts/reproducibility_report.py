"""Generate deterministic reproducibility stats across fixed seeds.

Run from project root:
    python scripts/reproducibility_report.py
"""

from __future__ import annotations

import os
from statistics import mean
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import IncidentAction
from server.incident_environment import IncidentEnvironment


SEEDS = [11, 22, 33, 44, 55]
TASKS = ["easy_config_error", "medium_cascading_db", "hard_intermittent_auth"]

GOOD_PATHS: Dict[str, List[IncidentAction]] = {
    "easy_config_error": [
        IncidentAction(action_type="query_logs", target="payment-service", parameters={"filter": "error"}),
        IncidentAction(action_type="check_metrics", target="payment-service", parameters={}),
        IncidentAction(action_type="read_runbook", target="payment-service", parameters={}),
        IncidentAction(action_type="identify_root_cause", target="", parameters={"cause": "misconfigured STRIPE_API_KEY environment variable"}),
        IncidentAction(action_type="get_status", target="", parameters={}),
        IncidentAction(action_type="execute_remedy", target="", parameters={"service": "payment-service", "remedy": "rollback_config"}),
    ],
    "medium_cascading_db": [
        IncidentAction(action_type="query_logs", target="api-gateway", parameters={"filter": "timeout"}),
        IncidentAction(action_type="check_metrics", target="db-primary", parameters={}),
        IncidentAction(action_type="query_logs", target="db-primary", parameters={"filter": "query"}),
        IncidentAction(action_type="read_runbook", target="db-primary", parameters={}),
        IncidentAction(action_type="identify_root_cause", target="", parameters={"cause": "long-running query exhausting db-primary connection pool and causing cascading failures"}),
        IncidentAction(action_type="get_status", target="", parameters={}),
        IncidentAction(action_type="execute_remedy", target="", parameters={"service": "db-primary", "remedy": "kill_query"}),
    ],
    "hard_intermittent_auth": [
        IncidentAction(action_type="query_logs", target="auth-service", parameters={"filter": "401"}),
        IncidentAction(action_type="check_metrics", target="token-cache", parameters={}),
        IncidentAction(action_type="query_logs", target="key-rotation-service", parameters={"filter": "rotation"}),
        IncidentAction(action_type="read_runbook", target="auth-service", parameters={}),
        IncidentAction(action_type="identify_root_cause", target="", parameters={"cause": "race condition between key rotation and token-cache invalidation causing stale token validation failures"}),
        IncidentAction(action_type="get_status", target="", parameters={}),
        IncidentAction(action_type="execute_remedy", target="", parameters={"service": "auth-service", "remedy": "token-cache-race-fix"}),
    ],
}


def run_episode(task_name: str, seed: int) -> float:
    env = IncidentEnvironment()
    env.reset(task_name=task_name, seed=seed)
    final_obs = None

    for action in GOOD_PATHS[task_name]:
        final_obs = env.step(action)
        if final_obs.done:
            break

    if final_obs is None:
        return 0.01
    return float(final_obs.final_score)


def main() -> None:
    print("Reproducibility report (fixed policy, fixed seeds)")
    print("Seeds:", ", ".join(str(s) for s in SEEDS))
    print()
    print("| Task | Mean Score | Min | Max |")
    print("|------|------------|-----|-----|")

    for task in TASKS:
        scores = [run_episode(task, seed) for seed in SEEDS]
        print(f"| {task} | {mean(scores):.4f} | {min(scores):.4f} | {max(scores):.4f} |")


if __name__ == "__main__":
    main()
