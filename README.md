---
title: Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - incident-response
---


An OpenEnv environment for training AI agents to investigate and resolve production incidents. The agent acts as an on-call engineer who receives alerts, queries logs and metrics, reads runbooks, identifies root causes, and applies remediation actions.

## Why This Environment?

Every tech company — from startups to Meta — runs on-call rotations. When production breaks, engineers must:
1. **Receive an alert** → understand the problem scope
2. **Investigate** → query logs, check metrics, read runbooks
3. **Identify root cause** → correlate signals across services
4. **Remediate** → apply the correct fix from operational procedures
5. **Verify** → confirm the incident is resolved

This is a **multi-step sequential decision problem** — perfect for training RL agents. Yet no RL environment existed for this critical real-world task.

## Tasks

| Task | Difficulty | Scenario | Max Steps |
|------|-----------|----------|-----------|
| `easy_config_error` | ⭐ Easy | Payment service returning 500s due to misconfigured API key | 10 |
| `medium_cascading_db` | ⭐⭐ Medium | Cascading failure from a long-running database query exhausting connection pools | 15 |
| `hard_intermittent_auth` | ⭐⭐⭐ Hard | Intermittent 401 errors caused by a race condition between JWT key rotation and token cache invalidation | 20 |

Each task uses deterministic seeded variants (same objective, shifted telemetry and distractors) to reduce hardcoded-policy overfitting.

## Action Space

The agent interacts via 9 tools:

| Tool | Parameters | Description |
|------|-----------|-------------|
| `query_logs` | `service`, `filter` | Query log output for a service |
| `check_metrics` | `service` | Check CPU, memory, error rate, latency, etc. |
| `read_runbook` | `service` | Read operational procedures and known fixes |
| `identify_root_cause` | `cause` | Declare the root cause of the incident |
| `execute_remedy` | `service`, `remedy` | Apply a remediation action |
| `escalate` | `reason` | Escalate to senior on-call (penalty) |
| `get_status` | — | Get current investigation summary |
| `communicate_status` | `message` | Send a concise stakeholder update |
| `noop` | — | Explicit no-op (tracked and penalized) |

## Observation Space

After each action, the agent receives:
- **Alert summary**: The initial alert text
- **Severity**: P1 (critical), P2 (high), P3 (medium)
- **Current findings**: List of discoveries made so far
- **Available services**: Services that can be queried
- **Last action result**: Output of the previous tool call
- **Time elapsed**: Simulated minutes since alert fired
- **Step count**: Current step / max steps

## Reward Function

Scores are strictly in (0.0, 1.0), composed of:

| Component | Max Score | Description |
|-----------|-----------|-------------|
| Root cause identification | 0.25 | Correct diagnosis |
| Partial diagnosis | 0.08 | Incomplete but directionally correct |
| Evidence-supported diagnosis | 0.08 | Root-cause claim backed by logs + metrics |
| Correct remedy | 0.28 | Applied the right fix from the runbook |
| Post-fix verification | 0.05 | Confirmed after mitigation |
| Communication discipline | 0.04 | Best score for exactly one concise update |
| Efficiency | 0.12 | Faster resolution with fewer steps |
| Clue discovery | 0.18 | Proportion of relevant evidence found |
| **Penalties** | | |
| Wrong remedy | -0.15 | Per incorrect remedy attempt |
| Destructive action | -0.30 | Blocked dangerous operations |
| Unnecessary escalation | -0.10 | Should have resolved independently |
| Risky pre-diagnosis action | -0.05 | Broad/high-risk mitigation before diagnosis |
| No-op | -0.02 | Per no-op action |

## Anti-Exploit Design

- Destructive remediations are blocked and strongly penalized.
- Escalation is allowed but heavily penalized when used as a shortcut.
- No-op and repetitive low-information behavior are penalized.
- Root-cause claims receive higher credit only when backed by evidence from logs and metrics.

## Determinism & Explainability

- Terminal observations include a deterministic score breakdown (`score_breakdown`) with all positive and penalty components.
- `get_status` includes a score preview and variant metadata for debugging/review.
- Seeded scenarios are deterministic and reproducible.

## Setup

### Prerequisites
- Python 3.10+
- `pip install openenv-core[core]`

### Local Development
```bash
cd incident_env

# Install dependencies
pip install -e .

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run the baseline agent
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
python inference.py
```

### Docker
```bash
docker build -t incident-env:latest -f server/Dockerfile .
docker run -p 8000:8000 incident-env:latest
```

## Human Review & API Testing

This submission supports both reviewer UI and API testing:

- `GET /` opens the reviewer cockpit UI.
- `GET /ui` also serves the same cockpit directly.
- Core API endpoints (`/reset`, `/step`, `/state`, `/tasks`, `/schema`) remain unchanged.

To validate behavior manually, use the API endpoints:
1. POST `/reset` with `{ "task_name": "easy_config_error" }`.
2. POST `/step` with action payloads, for example:
   - `query_logs` on `payment-service` with `{"filter": "error"}`.
   - `read_runbook` on `payment-service`.
   - `identify_root_cause` with cause text.
   - `execute_remedy` with `{"service": "payment-service", "remedy": "rollback_config"}`.
3. GET `/state` to inspect progress metadata.

### HuggingFace Space
```bash
openenv push --repo-id makrand098/incident-env
```

### Reproducibility Report
```bash
python scripts/reproducibility_report.py
```

This script runs a fixed deterministic policy across fixed seeds and prints mean/min/max scores per task.

## Baseline Scores

| Task | Baseline Score (max reward per step) |
|------|--------------------------------------|
| easy_config_error | 0.24 |
| medium_cascading_db | 0.09 |
| hard_intermittent_auth | 0.24 |

## Architecture

```
incident_env/
├── models.py              # Action, Observation, State Pydantic models
├── client.py              # Standard OpenEnv HTTP Client
├── inference.py           # Baseline LLM agent
├── openenv.yaml           # Environment manifest
├── pyproject.toml         # Dependencies
└── server/
    ├── app.py             # FastAPI server
    ├── incident_environment.py  # Core environment logic
    ├── scenarios.py       # 3 incident scenario definitions
    ├── graders.py         # Scoring logic in strict (0.0, 1.0)
    └── Dockerfile         # Container definition
```

## License

This project was created for the Meta x Scaler OpenEnv AI Hackathon 2026.
