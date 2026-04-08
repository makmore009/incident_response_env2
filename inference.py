"""Baseline inference script for the Incident Response Environment.

Follows the strict [START]/[STEP]/[END] logging format required by the
hackathon evaluation pipeline.
"""

import json
import math
import os
import re
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from client import IncidentEnv
from models import IncidentAction

# ── Configuration ──────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
MAX_STEPS_PER_TASK = 20
TEMPERATURE = 0.1
MAX_TOKENS = 300
ENV_NAME = "incident_env"

TASKS = ["easy_config_error", "medium_cascading_db", "hard_intermittent_auth"]
VALID_ACTIONS = {
    "query_logs", "check_metrics", "read_runbook", "identify_root_cause",
    "execute_remedy", "escalate", "get_status", "communicate_status", "noop",
}

TASK_CONTEXT = {
    "easy_config_error": {
        "services": ["payment-service", "api-gateway", "database"],
        "hint": "Start with payment-service — it's mentioned in the alert.",
    },
    "medium_cascading_db": {
        "services": ["api-gateway", "user-service", "order-service", "db-primary", "cache-redis"],
        "hint": "Multiple services are affected. Trace the dependency chain — check db-primary.",
    },
    "hard_intermittent_auth": {
        "services": ["auth-service", "token-cache", "key-rotation-service", "session-store", "api-gateway", "load-balancer", "user-db"],
        "hint": "Errors are intermittent. Check auth-service first, then correlate with key-rotation-service and token-cache.",
    },
}


def get_system_prompt(task_name: str) -> str:
    ctx = TASK_CONTEXT.get(task_name, {"services": [], "hint": ""})
    services_str = ", ".join(f"'{s}'" for s in ctx["services"])

    return textwrap.dedent(f"""\
        You are an expert on-call engineer responding to a production incident.

        AVAILABLE SERVICES: {services_str}
        IMPORTANT: You MUST use ONLY these exact service names.

        RESPOND WITH EXACTLY ONE JSON OBJECT per turn choosing an action:

        Available action_types:
        1. query_logs — Check logs. target=service name, parameters={{"filter": "error"}}
        2. check_metrics — Check metrics. target=service name
        3. read_runbook — Read operational runbook. target=service name
        4. identify_root_cause — Declare root cause. parameters={{"cause": "description"}}
        5. execute_remedy — Apply fix. parameters={{"service": "name", "remedy": "fix_name"}}
        6. escalate — Escalate (LAST RESORT). parameters={{"reason": "why"}}
        7. get_status — Check investigation progress.

        FORMAT: {{"action_type": "...", "target": "...", "parameters": {{...}}}}

        STRATEGY:
        - {ctx['hint']}
        - Check logs first (with filter="error"), then metrics, then runbook
        - Read the runbook to find the specific remedy command
        - Identify root cause BEFORE applying remedy
        - Use the EXACT remedy name from the runbook

        RESPOND WITH EXACTLY ONE JSON OBJECT PER TURN. Nothing else.
    """)


def parse_action(response_text: str) -> Optional[Dict]:
    """Extract a JSON action from LLM output."""
    json_pattern = re.compile(r'\{[^{}]*"action_type"[^{}]*\}', re.DOTALL)
    matches = json_pattern.findall(response_text)

    for match in matches:
        try:
            parsed = json.loads(match)
            if "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try nested JSON
    try:
        brace_count = 0
        start = response_text.find('{')
        if start >= 0:
            for i in range(start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        parsed = json.loads(response_text[start:i+1])
                        if "action_type" in parsed:
                            return parsed
                        break
    except (json.JSONDecodeError, ValueError):
        pass

    # Backwards compat: try old {"tool": ..., "args": ...} format
    try:
        tool_pattern = re.compile(r'\{[^{}]*"tool"[^{}]*\}', re.DOTALL)
        tool_matches = tool_pattern.findall(response_text)
        for match in tool_matches:
            parsed = json.loads(match)
            if "tool" in parsed:
                return {
                    "action_type": parsed["tool"],
                    "target": parsed.get("args", {}).get("service", ""),
                    "parameters": parsed.get("args", {}),
                }
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def format_action_str(action: Dict) -> str:
    action_type = action.get("action_type", "noop")
    target = action.get("target", "")
    params = action.get("parameters", {})

    parts = []
    if target:
        parts.append(f"'{target}'")
    for k, v in params.items():
        if k != "service" or not target:
            parts.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")
    return f"{action_type}({','.join(parts)})"


def clamp_reward(value: Optional[float]) -> float:
    try:
        numeric = float(value if value is not None else 0.01)
    except (TypeError, ValueError):
        numeric = 0.01
    if not math.isfinite(numeric):
        numeric = 0.01
    return round(max(0.01, min(0.95, numeric)), 2)


def sanitize_action(task_name: str, action: Dict, recent_action_keys: List[str]) -> Dict:
    """Normalize model output into a safe, valid IncidentAction payload."""
    services = TASK_CONTEXT.get(task_name, {}).get("services", [])
    default_service = services[0] if services else "payment-service"

    action_type = str(action.get("action_type", "noop")).strip().lower()
    if action_type not in VALID_ACTIONS:
        action_type = "noop"

    target = str(action.get("target", "") or "").strip()
    params = action.get("parameters", {})
    if not isinstance(params, dict):
        params = {}

    # Bias away from repeated low-information loops.
    action_key = f"{action_type}:{target}:{json.dumps(params, sort_keys=True)}"
    if len(recent_action_keys) >= 2 and recent_action_keys[-1] == action_key == recent_action_keys[-2]:
        action_type = "get_status"
        target = ""
        params = {}

    if action_type in {"query_logs", "check_metrics", "read_runbook"}:
        if not target or target not in services:
            target = default_service

    if action_type == "execute_remedy":
        service = str(params.get("service", target or default_service)).strip()
        remedy = str(params.get("remedy", "")).strip() or "rollback_config"
        params = {"service": service, "remedy": remedy}
        target = target or service

    if action_type == "identify_root_cause":
        cause = str(params.get("cause", target)).strip()
        if not cause:
            cause = "Likely service configuration mismatch causing incident symptoms"
        params = {"cause": cause}
        target = ""

    if action_type == "communicate_status":
        message = str(params.get("message", "Investigating and collecting evidence.")).strip()
        params = {"message": message}
        target = ""

    if action_type in {"get_status", "noop", "escalate"}:
        if action_type == "escalate":
            reason = str(params.get("reason", "Unable to progress safely")).strip()
            params = {"reason": reason}
        else:
            params = {}
        target = ""

    return {"action_type": action_type, "target": target, "parameters": params}


def run_task(client: OpenAI, env_base_url: str, task_name: str) -> tuple:
    """Run a single task. Returns (success, steps, score, rewards_list)."""

    rewards: List[float] = []
    recent_action_keys: List[str] = []
    step_count = 0
    success = False

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")

    try:
        with IncidentEnv(base_url=env_base_url).sync() as env:
            # Reset environment
            reset_result = env.reset(task_name=task_name)
            obs = reset_result.observation
            alert = obs.alert_summary or "Incident alert fired"
            services = obs.available_services or []

            system_prompt = get_system_prompt(task_name)
            services_str = ", ".join(services) if services else ", ".join(TASK_CONTEXT.get(task_name, {}).get("services", []))

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"ALERT: {alert}\n"
                        f"Available services: {services_str}\n"
                        f"Begin investigating. What is your first action?"
                    ),
                },
            ]

            for step_idx in range(MAX_STEPS_PER_TASK):
                step_count = step_idx + 1
                action_str = "noop()"
                reward = 0.01
                done = False
                error_str = "null"
                result_text = ""

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    llm_output = response.choices[0].message.content or ""

                    action = parse_action(llm_output)
                    if not action:
                        action = {
                            "action_type": "query_logs",
                            "target": services[0] if services else "payment-service",
                            "parameters": {"filter": "error"},
                        }
                        error_str = "Could not parse action from LLM output; used safe fallback"

                    action = sanitize_action(task_name, action, recent_action_keys)

                    action_str = format_action_str(action)
                    recent_action_keys.append(
                        f"{action['action_type']}:{action['target']}:{json.dumps(action['parameters'], sort_keys=True)}"
                    )

                    step_result = env.step(
                        IncidentAction(
                            action_type=action.get("action_type", "noop"),
                            target=action.get("target", ""),
                            parameters=action.get("parameters", {}),
                        )
                    )

                    reward = clamp_reward(step_result.reward)
                    done = step_result.done
                    step_obs = step_result.observation
                    result_text = step_obs.last_action_result or ""
                    is_error = bool(step_obs.last_action_error)

                    if is_error:
                        error_str = result_text[:200]
                    else:
                        error_str = "null"

                    if "✅" in result_text and "resolved" in result_text.lower():
                        success = True
                    elif "📞" in result_text and "escalated" in result_text.lower():
                        pass

                except Exception as e:
                    error_str = str(e).replace("\n", " ")[:200]
                    reward = 0.01

                rewards.append(clamp_reward(reward))
                done_str = "true" if done else "false"
                print(f"[STEP] step={step_count} action={action_str} reward={clamp_reward(reward):.2f} done={done_str} error={error_str}")

                if done:
                    break

                messages.append({"role": "assistant", "content": llm_output})
                truncated = result_text[:1500] if len(result_text) > 1500 else result_text
                messages.append({
                    "role": "user",
                    "content": f"Action result:\n{truncated}\n\nWhat is your next action? Respond with JSON only.",
                })

    except Exception as e:
        if step_count == 0:
            step_count = 1
            rewards.append(0.01)
            print(f"[STEP] step=1 action=noop() reward=0.01 done=false error={str(e)[:200]}")

    score = clamp_reward(sum(rewards) / len(rewards)) if rewards else 0.01
    return success, step_count, score, rewards


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    target_task = os.getenv("TASK_NAME")
    tasks_to_run = [target_task] if target_task else TASKS

    for task in tasks_to_run:
        success, steps, score, rewards = run_task(client, ENV_BASE_URL, task)
        rewards_str = ",".join(f"{clamp_reward(r):.2f}" for r in rewards) if rewards else "0.01"
        success_str = "true" if success else "false"
        print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}")


if __name__ == "__main__":
    main()
