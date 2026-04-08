"""Incident Response Environment — Environment base class implementation."""

import json
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import IncidentAction, IncidentObservation, IncidentState
except ImportError:
    from models import IncidentAction, IncidentObservation, IncidentState

from .graders import (
    EpisodeHistory,
    check_remedy,
    check_root_cause,
    get_grade_breakdown,
    grade_episode,
)
from .scenarios import Scenario, get_scenario


class IncidentEnvironment(Environment):
    """On-call incident response environment with 3 difficulty levels.

    The agent investigates production incidents using logs, metrics, and
    runbooks, then identifies the root cause and applies a remedy.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    VALID_ACTIONS = [
        "query_logs", "check_metrics", "read_runbook",
        "identify_root_cause", "execute_remedy", "escalate",
        "get_status", "communicate_status", "noop",
    ]

    DESTRUCTIVE_ACTIONS = {
        "drop_database", "delete_data", "format_disk",
        "shutdown_service", "terminate_all",
    }

    RISKY_REMEDY_KEYWORDS = {"restart", "kill", "scale", "flush", "force"}

    MIN_REWARD = 0.01
    MAX_REWARD = 0.95

    def __init__(self):
        self._scenario: Optional[Scenario] = None
        self._history: Optional[EpisodeHistory] = None
        self._findings: list = []
        self._done: bool = False
        self._step_count: int = 0
        self._time_elapsed: float = 0.0
        self._prev_clue_count: int = 0
        self._prev_root_cause_state: bool = False
        self._prev_wrong_remedies: int = 0
        self._prev_destructive: int = 0
        self._prev_escalations: int = 0
        self._trajectory_reward_sum: float = self.MIN_REWARD
        self._scenario_seed: int = 42
        self._scenario_variant: str = "v1"
        self._last_grade_breakdown: dict = {}
        self._state = IncidentState(
            episode_id=str(uuid4()),
            step_count=0,
            cum_reward=self.MIN_REWARD,
        )

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> IncidentObservation:
        task_name = kwargs.get("task_name", "easy_config_error")
        requested_seed = seed if seed is not None else kwargs.get("seed", 42)
        try:
            seed_value = int(requested_seed)
        except (TypeError, ValueError):
            seed_value = 42

        self._scenario_seed = seed_value
        self._scenario_variant = f"v{(abs(self._scenario_seed) % 3) + 1}"
        self._scenario = get_scenario(task_name, seed=seed_value)
        self._history = EpisodeHistory()
        self._findings = []
        self._done = False
        self._step_count = 0
        self._time_elapsed = 0.0
        self._prev_clue_count = 0
        self._prev_root_cause_state = False
        self._prev_wrong_remedies = 0
        self._prev_destructive = 0
        self._prev_escalations = 0
        self._trajectory_reward_sum = self.MIN_REWARD
        self._last_grade_breakdown = {}

        self._state = IncidentState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=task_name,
            task_difficulty=self._scenario.task_difficulty,
            severity=self._scenario.severity,
            cum_reward=self.MIN_REWARD,
            scenario_variant=self._scenario_variant,
        )

        return IncidentObservation(
            alert_summary=self._scenario.alert_summary,
            severity=self._scenario.severity,
            task_description=self._scenario.task_description,
            current_findings=[],
            available_services=list(self._scenario.services.keys()),
            available_actions=self.VALID_ACTIONS,
            last_action_result="Environment ready. Begin investigation.",
            last_action_error=False,
            time_elapsed_minutes=0.0,
            step_number=0,
            max_steps=self._scenario.max_steps,
            done=False,
            reward=self.MIN_REWARD,
            final_score=self.MIN_REWARD,
            score_breakdown={},
            scenario_variant=self._scenario_variant,
        )

    def step(self, action: IncidentAction) -> IncidentObservation:  # type: ignore[override]
        if not self._scenario or not self._history:
            return IncidentObservation(
                last_action_result="Error: call reset() before step().",
                last_action_error=True,
                done=False,
                reward=self.MIN_REWARD,
            )

        self._step_count += 1
        self._state.step_count = self._step_count
        self._time_elapsed += 2.0

        action_type = action.action_type.lower().strip()
        result_text, is_error = self._dispatch_action(action_type, action)

        self._history.steps_used = self._step_count
        is_done = self._done or self._step_count >= self._scenario.max_steps
        score_breakdown = {}

        if is_done:
            # Convert final grade to a terminal delta reward so the episode return
            # remains strictly below 1.0 even after incremental step rewards.
            score_breakdown = get_grade_breakdown(self._scenario, self._history)
            target_final_grade = float(score_breakdown.get("total", grade_episode(self._scenario, self._history)))
            terminal_delta = target_final_grade - self._trajectory_reward_sum
            reward = self._clamp_open_reward(terminal_delta)
            remaining_budget = round(max(self.MIN_REWARD, self.MAX_REWARD - self._trajectory_reward_sum), 4)
            reward = round(min(reward, remaining_budget), 4)
            self._last_grade_breakdown = score_breakdown
            result_text = f"{result_text}\n\n{self._format_score_breakdown(score_breakdown)}"
        else:
            reward = self._compute_step_reward()

        reward = self._clamp_open_reward(reward)
        self._trajectory_reward_sum = round(self._trajectory_reward_sum + reward, 4)

        self._state.root_cause_identified = self._history.root_cause_correct
        self._state.incident_resolved = self._history.remedy_correct
        self._state.cum_reward = self._trajectory_reward_sum
        self._state.relevant_clues_found = self._history.relevant_clues_found
        self._state.total_clues_available = self._scenario.total_clues
        self._state.steps_used = self._step_count
        self._state.wrong_actions_taken = self._history.wrong_remedies + self._history.destructive_actions
        self._state.scenario_variant = self._scenario_variant

        final_score = (
            float(score_breakdown.get("total", self._trajectory_reward_sum))
            if is_done
            else float(grade_episode(self._scenario, self._history))
        )

        return IncidentObservation(
            alert_summary=self._scenario.alert_summary,
            severity=self._scenario.severity,
            task_description=self._scenario.task_description,
            current_findings=list(self._findings),
            available_services=list(self._scenario.services.keys()),
            available_actions=self.VALID_ACTIONS,
            last_action_result=result_text,
            last_action_error=is_error,
            time_elapsed_minutes=self._time_elapsed,
            step_number=self._step_count,
            max_steps=self._scenario.max_steps,
            done=is_done,
            reward=reward,
            final_score=round(final_score, 4),
            score_breakdown=score_breakdown,
            scenario_variant=self._scenario_variant,
        )

    @property
    def state(self) -> IncidentState:
        return self._state

    # -- Action dispatch --

    def _dispatch_action(self, action_type: str, action: IncidentAction) -> tuple:
        """Route action to handler. Returns (result_text, is_error)."""
        target = action.target or ""
        params = action.parameters or {}

        if action_type == "query_logs":
            return self._handle_query_logs(target, params.get("filter", "")), False
        elif action_type == "check_metrics":
            return self._handle_check_metrics(target), False
        elif action_type == "read_runbook":
            return self._handle_read_runbook(target), False
        elif action_type == "identify_root_cause":
            cause = params.get("cause", target)
            return self._handle_identify_root_cause(cause), False
        elif action_type == "execute_remedy":
            remedy = params.get("remedy", "")
            service = params.get("service", target)
            return self._handle_execute_remedy(service, remedy), False
        elif action_type == "escalate":
            reason = params.get("reason", target)
            return self._handle_escalate(reason), False
        elif action_type == "get_status":
            return self._handle_get_status(), False
        elif action_type == "communicate_status":
            message = params.get("message", target)
            return self._handle_communicate_status(message), False
        elif action_type == "noop":
            return self._handle_noop(), False
        else:
            self._history.invalid_actions += 1
            return f"Unknown action '{action_type}'. Valid: {', '.join(self.VALID_ACTIONS)}", True

    # -- Tool handlers --

    def _handle_query_logs(self, service: str, filter_text: str = "") -> str:
        service_lower = service.lower().strip()
        service_info = self._scenario.services.get(service_lower)

        if not service_info:
            for name, info in self._scenario.services.items():
                if service_lower in name or name in service_lower:
                    service_info = info
                    break

        if not service_info:
            available = ", ".join(self._scenario.services.keys())
            return f"Error: Unknown service '{service}'. Available: {available}"

        self._history.services_queried_logs.add(service_info.name)

        logs = service_info.log_lines
        if filter_text:
            logs = [line for line in logs if filter_text.lower() in line.lower()]

        if not logs:
            return f"No log entries found for {service_info.name}" + (
                f" matching filter '{filter_text}'" if filter_text else ""
            )

        if service_info.is_relevant:
            self._history.relevant_clues_found = min(
                self._history.relevant_clues_found + 1,
                self._scenario.total_clues,
            )

        result = f"=== Logs for {service_info.name} ===\n" + "\n".join(logs)
        self._findings.append(f"Queried logs for {service_info.name}")
        return result

    def _handle_check_metrics(self, service: str) -> str:
        service_lower = service.lower().strip()
        service_info = self._scenario.services.get(service_lower)

        if not service_info:
            for name, info in self._scenario.services.items():
                if service_lower in name or name in service_lower:
                    service_info = info
                    break

        if not service_info:
            available = ", ".join(self._scenario.services.keys())
            return f"Error: Unknown service '{service}'. Available: {available}"

        self._history.services_queried_metrics.add(service_info.name)

        if service_info.is_relevant:
            self._history.relevant_clues_found = min(
                self._history.relevant_clues_found + 1,
                self._scenario.total_clues,
            )

        result = f"=== Metrics for {service_info.name} ===\n{json.dumps(service_info.metrics, indent=2)}"
        self._findings.append(f"Checked metrics for {service_info.name}")
        return result

    def _handle_read_runbook(self, service: str) -> str:
        service_lower = service.lower().strip()
        service_info = self._scenario.services.get(service_lower)

        if not service_info:
            for name, info in self._scenario.services.items():
                if service_lower in name or name in service_lower:
                    service_info = info
                    break

        if not service_info:
            available = ", ".join(self._scenario.services.keys())
            return f"Error: Unknown service '{service}'. Available: {available}"

        self._history.services_queried_runbook.add(service_info.name)

        if service_info.is_relevant:
            self._history.relevant_clues_found = min(
                self._history.relevant_clues_found + 1,
                self._scenario.total_clues,
            )

        if not service_info.runbook_entry:
            return f"No runbook entry found for {service_info.name}"

        self._findings.append(f"Read runbook for {service_info.name}")
        return service_info.runbook_entry

    def _handle_identify_root_cause(self, cause: str) -> str:
        self._history.root_cause_declared = cause
        exact, partial = check_root_cause(cause, self._scenario)
        self._history.root_cause_correct = exact
        self._history.root_cause_partial = partial

        # Evidence quality: root-cause claims are stronger when backed by both logs and
        # metrics from relevant services.
        relevant_seen_in_logs = len(self._history.services_queried_logs.intersection(self._scenario.relevant_services)) > 0
        relevant_seen_in_metrics = len(self._history.services_queried_metrics.intersection(self._scenario.relevant_services)) > 0
        self._history.evidence_supported_root_cause = (relevant_seen_in_logs and relevant_seen_in_metrics and (exact or partial))

        if exact:
            return "✅ Root cause correctly identified! Now apply the appropriate remedy."
        elif partial:
            return "⚠️ You're on the right track, but not complete. Gather more evidence."
        else:
            return "❌ Incorrect root cause. Continue investigating logs, metrics, and runbooks."

    def _handle_execute_remedy(self, service: str, remedy: str) -> str:
        attempt = {"service": service.lower().strip(), "remedy": remedy.lower().strip()}
        self._history.remedies_attempted.append(attempt)

        if remedy.lower().strip() in self.DESTRUCTIVE_ACTIONS:
            self._history.destructive_actions += 1
            return f"⛔ DESTRUCTIVE ACTION BLOCKED: '{remedy}' on {service}. Penalty applied."

        is_correct = check_remedy([attempt], self._scenario)

        # Risk-aware mechanic: broad/risky remediations before identifying root-cause
        # are penalized even if they may accidentally work.
        if not self._history.root_cause_correct:
            if any(k in attempt["remedy"] for k in self.RISKY_REMEDY_KEYWORDS):
                self._history.risky_actions += 1

        if is_correct:
            self._history.remedy_correct = True
            valid_services = {v["service"].lower().strip() for v in self._scenario.valid_remedies}
            has_post_fix_signal = (
                len(self._history.services_queried_metrics.intersection(valid_services)) > 0
                or self._history.status_checks > 0
            )
            self._history.verification_completed = has_post_fix_signal
            self._done = True
            return f"✅ Remedy '{remedy}' applied to {service}. Incident resolved!"
        else:
            self._history.wrong_remedies += 1
            return f"❌ Remedy '{remedy}' on {service} failed. Check the runbook for the correct procedure."

    def _handle_escalate(self, reason: str) -> str:
        self._history.unnecessary_escalations += 1
        self._done = True
        return f"📞 Escalated to senior on-call. Reason: {reason}. Resolving without escalation earns higher scores."

    def _handle_get_status(self) -> str:
        self._history.status_checks += 1
        score_preview = grade_episode(self._scenario, self._history)
        status = {
            "alert": self._scenario.alert_summary,
            "severity": self._scenario.severity,
            "time_elapsed_minutes": self._time_elapsed,
            "steps_used": self._step_count,
            "max_steps": self._scenario.max_steps,
            "steps_remaining": self._scenario.max_steps - self._step_count,
            "findings_count": len(self._findings),
            "recent_findings": self._findings[-5:] if self._findings else [],
            "root_cause_declared": self._history.root_cause_declared is not None,
            "incident_resolved": self._history.remedy_correct,
            "available_services": list(self._scenario.services.keys()),
            "communication_updates": self._history.communication_updates,
            "verification_completed": self._history.verification_completed,
            "scenario_variant": self._scenario_variant,
            "score_preview": score_preview,
            "score_breakdown": self._last_grade_breakdown if self._done else None,
        }
        return json.dumps(status, indent=2)

    def _handle_communicate_status(self, message: str) -> str:
        clean_message = (message or "").strip()
        self._history.communication_updates += 1
        if not clean_message:
            return "📣 Status update sent: Investigating incident and collecting evidence."
        return f"📣 Status update sent: {clean_message[:200]}"

    def _handle_noop(self) -> str:
        self._history.noop_count += 1
        return "No-op recorded. Continue investigating to improve score."

    @staticmethod
    def _format_score_breakdown(score_breakdown: dict) -> str:
        if not score_breakdown:
            return "Score breakdown unavailable"

        ordered_keys = [
            "root_cause",
            "root_cause_evidence",
            "remedy",
            "verification",
            "communication",
            "efficiency",
            "clue_discovery",
            "penalty_wrong_remedy",
            "penalty_destructive",
            "penalty_escalation",
            "penalty_risky_action",
            "penalty_noop",
            "total",
        ]
        filtered = {k: score_breakdown[k] for k in ordered_keys if k in score_breakdown}
        return "Score breakdown:\n" + json.dumps(filtered, indent=2)

    # -- Incremental reward --

    @staticmethod
    def _clamp_open_reward(value: float) -> float:
        return round(max(IncidentEnvironment.MIN_REWARD, min(IncidentEnvironment.MAX_REWARD, value)), 4)

    def _compute_step_reward(self) -> float:
        if not self._history:
            return self.MIN_REWARD

        reward = 0.0

        new_clues = self._history.relevant_clues_found - self._prev_clue_count
        if new_clues > 0:
            reward += 0.02 * new_clues
            self._prev_clue_count = self._history.relevant_clues_found

        if self._history.root_cause_correct and not self._prev_root_cause_state:
            reward += 0.05
            self._prev_root_cause_state = True

        new_wrong = self._history.wrong_remedies - self._prev_wrong_remedies
        if new_wrong > 0:
            reward -= 0.03 * new_wrong
            self._prev_wrong_remedies = self._history.wrong_remedies

        new_destructive = self._history.destructive_actions - self._prev_destructive
        if new_destructive > 0:
            reward -= 0.05 * new_destructive
            self._prev_destructive = self._history.destructive_actions

        new_escalations = self._history.unnecessary_escalations - self._prev_escalations
        if new_escalations > 0:
            reward -= 0.02 * new_escalations
            self._prev_escalations = self._history.unnecessary_escalations

        if self._history.noop_count > 0:
            reward -= 0.01

        return self._clamp_open_reward(reward)
