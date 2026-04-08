"""Data models for the Incident Response Environment."""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class IncidentAction(Action):
    """Action taken by the on-call agent."""

    action_type: str = Field(
        ...,
        description="One of: query_logs, check_metrics, read_runbook, identify_root_cause, execute_remedy, escalate, get_status, communicate_status, noop",
    )
    target: str = Field(default="", description="Target service or component")
    parameters: Dict[str, str] = Field(default_factory=dict, description="Action-specific parameters")


class IncidentObservation(Observation):
    """Observation returned to the agent after each action."""

    alert_summary: str = Field(default="", description="Initial alert text")
    severity: str = Field(default="P3", description="Severity: P1, P2, P3")
    task_description: str = Field(default="", description="Task objective")

    current_findings: List[str] = Field(default_factory=list, description="Discovered clues")
    available_services: List[str] = Field(default_factory=list, description="Queryable service names")
    available_actions: List[str] = Field(default_factory=list, description="Valid action types")

    last_action_result: str = Field(default="", description="Result of last action")
    last_action_error: bool = Field(default=False, description="Whether last action errored")

    time_elapsed_minutes: float = Field(default=0.0, description="Simulated time elapsed")
    step_number: int = Field(default=0, description="Current step")
    max_steps: int = Field(default=15, description="Maximum allowed steps")
    final_score: float = Field(default=0.0, description="Current or final deterministic task score")
    score_breakdown: Dict[str, float] = Field(default_factory=dict, description="Deterministic grading breakdown")
    scenario_variant: str = Field(default="", description="Deterministic scenario variant label")


class IncidentState(State):
    """Environment state metadata."""

    task_name: str = Field(default="", description="Current task identifier")
    task_difficulty: str = Field(default="", description="easy, medium, or hard")
    severity: str = Field(default="P3", description="Incident severity")

    root_cause_identified: bool = Field(default=False)
    incident_resolved: bool = Field(default=False)
    cum_reward: float = Field(default=0.0, description="Cumulative reward")
    relevant_clues_found: int = Field(default=0)
    total_clues_available: int = Field(default=0)
    steps_used: int = Field(default=0)
    wrong_actions_taken: int = Field(default=0)
    scenario_variant: str = Field(default="")
