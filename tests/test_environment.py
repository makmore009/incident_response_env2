"""Basic tests for the Incident Response Environment."""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.scenarios import get_scenario, list_tasks, Scenario
from server.graders import EpisodeHistory, grade_episode, check_root_cause, check_remedy
from server.incident_environment import IncidentEnvironment
from models import IncidentAction


def test_list_tasks():
    """Test that all 3 tasks are available."""
    tasks = list_tasks()
    assert len(tasks) == 3
    assert "easy_config_error" in tasks
    assert "medium_cascading_db" in tasks
    assert "hard_intermittent_auth" in tasks
    print("✅ test_list_tasks passed")


def test_scenario_loading():
    """Test that all scenarios load correctly."""
    for task_name in list_tasks():
        scenario = get_scenario(task_name)
        assert isinstance(scenario, Scenario)
        assert scenario.task_name == task_name
        assert len(scenario.services) >= 3
        assert scenario.root_cause != ""
        assert len(scenario.valid_remedies) >= 1
        assert scenario.max_steps > 0
        assert scenario.total_clues > 0
        print(f"  ✅ {task_name}: {len(scenario.services)} services, {scenario.total_clues} clues")
    print("✅ test_scenario_loading passed")


def test_scenario_seed_variants_are_stable_and_distinct():
    """Seeded scenario variants should be deterministic and visibly distinct."""
    s1 = get_scenario("easy_config_error", seed=42)
    s2 = get_scenario("easy_config_error", seed=42)
    s3 = get_scenario("easy_config_error", seed=43)

    assert s1.alert_summary == s2.alert_summary, "Same seed should produce same variant"
    assert s1.alert_summary != s3.alert_summary, "Different seeds should produce different variants"

    print("✅ test_scenario_seed_variants_are_stable_and_distinct passed")


def test_root_cause_check_easy():
    """Test root cause matching for easy scenario."""
    scenario = get_scenario("easy_config_error")

    # Exact match
    exact, partial = check_root_cause(
        "The STRIPE_API_KEY environment variable was misconfigured with an invalid config",
        scenario,
    )
    assert exact, "Should be exact match with multiple keywords"

    # Partial match
    exact, partial = check_root_cause("Something is wrong with the config", scenario)
    assert partial, "Should be partial match with 'config' keyword"

    # No match
    exact, partial = check_root_cause("The database is down", scenario)
    assert not exact and not partial, "Should not match"

    print("✅ test_root_cause_check_easy passed")


def test_remedy_check_easy():
    """Test remedy matching for easy scenario."""
    scenario = get_scenario("easy_config_error")

    # Correct remedy
    attempts = [{"service": "payment-service", "remedy": "rollback_config"}]
    assert check_remedy(attempts, scenario), "Should match valid remedy"

    # Wrong service
    attempts = [{"service": "database", "remedy": "rollback_config"}]
    assert not check_remedy(attempts, scenario), "Wrong service should not match"

    # Wrong remedy
    attempts = [{"service": "payment-service", "remedy": "restart_service"}]
    assert not check_remedy(attempts, scenario), "Wrong remedy should not match"

    print("✅ test_remedy_check_easy passed")


def test_grading_perfect_score():
    """Test that a perfect episode gets a high score."""
    scenario = get_scenario("easy_config_error")

    history = EpisodeHistory(
        root_cause_correct=True,
        root_cause_partial=True,
        remedy_correct=True,
        evidence_supported_root_cause=True,
        verification_completed=True,
        communication_updates=1,
        relevant_clues_found=3,
        steps_used=4,  # Very efficient
    )

    score = grade_episode(scenario, history)
    assert 0.9 <= score <= 1.0, f"Perfect episode should score ~1.0, got {score}"
    print(f"✅ test_grading_perfect_score passed (score: {score})")


def test_grading_no_progress():
    """Test that an empty episode gets a low score."""
    scenario = get_scenario("easy_config_error")

    history = EpisodeHistory(
        steps_used=10,
    )

    score = grade_episode(scenario, history)
    assert score <= 0.1, f"Empty episode should score low, got {score}"
    print(f"✅ test_grading_no_progress passed (score: {score})")


def test_grading_penalties():
    """Test that penalties reduce score."""
    scenario = get_scenario("easy_config_error")

    # Good investigation but wrong remedy
    history = EpisodeHistory(
        root_cause_correct=True,
        remedy_correct=False,
        wrong_remedies=2,
        relevant_clues_found=3,
        steps_used=8,
    )

    score = grade_episode(scenario, history)
    assert score < 0.5, f"Wrong remedies should reduce score significantly, got {score}"
    print(f"✅ test_grading_penalties passed (score: {score})")


def test_grading_difficulty_range():
    """Test that grading works for all 3 difficulty levels."""
    for task_name in list_tasks():
        scenario = get_scenario(task_name)

        # Perfect run
        perfect = EpisodeHistory(
            root_cause_correct=True,
            root_cause_partial=True,
            remedy_correct=True,
            evidence_supported_root_cause=True,
            verification_completed=True,
            communication_updates=1,
            relevant_clues_found=scenario.total_clues,
            steps_used=max(1, scenario.max_steps // 3),
        )
        perfect_score = grade_episode(scenario, perfect)
        assert 0.9 <= perfect_score <= 1.0, f"{task_name} perfect score: {perfect_score}"

        # Failed run
        failed = EpisodeHistory(steps_used=scenario.max_steps)
        failed_score = grade_episode(scenario, failed)
        assert failed_score < 0.1, f"{task_name} failed score: {failed_score}"

        print(f"  ✅ {task_name}: perfect={perfect_score:.2f}, failed={failed_score:.2f}")

    print("✅ test_grading_difficulty_range passed")


def test_score_range():
    """Test that all scores are strictly in (0.0, 1.0)."""
    for task_name in list_tasks():
        scenario = get_scenario(task_name)

        # Test various episode configurations
        configs = [
            EpisodeHistory(),
            EpisodeHistory(root_cause_correct=True, remedy_correct=True, steps_used=1, relevant_clues_found=10),
            EpisodeHistory(wrong_remedies=5, destructive_actions=3, unnecessary_escalations=2, steps_used=20),
        ]

        for history in configs:
            score = grade_episode(scenario, history)
            assert 0.0 < score < 1.0, f"Score {score} out of strict (0.0, 1.0) range"

    print("✅ test_score_range passed")


def test_environment_runtime_rewards_strict_open_interval():
    """Test reset/step runtime rewards are strictly in (0.0, 1.0)."""
    env = IncidentEnvironment()

    obs = env.reset(task_name="easy_config_error")
    assert 0.0 < obs.reward < 1.0, f"Reset reward {obs.reward} not in strict (0.0, 1.0)"
    assert 0.0 < env.state.cum_reward < 1.0, f"Reset state.cum_reward {env.state.cum_reward} not in strict (0.0, 1.0)"

    obs = env.step(IncidentAction(action_type="get_status", target="", parameters={}))
    assert 0.0 < obs.reward < 1.0, f"Neutral step reward {obs.reward} not in strict (0.0, 1.0)"

    obs = env.step(
        IncidentAction(
            action_type="execute_remedy",
            target="",
            parameters={"service": "payment-service", "remedy": "restart_service"},
        )
    )
    assert 0.0 < obs.reward < 1.0, f"Penalty step reward {obs.reward} not in strict (0.0, 1.0)"

    print("✅ test_environment_runtime_rewards_strict_open_interval passed")


def test_episode_return_strict_open_interval():
    """Test that cumulative episode return remains strictly in (0.0, 1.0)."""
    env = IncidentEnvironment()
    rewards = []

    env.reset(task_name="easy_config_error")

    obs = env.step(
        IncidentAction(
            action_type="query_logs",
            target="payment-service",
            parameters={"filter": "error"},
        )
    )
    rewards.append(obs.reward)

    obs = env.step(IncidentAction(action_type="check_metrics", target="payment-service", parameters={}))
    rewards.append(obs.reward)

    obs = env.step(IncidentAction(action_type="read_runbook", target="payment-service", parameters={}))
    rewards.append(obs.reward)

    obs = env.step(
        IncidentAction(
            action_type="identify_root_cause",
            target="",
            parameters={"cause": "misconfigured STRIPE_API_KEY environment variable"},
        )
    )
    rewards.append(obs.reward)

    obs = env.step(
        IncidentAction(
            action_type="execute_remedy",
            target="",
            parameters={"service": "payment-service", "remedy": "rollback_config"},
        )
    )
    rewards.append(obs.reward)
    assert obs.done, "Expected episode to finish after correct remedy"

    total = round(sum(rewards), 4)
    assert 0.0 < total < 1.0, f"Episode return {total} not in strict (0.0, 1.0)"

    print("✅ test_episode_return_strict_open_interval passed")


def test_episode_aggregation_formulas_strict_open_interval():
    """Test multiple plausible evaluator aggregations remain strictly in (0.0, 1.0)."""
    env = IncidentEnvironment()
    reset_obs = env.reset(task_name="easy_config_error")
    reset_reward = reset_obs.reward

    rewards = []

    rewards.append(
        env.step(
            IncidentAction(
                action_type="query_logs",
                target="payment-service",
                parameters={"filter": "error"},
            )
        ).reward
    )
    rewards.append(env.step(IncidentAction(action_type="check_metrics", target="payment-service", parameters={})).reward)
    rewards.append(env.step(IncidentAction(action_type="read_runbook", target="payment-service", parameters={})).reward)
    rewards.append(
        env.step(
            IncidentAction(
                action_type="identify_root_cause",
                target="",
                parameters={"cause": "misconfigured STRIPE_API_KEY environment variable"},
            )
        ).reward
    )
    final_obs = env.step(
        IncidentAction(
            action_type="execute_remedy",
            target="",
            parameters={"service": "payment-service", "remedy": "rollback_config"},
        )
    )
    rewards.append(final_obs.reward)

    assert final_obs.done, "Expected episode to be done after valid remedy"

    sum_steps = round(sum(rewards), 4)
    sum_with_reset = round(reset_reward + sum_steps, 4)
    mean_steps = round(sum_steps / len(rewards), 4)
    terminal_only = round(final_obs.reward, 4)

    assert 0.0 < sum_steps < 1.0, f"sum_steps {sum_steps} out of (0,1)"
    assert 0.0 < sum_with_reset < 1.0, f"sum_with_reset {sum_with_reset} out of (0,1)"
    assert 0.0 < mean_steps < 1.0, f"mean_steps {mean_steps} out of (0,1)"
    assert 0.0 < terminal_only < 1.0, f"terminal_only {terminal_only} out of (0,1)"
    assert 0.0 < env.state.cum_reward < 1.0, f"final state.cum_reward {env.state.cum_reward} out of (0,1)"

    print("✅ test_episode_aggregation_formulas_strict_open_interval passed")


def test_runtime_communication_and_noop_actions():
    """Test new communication/noop actions and status accounting."""
    env = IncidentEnvironment()
    env.reset(task_name="easy_config_error")

    obs = env.step(
        IncidentAction(
            action_type="communicate_status",
            target="",
            parameters={"message": "Investigating payment-service and collecting clues."},
        )
    )
    assert not obs.last_action_error, "communicate_status should be accepted"
    assert "Status update sent" in obs.last_action_result

    obs = env.step(IncidentAction(action_type="noop", target="", parameters={}))
    assert not obs.last_action_error, "noop should be accepted"

    status_obs = env.step(IncidentAction(action_type="get_status", target="", parameters={}))
    status = json.loads(status_obs.last_action_result)
    assert status["communication_updates"] >= 1

    print("✅ test_runtime_communication_and_noop_actions passed")


def test_runtime_verification_signal_after_correct_remedy():
    """Test that verification_completed is tracked for successful incidents."""
    env = IncidentEnvironment()
    env.reset(task_name="easy_config_error")

    env.step(IncidentAction(action_type="query_logs", target="payment-service", parameters={"filter": "error"}))
    env.step(IncidentAction(action_type="check_metrics", target="payment-service", parameters={}))
    env.step(IncidentAction(action_type="get_status", target="", parameters={}))
    env.step(
        IncidentAction(
            action_type="identify_root_cause",
            target="",
            parameters={"cause": "misconfigured STRIPE_API_KEY environment variable"},
        )
    )
    final_obs = env.step(
        IncidentAction(
            action_type="execute_remedy",
            target="",
            parameters={"service": "payment-service", "remedy": "rollback_config"},
        )
    )

    assert final_obs.done, "Episode should complete after correct remedy"
    assert env._history.verification_completed is True, "verification_completed should be true after post-fix checks"

    print("✅ test_runtime_verification_signal_after_correct_remedy passed")


def test_terminal_observation_contains_score_breakdown():
    """Terminal observation should contain a deterministic grading breakdown."""
    env = IncidentEnvironment()
    env.reset(task_name="easy_config_error", seed=42)

    env.step(IncidentAction(action_type="query_logs", target="payment-service", parameters={"filter": "error"}))
    env.step(IncidentAction(action_type="check_metrics", target="payment-service", parameters={}))
    env.step(
        IncidentAction(
            action_type="identify_root_cause",
            target="",
            parameters={"cause": "misconfigured STRIPE_API_KEY environment variable"},
        )
    )
    final_obs = env.step(
        IncidentAction(
            action_type="execute_remedy",
            target="",
            parameters={"service": "payment-service", "remedy": "rollback_config"},
        )
    )

    assert final_obs.done, "Expected episode termination"
    assert final_obs.score_breakdown, "Terminal score breakdown should be present"
    assert "total" in final_obs.score_breakdown, "Terminal breakdown should include total"

    print("✅ test_terminal_observation_contains_score_breakdown passed")


def test_exploit_noop_spam_scores_low():
    """No-op spamming should converge to a low terminal score."""
    env = IncidentEnvironment()
    env.reset(task_name="easy_config_error")

    obs = None
    for _ in range(12):
        obs = env.step(IncidentAction(action_type="noop", target="", parameters={}))
        if obs.done:
            break

    assert obs is not None and obs.done, "Episode should eventually terminate"
    assert obs.final_score < 0.2, f"No-op spam should score low, got {obs.final_score}"

    print("✅ test_exploit_noop_spam_scores_low passed")


def test_exploit_escalate_first_scores_low():
    """Immediate escalation should be allowed but heavily penalized."""
    env = IncidentEnvironment()
    env.reset(task_name="medium_cascading_db")

    obs = env.step(
        IncidentAction(
            action_type="escalate",
            target="",
            parameters={"reason": "Escalating before diagnosis"},
        )
    )

    assert obs.done, "Escalation should end episode"
    assert obs.final_score < 0.2, f"Escalate-first should score low, got {obs.final_score}"

    print("✅ test_exploit_escalate_first_scores_low passed")


def test_exploit_wrong_remedy_spam_scores_low():
    """Repeated risky wrong remedies should accrue penalties and low terminal score."""
    env = IncidentEnvironment()
    env.reset(task_name="medium_cascading_db")

    obs = None
    for _ in range(6):
        obs = env.step(
            IncidentAction(
                action_type="execute_remedy",
                target="",
                parameters={"service": "db-primary", "remedy": "restart_service"},
            )
        )
        if obs.done:
            break

    while obs is not None and not obs.done:
        obs = env.step(IncidentAction(action_type="noop", target="", parameters={}))

    assert obs is not None and obs.done, "Episode should terminate within max steps"
    assert env._history.risky_actions > 0, "Risky action counter should increase"
    assert obs.final_score < 0.25, f"Wrong-remedy spam should score low, got {obs.final_score}"

    print("✅ test_exploit_wrong_remedy_spam_scores_low passed")


if __name__ == "__main__":
    print("\n🧪 Running Incident Response Environment Tests\n")
    test_list_tasks()
    test_scenario_loading()
    test_scenario_seed_variants_are_stable_and_distinct()
    test_root_cause_check_easy()
    test_remedy_check_easy()
    test_grading_perfect_score()
    test_grading_no_progress()
    test_grading_penalties()
    test_grading_difficulty_range()
    test_score_range()
    test_environment_runtime_rewards_strict_open_interval()
    test_episode_return_strict_open_interval()
    test_episode_aggregation_formulas_strict_open_interval()
    test_runtime_communication_and_noop_actions()
    test_runtime_verification_signal_after_correct_remedy()
    test_terminal_observation_contains_score_breakdown()
    test_exploit_noop_spam_scores_low()
    test_exploit_escalate_first_scores_low()
    test_exploit_wrong_remedy_spam_scores_low()
    print("\n✅ All tests passed!\n")
