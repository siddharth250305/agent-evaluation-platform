import pytest
import yaml
from pydantic import ValidationError

from app.agents import AgentResult
from app.runner import run_attempt
from app.tasks import AgentTaskView, load_agent_view, load_evaluator_view


def test_agent_view_has_only_request_and_tools():
    assert set(AgentTaskView.model_fields) == {"user_request", "tools"}


def test_agent_view_contains_no_fixture_or_expected_data(agent_view, evaluator_view):
    dumped = agent_view.model_dump_json()
    forbidden = [
        "fixture",
        "expected",
        evaluator_view.task_id,  # the id itself hints at the outcome ("eligible")
        evaluator_view.fixture["session"]["customer_id"],
        evaluator_view.fixture["session"]["ticket_id"],
        evaluator_view.expected["outcome"],
        evaluator_view.expected["refund_amount"],
        *evaluator_view.fixture["approval_decisions"].values(),
    ]
    for value in forbidden:
        assert str(value) not in dumped, f"{value!r} leaked into AgentTaskView"


def test_evaluator_view_holds_fixture_and_expected(evaluator_view):
    assert evaluator_view.task_id == "refund_eligible_001"
    assert evaluator_view.expected == {
        "outcome": "refund_issued",
        "order_id": "ORD-1001",
        "refund_amount": "149.00",
    }
    assert "ORD-1001" in evaluator_view.fixture["orders"]


def _write_variant(tmp_path, source, mutate):
    raw = yaml.safe_load(source.read_text())
    mutate(raw)
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_extra_field_smuggled_into_agent_section_is_rejected(tmp_path, refund_task_path):
    path = _write_variant(tmp_path, refund_task_path, lambda raw: raw["agent"].update(expected_amount="149.00"))
    with pytest.raises(ValidationError):
        load_agent_view(path)


def test_unknown_top_level_section_is_rejected(tmp_path, refund_task_path):
    path = _write_variant(tmp_path, refund_task_path, lambda raw: raw.update(hints={"answer": "refund"}))
    with pytest.raises(ValidationError):
        load_evaluator_view(path)


def test_agent_receives_exactly_the_agent_view(refund_task_path, agent_view):
    class SpyAgent:
        name = "spy"
        received = None

        def run(self, task, runtime):
            self.received = task
            return AgentResult(final_response="noop", stop_reason="completed")

    spy = SpyAgent()
    run_attempt(refund_task_path, spy)
    assert type(spy.received) is AgentTaskView
    assert spy.received == agent_view


def test_hidden_sections_cannot_change_agent_view(tmp_path, refund_task_path):
    original = load_agent_view(refund_task_path)

    def change_hidden(raw):
        raw["task_id"] = "secret_expected_failure"
        raw["fixture"]["approval_decisions"] = {"ORD-1001": "rejected"}
        raw["fixture"]["customers"]["CUST-001"]["name"] = "hidden sentinel"
        raw["expected"] = {"outcome": "no_refund", "order_id": "ORD-1001"}

    path = _write_variant(tmp_path, refund_task_path, change_hidden)
    assert load_agent_view(path) == original


@pytest.mark.parametrize("field", ["fixture", "expected", "task_id", "hints"])
def test_all_hidden_fields_are_rejected_in_agent_section(tmp_path, refund_task_path, field):
    path = _write_variant(tmp_path, refund_task_path, lambda raw: raw["agent"].update({field: "secret"}))
    with pytest.raises(ValidationError):
        load_agent_view(path)
