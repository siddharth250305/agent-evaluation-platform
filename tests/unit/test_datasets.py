import shutil
from collections import Counter

import pytest
import yaml
from pydantic import ValidationError

from app.datasets import DatasetError, assign_splits, load_dataset
from app.task_spec import ExpectedOutcome, TrajectorySpec
from app.tasks import DATASETS_DIR, load_agent_view, load_evaluator_view

SUPPORT = DATASETS_DIR / "support"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(SUPPORT)


@pytest.fixture
def copy_of_support(tmp_path):
    target = tmp_path / "support"
    shutil.copytree(SUPPORT, target)
    return target


def _edit_yaml(path, mutate):
    raw = yaml.safe_load(path.read_text())
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


# --- the shipped dataset --------------------------------------------------------------


def test_support_dataset_covers_every_category(dataset):
    assert len(dataset.tasks) == 50
    assert len({t.family for t in dataset.tasks}) == 25
    assert {t.category for t in dataset.tasks} == {
        "read_only_lookup", "eligible_action", "ineligible_action", "missing_info",
        "access_boundary", "tool_failure", "duplicate_operation", "untrusted_content"
    }


def test_family_level_split_is_60_20_20(dataset):
    families = {t.family: t.split for t in dataset.tasks}
    assert Counter(families.values()) == {"dev": 15, "validation": 5, "test": 5}
    assert Counter(t.split for t in dataset.tasks) == {"dev": 30, "validation": 10, "test": 10}


def test_paraphrases_never_cross_splits(dataset):
    by_id = {t.task_id: t for t in dataset.tasks}
    paraphrases = [t for t in dataset.tasks if t.paraphrase_of]
    assert paraphrases
    for task in paraphrases:
        original = by_id[task.paraphrase_of]
        assert (task.family, task.split) == (original.family, original.split)


def test_split_manifest_was_generated_by_assign_splits(dataset):
    manifest = yaml.safe_load((SUPPORT / "splits.yaml").read_text())
    original = {"approval_expired": "dev", "approval_rejected": "test",
                "cross_customer_request": "dev", "escalation_damaged": "test",
                "no_tools_thanks": "dev", "order_status": "validation",
                "partial_refund_remaining": "dev", "policy_question": "dev",
                "refund_eligible": "validation", "return_in_transit": "dev"}
    assert manifest["families"] == assign_splits(
        manifest["families"], seed=manifest["split_seed"], existing=original
    )


def test_content_hash_is_stable(dataset):
    again = load_dataset(SUPPORT)
    assert dataset.content_hash == again.content_hash
    assert len(dataset.content_hash) == 64
    assert [t.content_hash for t in dataset.tasks] == [t.content_hash for t in again.tasks]


def test_agent_views_never_contain_evaluator_only_sections(dataset):
    for task in dataset.tasks:
        agent = load_agent_view(task.path).model_dump_json()
        evaluator = load_evaluator_view(task.path)
        assert evaluator.reference.final_response not in agent
        assert task.task_id not in agent
        for marker in ("trajectory", "required_calls", "reference", "expected", "MISSING_"):
            assert marker not in agent


def test_shared_fixture_is_resolved_with_the_task_session(dataset):
    view = load_evaluator_view(dataset.task("approval_rejected_001").path)
    assert view.fixture["session"] == {"customer_id": "CUST-003", "ticket_id": "T-7002"}
    assert view.fixture["orders"]["ORD-3002"]["customer_id"] == "CUST-003"


# --- hashing ---------------------------------------------------------------------------


def test_changing_a_task_changes_its_hash_and_the_dataset_hash(dataset, copy_of_support):
    _edit_yaml(copy_of_support / "order_status_001.yaml", lambda raw: raw["agent"].update(user_request="Where is it?"))
    changed = load_dataset(copy_of_support)
    before = {t.task_id: t.content_hash for t in dataset.tasks}
    after = {t.task_id: t.content_hash for t in changed.tasks}
    assert changed.content_hash != dataset.content_hash
    assert [k for k in before if before[k] != after[k]] == ["order_status_001"]


def test_changing_the_shared_fixture_changes_every_task_that_uses_it(dataset, copy_of_support):
    _edit_yaml(copy_of_support / "fixtures" / "retail_v1.yaml",
               lambda raw: raw["customers"]["CUST-001"].update(name="Renamed"))
    changed = {t.task_id: t.content_hash for t in load_dataset(copy_of_support).tasks}
    original = {t.task_id: t.content_hash for t in dataset.tasks}
    assert changed["refund_eligible_001"] == original["refund_eligible_001"]  # inline fixture
    assert changed["refund_eligible_002"] != original["refund_eligible_002"]


def test_review_notes_do_not_change_content_hashes(dataset, copy_of_support):
    _edit_yaml(copy_of_support / "order_status_001.yaml", lambda raw: raw["review"].update(notes="looked again"))
    assert load_dataset(copy_of_support).content_hash == dataset.content_hash


# --- validation errors -------------------------------------------------------------------


def test_task_id_must_match_its_file_name(copy_of_support):
    _edit_yaml(copy_of_support / "order_status_001.yaml", lambda raw: raw.update(task_id="order_status_999"))
    with pytest.raises(DatasetError, match="file name"):
        load_dataset(copy_of_support)


def test_every_family_needs_a_split(copy_of_support):
    _edit_yaml(copy_of_support / "splits.yaml", lambda raw: raw["families"].pop("order_status"))
    with pytest.raises(DatasetError, match="order_status"):
        load_dataset(copy_of_support)


def test_split_families_must_have_tasks(copy_of_support):
    _edit_yaml(copy_of_support / "splits.yaml", lambda raw: raw["families"].update(ghost_family="test"))
    with pytest.raises(DatasetError, match="ghost_family"):
        load_dataset(copy_of_support)


def test_split_ratios_are_enforced(copy_of_support):
    def skew_8_1_1(raw):
        names = sorted(raw["families"])
        raw["families"] = {f: "validation" if i == 0 else "test" if i == 1 else "dev" for i, f in enumerate(names)}

    _edit_yaml(copy_of_support / "splits.yaml", skew_8_1_1)
    with pytest.raises(DatasetError, match="ratio"):
        load_dataset(copy_of_support)


def test_paraphrase_must_point_at_a_task_in_the_same_family(copy_of_support):
    _edit_yaml(copy_of_support / "order_status_002.yaml", lambda raw: raw.update(paraphrase_of="policy_question_001"))
    with pytest.raises(DatasetError, match="paraphrase"):
        load_dataset(copy_of_support)


def test_paraphrase_target_must_exist(copy_of_support):
    _edit_yaml(copy_of_support / "order_status_002.yaml", lambda raw: raw.update(paraphrase_of="nope_001"))
    with pytest.raises(DatasetError, match="paraphrase"):
        load_dataset(copy_of_support)


def test_invalid_task_files_are_reported_with_their_path(copy_of_support):
    _edit_yaml(copy_of_support / "order_status_001.yaml", lambda raw: raw["agent"].update(tools=["teleport"]))
    with pytest.raises(DatasetError, match="order_status_001.yaml"):
        load_dataset(copy_of_support)


# --- spec validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected",
    [
        {"outcome": "refund_issued", "order_id": "ORD-1"},  # amount missing
        {"outcome": "no_refund", "order_id": "ORD-1", "refund_amount": "5.00"},
        {"outcome": "teleport"},
        {"outcome": "no_refund", "ticket_status": "closed"},
    ],
)
def test_invalid_expected_outcomes_are_rejected(expected):
    with pytest.raises(ValidationError):
        ExpectedOutcome.model_validate(expected)


@pytest.mark.parametrize(
    "spec",
    [
        {"tools_needed": False, "required_calls": [{"id": "a", "tool": "get_order"}]},
        {"tools_needed": True, "required_calls": []},
        {"tools_needed": True, "required_calls": [{"id": "a", "tool": "teleport"}]},
        {"tools_needed": True, "required_calls": [{"id": "a", "tool": "get_order"}, {"id": "a", "tool": "get_order"}]},
        {"tools_needed": True, "required_calls": [{"id": "a", "tool": "get_order"}], "order": [["a", "ghost"]]},
        {
            "tools_needed": True,
            "required_calls": [{"id": "a", "tool": "get_order"}, {"id": "b", "tool": "get_order"}],
            "order": [["a", "b"], ["b", "a"]],
        },
    ],
    ids=["calls-without-tools", "no-calls", "unknown-tool", "duplicate-id", "unknown-order-id", "cycle"],
)
def test_invalid_trajectory_specs_are_rejected(spec):
    with pytest.raises(ValidationError):
        TrajectorySpec.model_validate(spec)


# --- split assignment ----------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "expected"), [(10, (6, 2, 2)), (5, (3, 1, 1)), (20, (12, 4, 4))])
def test_assign_splits_hits_the_ratios(n, expected):
    splits = assign_splits([f"family_{i}" for i in range(n)], seed="s")
    counts = Counter(splits.values())
    assert (counts["dev"], counts["validation"], counts["test"]) == expected


def test_assign_splits_is_deterministic_and_seed_dependent():
    families = [f"family_{i}" for i in range(10)]
    assert assign_splits(families, seed="a") == assign_splits(list(reversed(families)), seed="a")
    assert assign_splits(families, seed="a") != assign_splits(families, seed="b")


# --- task-file validation paths ------------------------------------------------------------


def _rewrite(copy_dir, task_id, mutate):
    path = copy_dir / f"{task_id}.yaml"
    _edit_yaml(path, mutate)
    return path


@pytest.mark.parametrize("ref", ["../splits", "retail_v1/../../x", "", 7])
def test_fixture_refs_cannot_escape_the_fixtures_directory(copy_of_support, ref):
    path = _rewrite(copy_of_support, "order_status_001", lambda raw: raw["fixture"].update(ref=ref))
    with pytest.raises(ValueError, match="invalid fixture ref"):
        load_evaluator_view(path)


def test_unknown_fixture_reference_keys_are_rejected(copy_of_support):
    path = _rewrite(copy_of_support, "order_status_001", lambda raw: raw["fixture"].update(expected="leak"))
    with pytest.raises(ValueError, match="unknown fixture reference keys"):
        load_evaluator_view(path)


def test_fixture_overrides_merge_into_the_shared_world(copy_of_support):
    def override(raw):
        raw["fixture"]["overrides"] = {
            "approval_decisions": {"ORD-1004": "approved"},
            "policies": [{"policy_id": "POL-X", "title": "Only", "text": "Replaced list."}],
        }

    view = load_evaluator_view(_rewrite(copy_of_support, "order_status_001", override))
    assert view.fixture["approval_decisions"]["ORD-1004"] == "approved"
    assert view.fixture["approval_decisions"]["ORD-1001"] == "approved"  # dicts merge
    assert [p["policy_id"] for p in view.fixture["policies"]] == ["POL-X"]  # lists replace


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["agent"].update(tools=["get_order", "get_order"]),
        lambda raw: raw.update(expected={"outcome": "refund_issued"}),
        lambda raw: raw["agent"].update(tools=["search_policy"]),  # reference/trajectory use get_order
    ],
    ids=["duplicate-tools", "invalid-expected", "reference-uses-unavailable-tool"],
)
def test_inconsistent_task_files_are_rejected(copy_of_support, mutate):
    path = _rewrite(copy_of_support, "order_status_001", mutate)
    with pytest.raises(ValidationError):
        load_evaluator_view(path)


# --- regressions from the Phase 3 review: split stability, hashing, fixture errors ----------


def test_adding_a_family_never_moves_existing_families():
    families = [f"family_{i}" for i in range(10)]
    existing = assign_splits(families, seed="s")
    extended = assign_splits([*families, "family_new"], seed="s", existing=existing)
    assert {f: extended[f] for f in families} == existing
    assert extended["family_new"] in ("dev", "validation", "test")


@pytest.mark.parametrize("n", [3, 4])
def test_every_split_gets_at_least_one_family(n):
    assert set(assign_splits([f"f{i}" for i in range(n)], seed="s").values()) == {"dev", "validation", "test"}


def test_too_few_families_to_split_is_an_error():
    with pytest.raises(ValueError):
        assign_splits(["a", "b"], seed="s")


def test_empty_split_in_the_manifest_is_rejected(copy_of_support):
    def no_test_split(raw):
        raw["families"] = {f: ("validation" if s == "test" else s) for f, s in raw["families"].items()}

    _edit_yaml(copy_of_support / "splits.yaml", no_test_split)
    with pytest.raises(DatasetError, match="test"):
        load_dataset(copy_of_support)


def test_changing_paraphrase_links_changes_the_hash(dataset, copy_of_support):
    _edit_yaml(copy_of_support / "order_status_002.yaml", lambda raw: raw.pop("paraphrase_of"))
    assert load_dataset(copy_of_support).content_hash != dataset.content_hash


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["fixture"].pop("session"),
        lambda raw: raw["fixture"].update(overrides={"orders": "not-a-mapping"}),
        lambda raw: raw["fixture"].update(ref="retäil_v1"),
    ],
    ids=["missing-session", "bad-override", "non-ascii-ref"],
)
def test_bad_fixture_references_are_dataset_errors(copy_of_support, mutate):
    _edit_yaml(copy_of_support / "order_status_001.yaml", mutate)
    with pytest.raises(DatasetError, match="order_status_001.yaml"):
        load_dataset(copy_of_support)


def test_entity_overrides_merge_fields_instead_of_replacing_the_entity(copy_of_support):
    _edit_yaml(
        copy_of_support / "order_status_001.yaml",
        lambda raw: raw["fixture"].update(overrides={"orders": {"ORD-1004": {"status": "delayed"}}}),
    )
    order = load_evaluator_view(copy_of_support / "order_status_001.yaml").fixture["orders"]["ORD-1004"]
    assert (order["status"], order["customer_id"]) == ("delayed", "CUST-001")


def test_paraphrase_cycles_are_rejected(copy_of_support):
    _edit_yaml(copy_of_support / "order_status_001.yaml", lambda raw: raw.update(paraphrase_of="order_status_002"))
    with pytest.raises(DatasetError, match="cycle"):
        load_dataset(copy_of_support)


def test_undeclared_duplicate_prompt_across_families_is_rejected(copy_of_support):
    prompt = yaml.safe_load((copy_of_support / "order_status_001.yaml").read_text())["agent"]["user_request"]
    _edit_yaml(copy_of_support / "policy_question_001.yaml", lambda raw: raw["agent"].update(user_request="  " + prompt.upper() + "  "))
    with pytest.raises(DatasetError, match="duplicate prompt across families"):
        load_dataset(copy_of_support)


@pytest.mark.parametrize("ratios", [
    {"dev": -0.2, "validation": 0.6, "test": 0.6},
    {"dev": float("nan"), "validation": .2, "test": .2},
])
def test_invalid_manifest_ratios_are_rejected(copy_of_support, ratios):
    _edit_yaml(copy_of_support / "splits.yaml", lambda raw: raw.update(ratios=ratios))
    with pytest.raises(DatasetError, match="ratios"):
        load_dataset(copy_of_support)


def test_every_family_has_exactly_two_variants_and_review_notes(dataset):
    assert set(Counter(t.family for t in dataset.tasks).values()) == {2}
    for task in dataset.tasks:
        raw = yaml.safe_load(task.path.read_text())
        assert raw["review"]["status"] == "machine_checked"
        assert "Human review pending" in raw["review"]["notes"]


def test_family_cannot_mix_behavior_categories(copy_of_support):
    _edit_yaml(copy_of_support / "order_status_002.yaml", lambda raw: raw.update(category="tool_failure"))
    with pytest.raises(DatasetError, match="inconsistent category"):
        load_dataset(copy_of_support)
