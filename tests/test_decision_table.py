"""Decision table tests: tree compression, ordering, caps, validation.

Run with: python tests/test_decision_table.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import app.decision_table as decision_table_module  # noqa: E402
from app.main import app  # noqa: E402
from app.store import store  # noqa: E402

client = TestClient(app)


def setup_function():
    store.clear()


POLICY = {
    "rules": [
        {
            "id": "approve",
            "priority": 10,
            "conclusion": "approve",
            "expr": {
                "all": [
                    {"field": "income", "op": "ge", "value": 50000},
                    {"field": "debt_ratio", "op": "lt", "value": 0.4},
                ]
            },
        },
        {
            "id": "reject",
            "priority": 10,
            "conclusion": "reject",
            "expr": {"field": "debt_ratio", "op": "ge", "value": 0.6},
        },
    ]
}

SIG_APPROVE = {"status": "decided", "conclusion": "approve",
               "decided_priority": 10}
SIG_REJECT = {"status": "decided", "conclusion": "reject",
              "decided_priority": 10}
SIG_NOMATCH = {"status": "no_match", "conclusion": None,
               "decided_priority": None}
SIG_UNKNOWN = {"status": "unknown", "conclusion": None,
               "decided_priority": 10,
               "minimal_missing_field_sets": [["debt_ratio"]],
               "pending_conclusions": ["approve", "reject"]}

DOMAINS = {"debt_ratio": [0.2, 0.55, 0.7], "income": [40000, 80000]}


def create_policy(policy=None):
    r = client.post("/api/v1/policies", json=policy or POLICY)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def post_table(pid, body):
    return client.post(f"/api/v1/policies/{pid}/decision-table", json=body)


# --------------------------------------------------------------------------- #
# Tree construction and compression
# --------------------------------------------------------------------------- #

def test_basic_table_with_partial_compression():
    pid = create_policy()
    r = post_table(pid, {"domains": DOMAINS})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["policy_id"] == pid
    assert body["fields"] == ["debt_ratio", "income"]
    assert body["total_combinations"] == 6
    # debt_ratio=0.55 and 0.7 collapse to leaves (income irrelevant there);
    # debt_ratio=0.2 keeps its income branch; the identical no_match leaves
    # under different prefixes are never merged across the tree.
    assert body["tree"] == {
        "field": "debt_ratio",
        "branches": [
            {"value": 0.2, "node": {
                "field": "income",
                "branches": [
                    {"value": 40000, "node": {"signature": SIG_NOMATCH}},
                    {"value": 80000, "node": {"signature": SIG_APPROVE}},
                ],
            }},
            {"value": 0.55, "node": {"signature": SIG_NOMATCH}},
            {"value": 0.7, "node": {"signature": SIG_REJECT}},
        ],
    }


def test_fields_sorted_lexicographically_regardless_of_declaration_order():
    pid = create_policy()
    r = post_table(pid, {"domains": {
        "income": [40000, 80000],
        "debt_ratio": [0.2, 0.55, 0.7],
    }})
    assert r.status_code == 200, r.text
    assert r.json()["fields"] == ["debt_ratio", "income"]
    # The root branches on the lexicographically first field.
    assert r.json()["tree"]["field"] == "debt_ratio"


def test_domain_and_value_declaration_order_do_not_change_response():
    pid = create_policy()
    forward = post_table(pid, {"domains": DOMAINS})
    shuffled = post_table(pid, {"domains": {
        "income": [80000, 40000],
        "debt_ratio": [0.7, 0.2, 0.55],
    }})
    assert forward.status_code == 200 and shuffled.status_code == 200
    assert forward.json() == shuffled.json()


def test_branch_values_sorted_by_canonical_json():
    pid = create_policy({"rules": [
        {"id": "hit", "priority": 1, "conclusion": "hit",
         "expr": {"field": "n", "op": "eq", "value": 2}},
    ]})
    r = post_table(pid, {"domains": {"n": [2, 10, 1]}})
    assert r.status_code == 200, r.text
    tree = r.json()["tree"]
    # Canonical JSON string order: "1" < "10" < "2".
    assert [branch["value"] for branch in tree["branches"]] == [1, 10, 2]
    assert tree["branches"][0]["node"] == {"signature": SIG_NOMATCH}
    assert tree["branches"][1]["node"] == {"signature": SIG_NOMATCH}
    assert tree["branches"][2]["node"] == {
        "signature": {"status": "decided", "conclusion": "hit",
                      "decided_priority": 1}
    }


def test_fully_identical_table_collapses_to_single_leaf():
    pid = create_policy()
    # `other` is never referenced by the policy: every combination is the
    # same unknown signature, so the whole tree folds into one leaf.
    r = post_table(pid, {"domains": {"other": [1, 2, 3]}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"] == ["other"]
    assert body["total_combinations"] == 3
    assert body["tree"] == {"signature": SIG_UNKNOWN}


def test_irrelevant_field_collapses_away():
    pid = create_policy({"rules": [
        {"id": "adult", "priority": 1, "conclusion": "adult",
         "expr": {"field": "age", "op": "ge", "value": 18}},
    ]})
    r = post_table(pid, {"domains": {"age": [10, 30], "label": ["x", "y"]}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"] == ["age", "label"]
    assert body["total_combinations"] == 4
    sig_adult = {"status": "decided", "conclusion": "adult",
                 "decided_priority": 1}
    # `label` discriminates nothing, so only the `age` node remains.
    assert body["tree"] == {
        "field": "age",
        "branches": [
            {"value": 10, "node": {"signature": SIG_NOMATCH}},
            {"value": 30, "node": {"signature": sig_adult}},
        ],
    }


def test_partial_branch_agreement_never_merges():
    # Three of four leaves share a signature; only the (2, 2) combination
    # differs, so both levels keep branching where any sibling differs.
    pid = create_policy({"rules": [
        {"id": "hit", "priority": 1, "conclusion": "hit",
         "expr": {"all": [
             {"field": "a", "op": "eq", "value": 2},
             {"field": "b", "op": "eq", "value": 2}]}},
    ]})
    sig_hit = {"status": "decided", "conclusion": "hit",
               "decided_priority": 1}
    r = post_table(pid, {"domains": {"a": [1, 2], "b": [1, 2]}})
    assert r.status_code == 200, r.text
    assert r.json()["tree"] == {
        "field": "a",
        "branches": [
            {"value": 1, "node": {"signature": SIG_NOMATCH}},
            {"value": 2, "node": {
                "field": "b",
                "branches": [
                    {"value": 1, "node": {"signature": SIG_NOMATCH}},
                    {"value": 2, "node": {"signature": sig_hit}},
                ],
            }},
        ],
    }


def test_single_combination_yields_bare_leaf():
    pid = create_policy()
    r = post_table(pid, {"domains": {"debt_ratio": [0.2]}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_combinations"] == 1
    # income is missing everywhere: one unknown combination, one leaf.
    assert body["tree"] == {"signature": {
        "status": "unknown", "conclusion": None, "decided_priority": 10,
        "minimal_missing_field_sets": [["income"]],
        "pending_conclusions": ["approve"],
    }}


def test_conflict_and_unknown_signatures_as_leaves():
    pid = create_policy({"rules": [
        {"id": "r-yes", "priority": 5, "conclusion": "yes",
         "expr": {"field": "x", "op": "eq", "value": 1}},
        {"id": "r-no", "priority": 5, "conclusion": "no",
         "expr": {"field": "x", "op": "eq", "value": 1}},
    ]})
    r = post_table(pid, {"domains": {"x": [1, 2], "y": [None]}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_combinations"] == 2
    sig_conflict = {"status": "conflict", "conclusion": None,
                    "decided_priority": 5,
                    "conflicting_conclusions": ["no", "yes"]}
    # The single-value `y` domain collapses; x=1 conflicts, x=2 no_match.
    assert body["tree"] == {
        "field": "x",
        "branches": [
            {"value": 1, "node": {"signature": sig_conflict}},
            {"value": 2, "node": {"signature": SIG_NOMATCH}},
        ],
    }


def test_complex_literal_domain_values():
    pid = create_policy({"rules": [
        {"id": "tagged", "priority": 1, "conclusion": "tagged",
         "expr": {"field": "tag", "op": "eq", "value": [1, 2]}},
    ]})
    r = post_table(pid, {"domains": {"tag": [[3], [1, 2]]}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_combinations"] == 2
    sig_tagged = {"status": "decided", "conclusion": "tagged",
                  "decided_priority": 1}
    # Canonical order: "[1,2]" < "[3]".
    assert body["tree"] == {
        "field": "tag",
        "branches": [
            {"value": [1, 2], "node": {"signature": sig_tagged}},
            {"value": [3], "node": {"signature": SIG_NOMATCH}},
        ],
    }


def test_repeated_requests_are_byte_identical():
    pid = create_policy()
    first = post_table(pid, {"domains": DOMAINS})
    second = post_table(pid, {"domains": DOMAINS})
    assert first.status_code == 200 and second.status_code == 200
    assert first.content == second.content


# --------------------------------------------------------------------------- #
# Combination cap
# --------------------------------------------------------------------------- #

def test_env_variable_configures_default_cap():
    key = "MAX_DECISION_TABLE_COMBINATIONS"
    original = os.environ.get(key)
    try:
        os.environ[key] = "7"
        assert decision_table_module._configured_max_combinations() == 7
        os.environ[key] = "not-a-number"
        assert decision_table_module._configured_max_combinations() == 100_000
        os.environ[key] = "-5"
        assert decision_table_module._configured_max_combinations() == 100_000
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def test_cap_exceeded_returns_422_and_exact_cap_succeeds():
    pid = create_policy()
    module = decision_table_module
    original = module.MAX_DECISION_TABLE_COMBINATIONS
    module.MAX_DECISION_TABLE_COMBINATIONS = 5
    try:
        r = post_table(pid, {"domains": DOMAINS})  # 3 x 2 = 6 > 5
        assert r.status_code == 422, r.text
        out = r.json()
        assert out["detail"] == "Decision table construction failed"
        assert isinstance(out["errors"], list) and out["errors"]
        assert "MAX_DECISION_TABLE_COMBINATIONS" in out["errors"][0]

        # Dropping one value lands exactly on the cap and succeeds.
        r = post_table(pid, {"domains": {"debt_ratio": [0.2, 0.55, 0.7],
                                         "income": [40000]}})
        assert r.status_code == 200, r.text
        assert r.json()["total_combinations"] == 3
    finally:
        module.MAX_DECISION_TABLE_COMBINATIONS = original


def test_cap_check_is_pure_function_level():
    record = {"rules": POLICY["rules"]}
    try:
        decision_table_module.decision_table(record, DOMAINS,
                                             max_combinations=2)
        raised = False
    except decision_table_module.DecisionTableSearchError as exc:
        raised = True
        assert "exceeds 2 combinations" in str(exc)
    assert raised
    result = decision_table_module.decision_table(record, DOMAINS,
                                                  max_combinations=6)
    assert result["total_combinations"] == 6


# --------------------------------------------------------------------------- #
# Validation and storage guarantees
# --------------------------------------------------------------------------- #

def test_invalid_bodies_return_422():
    pid = create_policy()
    bad_bodies = [
        [1, 2],                                        # not an object
        {},                                            # no domains
        {"domains": {}},                               # empty domains
        {"domains": "nope"},                           # domains not an object
        {"domains": {"x": 5}},                         # domain not an array
        {"domains": {"x": []}},                        # empty domain
        {"domains": {"x": [1]}, "extra": 1},           # unknown key
        {"domains": {"x": [1, 1]}},                    # duplicate values
        {"domains": {"x": [[1, 2], [1, 2]]}},          # duplicate arrays
        {"domains": {"x": [{"a": 1, "b": 2},           # duplicate objects
                           {"b": 2, "a": 1}]}},
        {"domains": {"": [1]}},                        # empty field name
    ]
    for body in bad_bodies:
        r = post_table(pid, body)
        assert r.status_code == 422, body
        out = r.json()
        assert out["detail"] == "Policy structure validation failed"
        assert isinstance(out["errors"], list) and out["errors"]


def test_unknown_keys_and_domain_errors_are_reported_together():
    pid = create_policy()
    r = post_table(pid, {"domains": {"x": []}, "bogus": True})
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert any("unknown keys" in e and "bogus" in e for e in errors)
    assert any(e.startswith("domains.x:") for e in errors)


def test_true_and_one_are_distinct_domain_values():
    pid = create_policy()
    r = post_table(pid, {"domains": {"x": [True, 1]}})
    assert r.status_code == 200, r.text
    assert r.json()["total_combinations"] == 2


def test_non_finite_numbers_are_not_json_literals():
    pid = create_policy()
    for literal in ("NaN", "Infinity", "-Infinity"):
        r = client.post(
            f"/api/v1/policies/{pid}/decision-table",
            content='{"domains": {"x": [%s]}}' % literal,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422, literal
        assert any("JSON literal" in e for e in r.json()["errors"])


def test_unknown_policy_returns_404():
    r = post_table("nope", {"domains": {"x": [1]}})
    assert r.status_code == 404


def test_malformed_json_is_400():
    pid = create_policy()
    r = client.post(
        f"/api/v1/policies/{pid}/decision-table",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_decision_table_never_writes_to_store():
    pid = create_policy()
    before = client.get(f"/api/v1/policies/{pid}").json()

    r = post_table(pid, {"domains": DOMAINS})
    assert r.status_code == 200
    assert client.get(f"/api/v1/policies/{pid}").json() == before
    assert store.count() == 1

    # A rejected (422) request does not create or modify policies either.
    r = post_table(pid, {"domains": {"x": []}})
    assert r.status_code == 422
    assert store.count() == 1
    assert client.get(f"/api/v1/policies/{pid}").json() == before


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        setup_function()
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
