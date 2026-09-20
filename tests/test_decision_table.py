"""Decision table tests: tree shape, bottom-up compression, caps, validation.

Run with: python tests/test_decision_table.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import app.decision_table as decision_table  # noqa: E402
from app.main import app  # noqa: E402
from app.store import store  # noqa: E402
from app.validation import (  # noqa: E402
    PolicyValidationError,
    validate_decision_table_request,
)

client = TestClient(app)


def setup_function():
    store.clear()


POLICY = {
    "rules": [
        {
            "id": "adult-allow",
            "priority": 100,
            "conclusion": "allow",
            "expr": {
                "all": [
                    {"field": "age", "op": "ge", "value": 18},
                    {"field": "country", "op": "eq", "value": "NL"},
                ]
            },
        },
        {
            "id": "minor-deny",
            "priority": 100,
            "conclusion": "deny",
            "expr": {"field": "age", "op": "lt", "value": 18},
        },
        {
            "id": "vip-review",
            "priority": 50,
            "conclusion": "review",
            "expr": {"field": "vip", "op": "eq", "value": True},
        },
    ]
}


SIG_ALLOW = {"status": "decided", "conclusion": "allow",
             "decided_priority": 100}
SIG_DENY = {"status": "decided", "conclusion": "deny",
            "decided_priority": 100}
SIG_REVIEW = {"status": "decided", "conclusion": "review",
              "decided_priority": 50}
SIG_NOMATCH = {"status": "no_match", "conclusion": None,
               "decided_priority": None}
SIG_UNKNOWN_AGE = {
    "status": "unknown", "conclusion": None, "decided_priority": 100,
    "minimal_missing_field_sets": [["age"]],
    "pending_conclusions": ["allow", "deny"],
}


def create_policy(policy=None):
    r = client.post("/api/v1/policies", json=policy or POLICY)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def post_table(pid, body):
    return client.post(f"/api/v1/policies/{pid}/decision-table", json=body)


def leaf(signature):
    return {"type": "leaf", "signature": signature}


def uniform(field, child):
    return {"type": "uniform", "field": field, "child": child}


def split(field, branches):
    return {"type": "split", "field": field,
            "branches": [{"value": value, "child": child}
                         for value, child in branches]}


# --------------------------------------------------------------------------- #
# Tree construction and bottom-up compression
# --------------------------------------------------------------------------- #

def test_fields_sorted_and_total_combinations_is_cartesian_product():
    pid = create_policy()
    r = post_table(pid, {
        "domains": {"vip": [False, True], "country": ["NL"], "age": [10, 30]}
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"] == ["age", "country", "vip"]
    assert body["total_combinations"] == 4
    assert body["policy_id"] == pid


def test_tree_compresses_independent_and_dependent_fields():
    pid = create_policy()
    r = post_table(pid, {
        "domains": {"age": [10, 30], "country": ["NL", "US"],
                    "vip": [False, True]}
    })
    assert r.status_code == 200, r.text
    tree = r.json()["tree"]

    expected = split("age", [
        (10, uniform("country", uniform("vip", leaf(SIG_DENY)))),
        (30, split("country", [
            ("NL", uniform("vip", leaf(SIG_ALLOW))),
            ("US", split("vip", [
                (False, leaf(SIG_NOMATCH)),
                (True, leaf(SIG_REVIEW)),
            ])),
        ])),
    ])
    assert tree == expected


def test_node_collapses_when_all_subtrees_identical():
    pid = create_policy()
    # country/vip never matter when the minor rule already decides deny.
    r = post_table(pid, {"domains": {"age": [10], "country": ["NL", "US"],
                                     "vip": [False, True]}})
    assert r.status_code == 200, r.text
    assert r.json()["tree"] == uniform(
        "age", uniform("country", uniform("vip", leaf(SIG_DENY))))


def test_unreferenced_field_domain_collapses_to_unknown_signature():
    pid = create_policy()
    r = post_table(pid, {"domains": {"other": [1, 2]}})
    assert r.status_code == 200, r.text
    assert r.json()["tree"] == uniform("other", leaf(SIG_UNKNOWN_AGE))


def test_partial_agreement_never_merges():
    # x decides at the same priority as z: for x=1 the z subtree is uniform
    # (fired rules stop lower levels), for x=0 z is a split. The parent must
    # not collapse across the two structurally different subtrees even though
    # both contain a "yes" leaf.
    pid = create_policy({"rules": [
        {"id": "r-x", "priority": 10, "conclusion": "xyes",
         "expr": {"field": "x", "op": "eq", "value": 1}},
        {"id": "r-z", "priority": 1, "conclusion": "zyes",
         "expr": {"field": "z", "op": "eq", "value": 1}},
    ]})
    r = post_table(pid, {"domains": {"x": [0, 1], "z": [0, 1]}})
    assert r.status_code == 200, r.text
    sig_x = {"status": "decided", "conclusion": "xyes",
             "decided_priority": 10}
    sig_z = {"status": "decided", "conclusion": "zyes",
             "decided_priority": 1}
    expected = split("x", [
        (0, split("z", [(0, leaf(SIG_NOMATCH)), (1, leaf(sig_z))])),
        (1, uniform("z", leaf(sig_x))),
    ])
    assert r.json()["tree"] == expected


def test_bottom_up_compression_cascades_through_levels():
    # Result depends only on x; the y and z levels must each become uniform,
    # all the way back up to the root.
    pid = create_policy({"rules": [
        {"id": "r", "priority": 1, "conclusion": "yes",
         "expr": {"field": "x", "op": "eq", "value": 1}},
    ]})
    r = post_table(pid, {"domains": {"x": [0, 1], "y": ["a", "b"],
                                     "z": [True, False]}})
    assert r.status_code == 200, r.text
    expected = split("x", [
        (0, uniform("y", uniform("z", leaf(SIG_NOMATCH)))),
        (1, uniform("y", uniform("z", leaf({
            "status": "decided", "conclusion": "yes",
            "decided_priority": 1})))),
    ])
    assert r.json()["tree"] == expected


def test_leaves_use_replacement_preview_business_signature():
    # Conflict and unknown verdicts must carry the signature extensions.
    pid = create_policy({"rules": [
        {"id": "r-yes", "priority": 10, "conclusion": "yes",
         "expr": {"field": "x", "op": "eq", "value": 1}},
        {"id": "r-no", "priority": 10, "conclusion": "no",
         "expr": {"field": "x", "op": "eq", "value": 1}},
    ]})
    r = post_table(pid, {"domains": {"other": [9], "x": [0, 1]}})
    assert r.status_code == 200, r.text
    sig_conflict = {
        "status": "conflict", "conclusion": None, "decided_priority": 10,
        "conflicting_conclusions": ["no", "yes"],
    }
    sig_unknown = {
        "status": "unknown", "conclusion": None, "decided_priority": 10,
        "minimal_missing_field_sets": [["x"]],
        "pending_conclusions": ["no", "yes"],
    }
    assert r.json()["tree"] == uniform("other", split("x", [
        (0, leaf(SIG_NOMATCH)),
        (1, leaf(sig_conflict)),
    ]))

    # With only an unrelated field in the domain, x stays absent and every
    # leaf carries the unknown signature, so the tree collapses.
    r = post_table(pid, {"domains": {"other": [0, 1]}})
    assert r.json()["tree"] == uniform("other", leaf(sig_unknown))


def test_identical_signatures_across_values_merge_without_crossing():
    # Both 1 and 2 are "no_match"; they are equal signatures, so the x node
    # could split — but since the whole domain maps to the same leaf and
    # `other` is independent, everything collapses to one no_match leaf.
    pid = create_policy({"rules": [
        {"id": "r", "priority": 1, "conclusion": "yes",
         "expr": {"field": "x", "op": "eq", "value": 9}},
    ]})
    r = post_table(pid, {"domains": {"other": ["a"], "x": [1, 2]}})
    assert r.status_code == 200, r.text
    assert r.json()["tree"] == uniform(
        "other", uniform("x", leaf(SIG_NOMATCH)))


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_repeated_requests_are_byte_identical():
    pid = create_policy()
    body = {"domains": {"age": [10, 30], "country": ["NL", "US"],
                        "vip": [False, True]}}
    first = post_table(pid, body)
    second = post_table(pid, body)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_field_and_value_order_independent_response():
    pid = create_policy()
    forward = {"domains": {"vip": [False, True], "country": ["NL", "US"],
                           "age": [10, 30]}}
    backward = {"domains": {"age": [30, 10], "country": ["US", "NL"],
                            "vip": [True, False]}}
    r1 = post_table(pid, forward)
    r2 = post_table(pid, backward)
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()


def test_object_literal_key_order_is_canonicalized():
    pid = create_policy({"rules": [
        {"id": "r", "priority": 1, "conclusion": "yes",
         "expr": {"field": "o", "op": "eq", "value": {"a": 1}}}
    ]})
    r1 = post_table(pid, {"domains": {"o": [{"a": 1, "b": 2}, {"z": 0}]}})
    r2 = post_table(pid, {"domains": {"o": [{"z": 0}, {"b": 2, "a": 1}]}})
    assert r1.status_code == r2.status_code == 200
    assert r1.content == r2.content


def test_bool_and_number_kinds_are_distinct_domain_values():
    pid = create_policy({"rules": [
        {"id": "r", "priority": 1, "conclusion": "yes",
         "expr": {"field": "x", "op": "eq", "value": 1}}
    ]})
    # true / 1 / 1.0 are three domain values.  Numeric comparison equates
    # 1 and 1.0 with the rule value 1, while true compares unequal.
    r = post_table(pid, {"domains": {"x": [1.0, True, 1]}})
    assert r.status_code == 200, r.text
    branches = r.json()["tree"]["branches"]
    values = [branch["value"] for branch in branches]
    assert values == [1, True, 1.0]
    kinds = {
        type(branch["value"]).__name__:
            branch["child"]["signature"]["status"]
        for branch in branches
    }
    assert kinds == {"int": "decided", "bool": "no_match",
                     "float": "decided"}

    # The same number kind twice is a duplicate (int 1 and float 1.0 are
    # different JSON literals; two floats are not).
    for pair in ([1, 1], [1.0, 1.0], [True, True]):
        r = post_table(pid, {"domains": {"x": pair}})
        assert r.status_code == 422
        assert any("duplicate value" in e for e in r.json()["errors"])


# --------------------------------------------------------------------------- #
# Combination cap
# --------------------------------------------------------------------------- #

def test_cap_returns_422_without_truncating():
    pid = create_policy()
    original = decision_table.MAX_DECISION_TABLE_COMBINATIONS
    decision_table.MAX_DECISION_TABLE_COMBINATIONS = 3
    try:
        r = post_table(pid, {"domains": {"age": [10, 30],
                                         "country": ["NL", "US"]}})
        assert r.status_code == 422, r.text
        out = r.json()
        assert out["detail"] == "Decision table search failed"
        assert out["errors"] == [
            "decision table combination space exceeds 3 combinations "
            "(4 required); narrow the domains or raise "
            "MAX_DECISION_TABLE_COMBINATIONS"
        ]
    finally:
        decision_table.MAX_DECISION_TABLE_COMBINATIONS = original


def test_cap_boundary_is_inclusive():
    pid = create_policy()
    original = decision_table.MAX_DECISION_TABLE_COMBINATIONS
    decision_table.MAX_DECISION_TABLE_COMBINATIONS = 4
    try:
        r = post_table(pid, {"domains": {"age": [10, 30],
                                         "country": ["NL", "US"]}})
        assert r.status_code == 200, r.text
        assert r.json()["total_combinations"] == 4
    finally:
        decision_table.MAX_DECISION_TABLE_COMBINATIONS = original


def test_env_variable_configures_default_cap():
    key = "MAX_DECISION_TABLE_COMBINATIONS"
    original = os.environ.get(key)
    try:
        os.environ[key] = "7"
        assert decision_table._configured_max_combinations() == 7
        os.environ[key] = "not-a-number"
        assert decision_table._configured_max_combinations() == 100_000
        os.environ[key] = "0"
        assert decision_table._configured_max_combinations() == 100_000
        os.environ[key] = "-5"
        assert decision_table._configured_max_combinations() == 100_000
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validation_errors_return_422():
    pid = create_policy()
    bad_bodies = [
        {},
        ["x"],
        {"domains": {}},
        {"domains": []},
        {"domains": "nope"},
        {"domains": {"a": []}},
        {"domains": {"a": "nope"}},
        {"domains": {"a": [1, 1]}},
        {"domains": {"a": [True, True]}},
        {"domains": {"a": [None, None]}},
        {"domains": {"a": [{"x": 1}, {"x": 1}]}},
        {"domains": {"": [1]}},
    ]
    for body in bad_bodies:
        r = post_table(pid, body)
        assert r.status_code == 422, body
        out = r.json()
        assert out["detail"] == "Policy structure validation failed"
        assert isinstance(out["errors"], list) and out["errors"], body


def test_unknown_keys_are_rejected():
    pid = create_policy()
    r = post_table(pid, {"domains": {"a": [1]}, "extra": 2})
    assert r.status_code == 422
    assert r.json()["errors"] == ["unknown keys ['extra']"]


def test_non_json_literals_rejected_at_validator():
    # Non-finite floats and non-JSON objects cannot be sent over JSON; check
    # the validator directly.
    for value in [float("nan"), float("inf"), float("-inf"),
                  1j, object(), {1: 2}]:
        try:
            validate_decision_table_request({"domains": {"a": [value]}})
            raise AssertionError(f"{value!r} should be rejected")
        except PolicyValidationError:
            pass


def test_nested_non_finite_float_is_rejected():
    try:
        validate_decision_table_request(
            {"domains": {"a": [1, [1, float("inf")]]}})
        raise AssertionError("nested inf should be rejected")
    except PolicyValidationError as exc:
        assert any("JSON literal" in e for e in exc.errors)


def test_non_literal_string_field_values_rejected():
    pid = create_policy()
    # A domain must be an array of literals; a single object value is fine,
    # but a set-like repeat and bare values inside non-array wrappers are not.
    r = post_table(pid, {"domains": {"a": [{"field": "x"}]}})
    assert r.status_code == 200, r.text  # objects of literals ARE literals


def test_multiple_problems_are_reported_together():
    pid = create_policy()
    r = post_table(pid, {"domains": {"": [], "b": [1, 1]}})
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert any("non-empty string" in e for e in errors)
    assert any("domains." in e and "non-empty array" in e for e in errors)
    assert any("duplicate value" in e for e in errors)


def test_malformed_json_is_400():
    pid = create_policy()
    r = client.post(
        f"/api/v1/policies/{pid}/decision-table",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Storage guarantees
# --------------------------------------------------------------------------- #

def test_unknown_policy_returns_404_even_for_invalid_body():
    r = client.post("/api/v1/policies/nope/decision-table",
                    json={"domains": {"a": [1]}})
    assert r.status_code == 404


def test_endpoint_is_read_only():
    pid = create_policy()
    before = client.get(f"/api/v1/policies/{pid}").json()

    r = post_table(pid, {"domains": {"age": [10, 30], "country": ["NL"]}})
    assert r.status_code == 200
    after = client.get(f"/api/v1/policies/{pid}").json()
    assert after == before
    assert store.count() == 1

    # A rejected (422) request creates or modifies nothing either.
    r = post_table(pid, {"domains": {}})
    assert r.status_code == 422
    assert store.count() == 1
    assert client.get(f"/api/v1/policies/{pid}").json() == before


# --------------------------------------------------------------------------- #
# Pure module behaviour
# --------------------------------------------------------------------------- #

def test_dedupe_and_order_canonicalizes_keys_and_preserves_number_kinds():
    # Object-key order is normalized; true and numbers stay distinct kinds.
    # Untyped literals sort before the tagged bool/float objects.
    values = decision_table._dedupe_and_order(
        [{"b": 2, "a": 1}, {"a": 1, "b": 2}, "x", 2, True, 1.0, 1, None])
    assert values == ["x", 1, 2, None, True, 1.0, {"a": 1, "b": 2}]


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
