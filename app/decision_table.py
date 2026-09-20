"""Decision tables: full Cartesian-product adjudication as a compressed tree.

Given a request mapping field names to non-empty domains (deduplicated arrays
of JSON literals), the fields are ordered lexicographically and every
assignment of their Cartesian product is adjudicated with the regular
priority/three-valued engine.  Each product leaf carries the *business
signature* of its verdict — the same signature used by replacement previews
(``preview.verdict_signature``): status, conclusion and decided priority,
plus conflicting conclusions or minimal missing-field sets.

The product is arranged into a decision tree following the field order, then
compressed bottom-up.  A node collapses to a single value-independent subtree
only when the compressed subtrees for **every** value in that field's domain
are completely identical:

* collapsing never crosses signatures — two leaves merge only when their full
  business signatures are equal under canonical JSON;
* partial agreement never merges — a node stays a value-branching split as
  long as any two of its children differ structurally.

Branches are ordered by a type-aware canonical JSON encoding of their
values (booleans and floats carry an explicit type tag), so the compressed
tree is stable and repeated identical requests are byte-identical.  The
store is only read.  The number of adjudicated combinations is capped via
the ``MAX_DECISION_TABLE_COMBINATIONS`` environment variable; exceeding the
cap aborts with :class:`DecisionTableSearchError` rather than truncating.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .counterfactual import canonical_json
from .engine import adjudicate
from .preview import verdict_signature

_DEFAULT_MAX_COMBINATIONS = 100_000


def _configured_max_combinations() -> int:
    raw = os.environ.get("MAX_DECISION_TABLE_COMBINATIONS")
    if raw is None:
        return _DEFAULT_MAX_COMBINATIONS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_COMBINATIONS
    return value if value > 0 else _DEFAULT_MAX_COMBINATIONS


# Safety cap on Cartesian-product combinations (one adjudication per leaf).
MAX_DECISION_TABLE_COMBINATIONS = _configured_max_combinations()


class DecisionTableSearchError(ValueError):
    """Raised when the Cartesian product exceeds the configured cap."""


def typed_literal_json(value: Any) -> str:
    """Canonical JSON encoding that tags the JSON number kinds.

    Plain canonical JSON already renders ``true`` and ``1`` differently, but
    the distinction between an integer and a float that happen to share a
    textual form (and ordering ties like ``1`` versus the booleans) is left
    implicit.  Tagging booleans and floats with an explicit wrapper makes
    domain-value identity and ordering total, stable and independent of
    number formatting.
    """

    def tagged(node: Any) -> Any:
        if isinstance(node, bool):
            return {"__type__": "bool", "value": node}
        if isinstance(node, float):
            return {"__type__": "float", "value": node}
        if isinstance(node, list):
            return [tagged(item) for item in node]
        if isinstance(node, dict):
            return {key: tagged(item) for key, item in node.items()}
        return node

    return canonical_json(tagged(value))


def _canonical_literal(value: Any) -> Any:
    """Re-order object keys canonically, keeping scalar values untouched.

    A ``json.loads`` round-trip would coerce ``1.0`` to ``1``; rebuilding the
    literal recursively normalizes only object key order so downstream
    serialization is key-order independent.
    """
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, list):
        return [_canonical_literal(item) for item in value]
    return {
        key: _canonical_literal(value[key])
        for key in sorted(value)
    }


def _dedupe_and_order(values: list[Any]) -> list[Any]:
    """Drop values equal under type-aware canonical JSON, then order them.

    Request validation rejects duplicates outright; this additionally makes
    the emitted value set independent of the request's array order and of
    object-key insertion order.
    """
    unique: dict[str, Any] = {}
    for value in values:
        unique.setdefault(typed_literal_json(value), value)
    ordered = [unique[key] for key in sorted(unique)]
    return [_canonical_literal(value) for value in ordered]


def build_decision_table(
    policy: dict,
    domains: dict[str, list[Any]],
    max_combinations: Optional[int] = None,
) -> tuple[list[str], int, dict]:
    """Adjudicate the full Cartesian product and compress the decision tree.

    ``domains`` maps field name to its value array (validated upstream).
    Returns ``(fields, total_combinations, tree)`` with fields in
    lexicographic order and values ordered by type-aware canonical JSON.
    Tree nodes are ``{"type": "leaf", "signature"}``,
    ``{"type": "split", "field", "branches": [{"value", "child"}, ...]}`` or
    ``{"type": "uniform", "field", "child"}`` for a node whose subtree is
    identical for every value of ``field``.
    """
    if max_combinations is None:
        max_combinations = MAX_DECISION_TABLE_COMBINATIONS

    fields = sorted(domains)
    ordered_domains = {
        field: _dedupe_and_order(domains[field]) for field in fields
    }

    total_combinations = 1
    for values in ordered_domains.values():
        total_combinations *= len(values)
    if total_combinations > max_combinations:
        raise DecisionTableSearchError(
            "decision table combination space exceeds "
            f"{max_combinations} combinations ({total_combinations} required); "
            "narrow the domains or raise MAX_DECISION_TABLE_COMBINATIONS"
        )

    depth = len(fields)
    # Prefixes are tuples of domain *indices* so dict/list values stay usable
    # as dict keys.  The tree is built and compressed post-order
    # (children before parents) with an explicit task stack.
    nodes: dict[tuple[int, ...], dict] = {}
    tasks: list[tuple[str, tuple[int, ...]]] = [("expand", ())]

    while tasks:
        action, prefix = tasks.pop()

        if action == "leaf":
            facts = {
                fields[i]: ordered_domains[fields[i]][prefix[i]]
                for i in range(depth)
            }
            signature = verdict_signature(adjudicate(policy, facts))
            nodes[prefix] = {"type": "leaf", "signature": signature}
            continue

        level = len(prefix)
        field = fields[level]
        values = ordered_domains[field]
        if action == "expand":
            tasks.append(("finish", prefix))
            for index in range(len(values) - 1, -1, -1):
                if level + 1 == depth:
                    tasks.append(("leaf", prefix + (index,)))
                else:
                    tasks.append(("expand", prefix + (index,)))
            continue

        # Finish: all children have already been built and compressed; fold
        # into one subtree only when every per-value subtree is identical.
        children = [nodes[prefix + (index,)] for index in range(len(values))]
        # Type-aware encoding of the already-canonicalized nodes is an exact
        # structural identity (otherwise 1/1.0 branches could collide): leaves
        # merge only on equal signatures and split nodes only when every
        # branch is identical subtree-for-subtree.
        identities = {typed_literal_json(child) for child in children}
        if len(identities) == 1:
            nodes[prefix] = {
                "type": "uniform",
                "field": field,
                "child": children[0],
            }
        else:
            nodes[prefix] = {
                "type": "split",
                "field": field,
                "branches": [
                    {"value": values[index], "child": children[index]}
                    for index in range(len(values))
                ],
            }

    return fields, total_combinations, nodes[()]
