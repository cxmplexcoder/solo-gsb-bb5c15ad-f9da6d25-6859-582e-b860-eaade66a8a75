"""Decision tables: the policy's business signature over a field-domain grid.

A decision-table request submits one finite domain (a list of distinct JSON
literals) per field.  The fields are ordered lexicographically and every
combination of the cartesian product is adjudicated with the regular
priority/three-valued engine; each verdict is distilled into the *business
signature* used by replacement previews (``preview.verdict_signature``).

The signatures are arranged into a decision tree that branches on the fields
in that lexicographic order, then compressed bottom-up: a node collapses into
a single subtree only when the subtrees reachable through *every* value of
its field domain are completely identical.  Partial agreement never merges —
a node keeps branching as long as any two domain values lead to different
subtrees, and leaves with different signatures are never folded together.

Branches are emitted sorted by the canonical JSON encoding of their domain
value, so the response is independent of the order in which the domains were
declared.  The number of adjudicated combinations is capped
(``MAX_DECISION_TABLE_COMBINATIONS``, configurable via environment
variable); exceeding the cap aborts with an error rather than truncating the
table.  The policy store is only read, never modified.
"""
from __future__ import annotations

import os
from itertools import product
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


# Safety cap on adjudicated domain combinations.
MAX_DECISION_TABLE_COMBINATIONS = _configured_max_combinations()


class DecisionTableSearchError(ValueError):
    """Raised when the combination space exceeds the configured cap."""


def _order_values(values: list[Any]) -> list[Any]:
    """Order domain values by canonical JSON (already de-duplicated)."""
    return sorted(values, key=canonical_json)


def _leaf_key(signature: dict) -> tuple:
    return ("leaf", canonical_json(signature))


def _node_key(field: str, values: list[Any], child_keys: list[tuple]) -> tuple:
    return (
        "node",
        field,
        tuple(
            (canonical_json(value), child_key)
            for value, child_key in zip(values, child_keys)
        ),
    )


def _compress_tree(
    fields: list[str],
    ordered_values: dict[str, list[Any]],
    signatures: list[dict],
) -> dict:
    """Fold leaf signatures into the bottom-up compressed decision tree.

    ``signatures`` holds one leaf per combination in ``itertools.product``
    order over the ordered domains (the last field varies fastest).  Folding
    proceeds one field at a time from the last: each consecutive group of
    ``len(domain)`` sibling subtrees collapses into its shared subtree when
    every member is identical, otherwise a branching node is emitted.
    """
    # (key, node) per combination; keys make sibling comparison O(1).
    level_nodes = [
        (_leaf_key(signature), {"signature": signature})
        for signature in signatures
    ]
    for field in reversed(fields):
        values = ordered_values[field]
        fanout = len(values)
        grouped: list[tuple[tuple, dict]] = []
        for start in range(0, len(level_nodes), fanout):
            chunk = level_nodes[start:start + fanout]
            child_keys = [key for key, _ in chunk]
            first_key = child_keys[0]
            if all(key == first_key for key in child_keys):
                # Identical subtrees for every domain value: the field
                # discriminates nothing here, so the node folds away.
                grouped.append((first_key, chunk[0][1]))
                continue
            branches = [
                {"value": value, "node": node}
                for value, (_, node) in zip(values, chunk)
            ]
            grouped.append(
                (
                    _node_key(field, values, child_keys),
                    {"field": field, "branches": branches},
                )
            )
        level_nodes = grouped
    (_, tree), = level_nodes
    return tree


def decision_table(
    policy: dict,
    domains: dict[str, list],
    max_combinations: Optional[int] = None,
) -> dict:
    """Adjudicate every domain combination and compress the result tree.

    ``domains`` maps field names to non-empty lists of distinct JSON
    literals (already validated).  Returns ``{"fields", "total_combinations",
    "tree"}`` where ``fields`` is the lexicographic field order and ``tree``
    the compressed decision tree: leaves are ``{"signature": ...}`` and
    internal nodes ``{"field": ..., "branches": [{"value", "node"}, ...]}``
    with branches sorted by the canonical JSON of the domain value.
    """
    if max_combinations is None:
        max_combinations = MAX_DECISION_TABLE_COMBINATIONS

    fields = sorted(domains)
    ordered_values = {field: _order_values(domains[field]) for field in fields}

    total = 1
    for field in fields:
        total *= len(ordered_values[field])
        if total > max_combinations:
            raise DecisionTableSearchError(
                "decision table combination space exceeds "
                f"{max_combinations} combinations; reduce the domain sizes "
                "or raise MAX_DECISION_TABLE_COMBINATIONS"
            )

    signatures = [
        verdict_signature(adjudicate(policy, dict(zip(fields, combination))))
        for combination in product(*(ordered_values[f] for f in fields))
    ]
    tree = _compress_tree(fields, ordered_values, signatures)
    return {
        "fields": fields,
        "total_combinations": total,
        "tree": tree,
    }
