"""Load axes, their conditions, and the number of problems (K) in each condition.

Each runner asks ``conditions`` for the conditions of an axis and builds one
prompt per (condition, variant). K is the number of problems (Math, Context) or
analogy chains shown in a prompt, one of which is the target.
"""
from __future__ import annotations

from typing import List, NamedTuple, Sequence

# The five load axes, in the order the runners process them for --axis all.
AXES = ("quantity", "similarity", "interference", "discontinuity", "underspecification")

# K values of the quantity axis; its distractors are not matched to the target's domain.
QUANTITY_K = {
    "math": (1, 5, 10, 20, 50),
    "context": (1, 5, 10, 20, 50),
    "analogy": (1, 5, 10, 20),
}
# K of the low and high similarity conditions, and the K values allowed for the
# extra high_k<K> conditions selected with --similarity_k.
SIMILARITY_K = {"math": 10, "context": 10, "analogy": 5}
SIMILARITY_SWEEP_K = {"math": (5, 20, 50), "context": (5, 20, 50), "analogy": (10, 20)}
# K of every interference condition and of both underspecification conditions.
INTERFERENCE_K = 10
UNDERSPECIFICATION_K = {"math": 10, "context": 10, "analogy": 5}

# Condition names in run order; interference "none" has no secondary task, and discontinuity
# "ordered" keeps each problem's lines together and in order.
INTERFERENCE_CONDITIONS = ("none", "memory", "counting", "arithmetic")
DISCONTINUITY_CONDITIONS = ("ordered", "local_shuffle", "global_shuffle")
UNDERSPECIFICATION_CONDITIONS = ("baseline", "underspecified")


class Condition(NamedTuple):
    """One load condition: its axis, its name, and K (None for discontinuity, whose prompts have a fixed size)."""
    axis: str
    name: str
    k: int | None


def conditions(task: str, axis: str, similarity_k: Sequence[int] = ()) -> List[Condition]:
    """Return the conditions of one axis in run order."""
    if axis == "quantity":
        return [Condition(axis, f"k{k}", k) for k in QUANTITY_K[task]]
    if axis == "similarity":
        # Low and high share the task's similarity K; each high_k<K> condition has its own K.
        k = SIMILARITY_K[task]
        return ([Condition(axis, "low", k), Condition(axis, "high", k)]
                + [Condition(axis, f"high_k{k}", k) for k in similarity_k])
    if axis == "interference":
        return [Condition(axis, name, INTERFERENCE_K) for name in INTERFERENCE_CONDITIONS]
    if axis == "discontinuity":
        return [Condition(axis, name, None) for name in DISCONTINUITY_CONDITIONS]
    if axis == "underspecification":
        return [Condition(axis, name, UNDERSPECIFICATION_K[task]) for name in UNDERSPECIFICATION_CONDITIONS]
    raise ValueError(f"Unknown axis: {axis}")


def similarity(condition: Condition) -> str:
    """Return 'high' for high-similarity distractors and 'low' otherwise."""
    return "high" if condition.axis == "similarity" and condition.name.startswith("high") else "low"

