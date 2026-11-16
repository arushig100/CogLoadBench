"""Base-problem pools, distractor sampling, and prompt layout shared by the Math and Context tasks.

A prompt shows several base problems, one of which is the target; the target's
question is repeated at the end and its answer is the gold answer. ``ProblemPool``
chooses the problems for a variant, ``ProblemPrompts.build`` picks the sampling
scheme of each axis, and ``ProblemPrompts.layout`` renders the prompt with the
task-specific pieces in ``TaskFormat``.

Quantity, similarity, and underspecification share a fixed-target scheme: the
target and its relative position depend only on the variant seed, and the
distractors only on the seed, the target, and the similarity level, so
conditions of the same variant differ only in K or in the kind of distractors.
Similarity "low" and quantity "k10" therefore hold the same prompts.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .axes import Condition, similarity
from .common import add_interference, discontinuity_lines, read_jsonl, variant_seed

# One base problem as read from JSONL: id, base_id, domain, structure, lines s1, s2, ...,
# question_only, and answer.
Row = Dict[str, Any]

# Text placed before the repeated target question at the end of every prompt.
QUERY_LABEL = "\nQuestion to answer (repeated):\n"
# Offsets that give low- and high-similarity distractors their own random streams.
SEED_OFFSETS = {"low": 11, "high": 66}
# Distractors sampled per variant and similarity level; a K-problem prompt uses the first K-1.
STORED_DISTRACTORS = 9


def unique_by_base(rows: List[Row]) -> List[Row]:
    """Keep the smallest row ID per base problem and sort the result by row ID."""
    by_base: Dict[str, Row] = {}
    for row in rows:
        base = str(row.get("base_id", ""))
        if base and (base not in by_base or str(row.get("id", "")) < str(by_base[base].get("id", ""))):
            by_base[base] = row
    return sorted(by_base.values(), key=lambda row: str(row.get("id", "")))


class ProblemPool:
    """Base problems indexed by domain and structure, with optional replacement targets."""

    def __init__(self, path: str, target_variant_path: str = ""):
        """Read the base problems and, when given, the replacement targets keyed by base_id."""
        self.rows = read_jsonl(path)
        if not self.rows:
            raise ValueError(f"No problems in {path}")
        # Index rows by ID, by domain (high-similarity pools), and by
        # (domain, structure) cell (discontinuity groups).
        self.by_id: Dict[str, Row] = {}
        self.by_domain: Dict[str, List[Row]] = {}
        self.by_cell: Dict[str, Dict[str, List[Row]]] = {}
        for row in self.rows:
            domain, structure = row.get("domain", "unknown"), row.get("structure", "unknown")
            self.by_id[str(row.get("id", ""))] = row
            self.by_domain.setdefault(domain, []).append(row)
            self.by_cell.setdefault(domain, {}).setdefault(structure, []).append(row)
        self.target_variants = self._load_target_variants(target_variant_path) if target_variant_path else {}

    def _load_target_variants(self, path: str) -> Dict[str, Row]:
        """Index replacement targets by base_id; every base problem in the pool needs one."""
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"No target variants in {path}")
        variants: Dict[str, Row] = {}
        for row in rows:
            base = row.get("base_id", "")
            if not isinstance(base, str) or not base:
                raise ValueError(f"Target variant in {path} has no base_id")
            variants.setdefault(base, row)
        missing = {row["base_id"] for row in self.rows} - variants.keys()
        if missing:
            raise ValueError(f"{path} has no target variant for {len(missing)} base problems")
        return variants

    def plan_low(self, num: int, rng: random.Random) -> List[Row]:
        """Sample distinct base problems, usually from different (domain, structure) cells."""
        chosen: List[Row] = []
        seen = set()
        # The try limit ends the loop when the pool has too few distinct base problems for K.
        tries = 0
        while len(chosen) < min(num, len(self.rows)) and tries < 5000:
            tries += 1
            row = rng.choice(self.rows)
            base = row.get("base_id", "")
            if base in seen:
                continue
            # Reject a row from an already used cell half of the time, so the sample
            # spreads across cells without requiring every cell to be distinct.
            cells = {(c.get("domain", "unknown"), c.get("structure", "unknown")) for c in chosen}
            if (row.get("domain", "unknown"), row.get("structure", "unknown")) in cells and rng.random() < 0.5:
                continue
            chosen.append(row)
            seen.add(base)
        if len(chosen) < num:
            raise ValueError(f"Not enough distinct base problems for K={num}")
        return chosen

    def grouped_rows(self, rng: random.Random) -> List[Row]:
        """Sample nine problems as three groups of three that share a (domain, structure) cell."""
        # Only cells with at least three distinct base problems can supply a group.
        cells = [(d, s) for d in self.by_cell for s in self.by_cell[d]
                 if len({row["base_id"] for row in self.by_cell[d][s]}) >= 3]
        if not cells:
            raise ValueError("Discontinuity needs a (domain, structure) cell with three base problems")
        groups: List[List[Row]] = []
        attempts = 0
        while len(groups) < 3 and attempts < 500:
            attempts += 1
            d, s = rng.choice(cells)
            unique: Dict[str, Row] = {}
            for row in self.by_cell[d][s]:
                unique.setdefault(row["base_id"], row)
            chosen = rng.sample(list(unique.values()), 3)
            # Usually skip a cell that already supplied a group, so the three groups
            # tend to come from different cells.
            if any((g[0]["domain"], g[0]["structure"]) == (d, s) for g in groups) and len(cells) > 3 and rng.random() < 0.7:
                continue
            groups.append(chosen)
        if len(groups) < 3:
            raise ValueError("Could not sample three discontinuity groups")
        # Flatten group by group, so rows 0-2, 3-5, and 6-8 each share a cell.
        return sum(groups, [])

    def target(self, seed: int) -> Tuple[Row, float]:
        """Choose the target of a variant and its relative position in the context."""
        # A separate stream keeps the target independent of K and of the distractor sampling.
        rng = random.Random(int(seed) + 777777)
        # Sorting by ID makes the choice depend only on the seed, not on the file order.
        candidates = sorted((row for row in self.rows if str(row.get("question_only", "")).strip()),
                            key=lambda row: str(row.get("id", "")))
        if not candidates:
            raise ValueError("No base problem has a question")
        target = rng.choice(candidates)
        position = rng.random()
        # The underspecified condition swaps in the variant of the chosen base problem.
        return self.target_variants.get(target.get("base_id", ""), target), position

    def distractor_ids(self, sim: str, seed: int, target: Row) -> List[str]:
        """Sample the variant's stored distractors; a condition with K problems uses the first K-1."""
        rng = random.Random(int(seed) + SEED_OFFSETS[sim])
        base = str(target.get("base_id", ""))
        # High-similarity distractors share the target's domain but use a different
        # structure; low-similarity distractors come from the whole pool.
        if sim == "high":
            pool = [row for row in self.by_domain.get(target.get("domain", "unknown"), [])
                    if row.get("structure") != target.get("structure", "unknown")]
        else:
            pool = self.rows
        pool = unique_by_base([row for row in pool if str(row.get("base_id", "")) != base])
        # Top up a small pool with other base problems in ID order.
        if len(pool) < STORED_DISTRACTORS:
            seen = {str(row.get("base_id", "")) for row in pool}
            for row in unique_by_base([row for row in self.rows if str(row.get("base_id", "")) != base]):
                if str(row.get("base_id", "")) not in seen:
                    pool.append(row)
                    seen.add(str(row.get("base_id", "")))
                if len(pool) >= STORED_DISTRACTORS:
                    break
        # Return IDs in sampled order; the first K-1 form the distractors of a K-problem prompt.
        return [str(row["id"]) for row in rng.sample(pool, min(STORED_DISTRACTORS, len(pool)))]

    def fixed_target_rows(self, k: int, sim: str, seed: int) -> Tuple[List[Row], int]:
        """Return K problems with the variant's target at its position; extra slots take base problems in ID order."""
        target, position = self.target(seed)
        # Place the target at the same relative depth for every K.
        target_idx = max(0, min(k - 1, int(math.floor(position * float(k)))))
        distractors = [self.by_id[i] for i in self.distractor_ids(sim, seed, target)[:max(0, k - 1)]]
        rows: List[Optional[Row]] = [None] * k
        rows[target_idx] = target
        slots = [i for i in range(k) if i != target_idx]
        for i, row in zip(slots, distractors):
            rows[i] = row
        # K above STORED_DISTRACTORS + 1 leaves empty slots; at either similarity level,
        # fill them with unused base problems in ID order.
        if any(row is None for row in rows):
            used = {str(row.get("base_id", "")) for row in rows if row is not None}
            fill = [row for row in unique_by_base([r for r in self.rows if str(r.get("base_id", "")) != str(target.get("base_id", ""))])
                    if str(row.get("base_id", "")) not in used]
            for i in range(k):
                if rows[i] is None:
                    if not fill:
                        raise ValueError(f"Not enough base problems for K={k}")
                    rows[i] = fill.pop(0)
        return rows, target_idx


class TaskFormat(NamedTuple):
    """Task-specific prompt pieces for Math or Context."""
    header: Callable[[int], str]
    render: Callable[[Row, int, Optional[Row]], List[str]]
    format_block: str
    material: str


def target_ids(target: Row) -> Dict[str, str]:
    """The target problem's ID and its base problem's ID, stored with each prompt."""
    missing = [field for field in ("id", "base_id") if target.get(field) is None]
    if missing:
        raise ValueError(f"problem row is missing {', '.join(missing)}: {target}")
    return {"target_id": str(target["id"]), "target_base_id": str(target["base_id"])}


class ProblemPrompts:
    """Build Math or Context prompts and answers for every condition."""

    def __init__(self, task_format: TaskFormat, args):
        """Store the task format, the seed, and the input file of each pool; pools load on first use."""
        self.format = task_format
        self.seed = args.seed
        # Every axis except underspecification uses the main base file. Both
        # underspecification conditions draw from the same base file and differ only
        # in the target, which the underspecified condition replaces with its variant.
        self.paths = {
            "main": (args.baseline_file, ""),
            "baseline": (args.underspec_baseline_file, ""),
            "underspecified": (args.underspec_baseline_file, args.underspec_variant_file),
        }
        self.pools: Dict[str, ProblemPool] = {}

    def pool(self, condition: Condition) -> ProblemPool:
        """Return the base-problem pool of a condition."""
        key = condition.name if condition.axis == "underspecification" else "main"
        if key not in self.pools:
            self.pools[key] = ProblemPool(*self.paths[key])
        return self.pools[key]

    def build(self, condition: Condition, variant: int) -> Tuple[str, str, Dict[str, str]]:
        """Return the prompt, gold answer, and target IDs of one condition and variant."""
        seed = variant_seed(self.seed, variant)
        rng = random.Random(seed)
        pool = self.pool(condition)
        # Interference: K low-similarity problems with a random target, followed by
        # the secondary task, which replaces the answer format.
        if condition.axis == "interference":
            rows = pool.plan_low(condition.k, rng)
            candidates = [i for i, row in enumerate(rows)
                          if isinstance(row.get("question_only"), str) and row["question_only"].strip()]
            if not candidates:
                raise ValueError("No sampled problem has a question")
            target_idx = rng.choice(candidates)
            prompt, answer = self.layout(rows, target_idx)
            prompt = add_interference(prompt, condition.name, self.format.format_block, self.format.material)
            return prompt, answer, target_ids(rows[target_idx])
        # Discontinuity: nine problems in three same-cell groups, drawn identically for
        # all three orderings. The target index comes from its own stream, and the
        # remaining draws of rng set the line order.
        if condition.axis == "discontinuity":
            rows = pool.grouped_rows(rng)
            # XOR with a fixed constant gives the target index a stream separate from rng.
            target_idx = random.Random(seed ^ 0xCAFEBABE).randrange(9)
            return (*self.layout(rows, target_idx, condition.name, rng), target_ids(rows[target_idx]))
        # Quantity, similarity, and underspecification: the fixed-target scheme.
        rows, target_idx = pool.fixed_target_rows(condition.k, similarity(condition), seed)
        return (*self.layout(rows, target_idx), target_ids(rows[target_idx]))

    def layout(self, rows: List[Row], target_idx: int, order: str = "",
               rng: Optional[random.Random] = None) -> Tuple[str, str]:
        """Render the problems, the repeated target question, and the answer format."""
        target = rows[target_idx]
        # Distractors are rendered with the target passed in, so the task format can
        # adapt them to it; the target itself is rendered as stored.
        blocks = [self.format.render(row, i + 1, None if i == target_idx else target) for i, row in enumerate(rows)]
        lines = discontinuity_lines(blocks, order, rng) if order else [line for block in blocks for line in block]
        question = str(target.get("question_only", "")).strip()
        if not question:
            raise ValueError(f"Target {target.get('id', '')} has no question")
        # The header states the number of lines per problem, counted from the target's s1, s2, ... fields.
        steps = sum(1 for key in target if key.startswith("s") and key[1:].isdigit())
        # Prompt: header, problem lines, the repeated target question, then the answer format.
        parts = [self.format.header(steps)]
        if lines:
            parts.append("\n".join(lines) + "\n")
        parts += [QUERY_LABEL, question]
        return "\n".join(parts) + self.format.format_block, str(target.get("answer", "")).strip()
