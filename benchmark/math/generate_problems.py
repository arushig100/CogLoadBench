#!/usr/bin/env python3
"""Generate the Math base problems: seeded arithmetic word problems over five domains.

Each (domain, arithmetic schema) cell gets num_per_cell problems. A problem is one
person and one item, a few lines that change the item count, and the question
"How many <item> does <name> have now?". The task runners read the output as the
base-problem pool (fields s1, s2, ..., question_only, answer, domain, structure).

    python -m benchmark.math.generate_problems --output-dir outputs/generated/math/baselines

The defaults reproduce benchmark/math/baselines/baseline.jsonl.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

# Domains and their 20-item pools.
DOMAIN_ITEMS = {
    "classroom": [
        "stickers", "pencils", "markers", "notebooks", "erasers",
        "crayons", "rulers", "folders", "books", "pens",
        "glue sticks", "paperclips", "stamps", "flashcards", "magnets",
        "ribbons", "beads", "buttons", "stars", "badges",
    ],
    "zoo": [
        "apples", "bananas", "carrots", "fish", "peanuts",
        "berries", "grapes", "biscuits", "pellets", "seeds",
        "leaves", "melons", "corn cobs", "hay bales", "crickets",
        "worms", "shrimp", "mealworms", "oranges", "lettuce heads",
    ],
    "kitchen": [
        "cookies", "cupcakes", "muffins", "pancakes", "sandwiches",
        "bagels", "rolls", "scones", "brownies", "donuts",
        "slices of bread", "crackers", "pretzels", "waffles", "biscuits",
        "tarts", "pies", "croissants", "dumplings", "pastries",
    ],
    "sports": [
        "points", "laps", "push-ups", "sit-ups", "jumping jacks",
        "free throws", "goals", "passes", "sprints", "stretches",
        "burpees", "squats", "lunges", "rebounds", "assists",
        "serves", "volleys", "catches", "throws", "kicks",
    ],
    "garden": [
        "seeds", "flowers", "tomatoes", "peppers", "carrots",
        "radishes", "beans", "peas", "herbs", "bulbs",
        "seedlings", "roses", "tulips", "daisies", "sunflowers",
        "strawberries", "cucumbers", "zucchinis", "potatoes", "onions",
    ],
}

# Arithmetic schemas. Each takes the drawn numbers, the person, the item, and the
# random generator, and returns the problem lines (one phrasing drawn per step),
# the question, and the answer.


def earn_spend_earn(nums, name, item, rng):
    """start + earn - spend + earn = answer"""
    start, earn1, spend, earn2 = nums
    answer = start + earn1 - spend + earn2
    starts = [
        f"{name} has {start} {item}.",
        f"{name} begins with {start} {item}.",
        f"{name} currently owns {start} {item}.",
        f"There are {start} {item} that belong to {name}.",
    ]
    gains1 = [
        f"{name} gets {earn1} more {item}.",
        f"{name} receives {earn1} additional {item}.",
        f"{name} earns {earn1} {item}.",
        f"Someone gives {name} {earn1} {item}.",
    ]
    losses = [
        f"{name} uses {spend} {item}.",
        f"{name} gives away {spend} {item}.",
        f"{name} spends {spend} {item}.",
        f"{name} loses {spend} {item}.",
    ]
    gains2 = [
        f"{name} receives {earn2} more {item}.",
        f"{name} collects another {earn2} {item}.",
        f"{name} finds {earn2} extra {item}.",
        f"{name} is given {earn2} more {item}.",
    ]
    lines = [rng.choice(starts), rng.choice(gains1), rng.choice(losses), rng.choice(gains2)]
    question = f"How many {item} does {name} have now?"
    return lines, question, answer


def fill_drain_refill(nums, name, item, rng):
    """start + fill - drain + refill = answer"""
    start, fill, drain, refill = nums
    answer = start + fill - drain + refill
    starts = [
        f"{name} starts with {start} {item}.",
        f"{name} has {start} {item} to begin with.",
        f"{name} begins the day with {start} {item}.",
        f"Initially, {name} has {start} {item}.",
    ]
    adds = [
        f"{name} adds {fill} {item}.",
        f"{name} puts in {fill} more {item}.",
        f"{name} brings in {fill} additional {item}.",
        f"{name} stocks up with {fill} {item}.",
    ]
    removes = [
        f"{name} removes {drain} {item}.",
        f"{name} takes out {drain} {item}.",
        f"{name} gets rid of {drain} {item}.",
        f"{drain} {item} are taken away from {name}.",
    ]
    refills = [
        f"{name} puts back {refill} {item}.",
        f"{name} restores {refill} {item}.",
        f"{name} adds {refill} {item} back.",
        f"Another {refill} {item} are returned to {name}.",
    ]
    lines = [rng.choice(starts), rng.choice(adds), rng.choice(removes), rng.choice(refills)]
    question = f"How many {item} does {name} have now?"
    return lines, question, answer


def buy_discount_fee(nums, name, item, rng):
    """start - (count * price) + discount - fee = answer"""
    start, count, price, discount, fee = nums
    answer = start - (count * price) + discount - fee
    starts = [
        f"{name} has {start} {item}.",
        f"{name} owns {start} {item}.",
        f"{name} starts out with {start} {item}.",
        f"At first, {name} has {start} {item}.",
    ]
    buys = [
        f"{name} gives away {count} batches of {price} {item} each.",
        f"{name} distributes {count} groups of {price} {item}.",
        f"{name} hands out {count} sets of {price} {item} each.",
        f"{name} shares {count} packs of {price} {item} each.",
    ]
    discounts = [
        f"{name} gets back {discount} {item}.",
        f"{name} recovers {discount} {item}.",
        f"Someone returns {discount} {item} to {name}.",
        f"{name} finds {discount} {item} that were misplaced.",
    ]
    fees = [
        f"{name} loses {fee} {item}.",
        f"{name} accidentally drops {fee} {item}.",
        f"{fee} {item} go missing from {name}.",
        f"{name} misplaces {fee} {item}.",
    ]
    lines = [rng.choice(starts), rng.choice(buys), rng.choice(discounts), rng.choice(fees)]
    question = f"How many {item} does {name} have now?"
    return lines, question, answer


# Schema name -> line builder and number sampler. Each sampler draws the numbers in the
# order its schema function unpacks them.
STRUCTURES = {
    "earn_spend_earn": {
        "fn": earn_spend_earn,
        "num_generator": lambda rng: (
            rng.randint(50, 200),   # start
            rng.randint(10, 80),    # earn1
            rng.randint(10, 60),    # spend
            rng.randint(10, 80),    # earn2
        ),
    },
    "fill_drain_refill": {
        "fn": fill_drain_refill,
        "num_generator": lambda rng: (
            rng.randint(100, 300),  # start
            rng.randint(20, 100),   # fill
            rng.randint(10, 80),    # drain
            rng.randint(10, 60),    # refill
        ),
    },
    "buy_discount_fee": {
        "fn": buy_discount_fee,
        "num_generator": lambda rng: (
            rng.randint(100, 300),  # start
            rng.randint(2, 5),      # count
            rng.randint(5, 20),     # price
            rng.randint(5, 30),     # discount
            rng.randint(3, 15),     # fee
        ),
    },
}

# Person names, assigned to problems in turn.
NAMES = [
    "Aiden", "Bella", "Carlos", "Diana", "Ethan", "Fiona", "Gavin", "Hana",
    "Isaac", "Julia", "Kai", "Luna", "Marco", "Nina", "Omar", "Priya",
    "Quinn", "Ravi", "Sara", "Tariq", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane", "Amara", "Ben", "Cleo", "Derek", "Elena", "Farah",
    "Grace", "Hugo", "Ivy", "Jake", "Kira", "Leo", "Maya", "Noah",
]


def generate_problems(num_per_cell: int, seed: int = 42) -> List[Dict]:
    """Generate seeded problems for every domain and arithmetic schema, in shuffled order."""
    rng = random.Random(seed)

    domains = sorted(DOMAIN_ITEMS.keys())
    struct_names = sorted(STRUCTURES.keys())

    problems = []
    name_idx = 0

    # Names cycle through NAMES across all cells; items are shuffled per domain and
    # cycle through the domain's pool, one item per problem.
    for domain in domains:
        items = list(DOMAIN_ITEMS[domain])
        rng.shuffle(items)
        item_idx = 0

        for struct_name in struct_names:
            struct = STRUCTURES[struct_name]

            for i in range(num_per_cell):
                name = NAMES[name_idx % len(NAMES)]
                name_idx += 1

                item = items[item_idx % len(items)]
                item_idx += 1

                # Redraw the numbers until the answer is positive.
                for _ in range(100):
                    nums = struct["num_generator"](rng)
                    lines, question, answer = struct["fn"](nums, name, item, rng=rng)
                    if answer > 0:
                        break

                # Skip the slot when every draw gives a non-positive answer.
                if answer <= 0:
                    continue

                # base_id is the domain, schema, and item; id adds the running problem index.
                problem_id = f"{domain}__{struct_name}__{item}__{len(problems)}"
                base_id = f"{domain}__{struct_name}__{item}"
                row = {
                    "id": problem_id,
                    "base_id": base_id,
                    "domain": domain,
                    "structure": struct_name,
                    "entity_name": name,
                    "item": item,
                    "num_steps": len(lines),
                    # Index in generation order, before the final shuffle.
                    "variant": len(problems),
                    "answer": answer,
                    "question_only": question,
                    "meta": {"domain": domain, "structure": struct_name, "item": item, "entity_name": name},
                }
                # Lines are stored as s1, s2, ...; the question follows as the last s-field.
                for j, line in enumerate(lines):
                    row[f"s{j+1}"] = line
                row[f"s{len(lines)+1}"] = question

                problems.append(row)

    # Shuffle so the file does not list problems cell by cell.
    rng.shuffle(problems)
    return problems


def main():
    """Write the generated problems and print their domain, schema, and item counts."""
    parser = argparse.ArgumentParser(description="Generate the Math base problems.")
    parser.add_argument("--num_per_cell", type=int, default=20, help="Problems per domain and arithmetic schema")
    parser.add_argument("--seed", type=int, default=42, help="Seed for problem sampling and output order")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "outputs/generated/math/baselines",
                        help="Directory for baseline.jsonl")
    args = parser.parse_args()
    # Each problem in a (domain, schema) cell uses a different item, so base IDs are unique.
    if not 1 <= args.num_per_cell <= min(len(items) for items in DOMAIN_ITEMS.values()):
        parser.error("--num_per_cell must be between 1 and the number of items per domain")

    problems = generate_problems(args.num_per_cell, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "baseline.jsonl"

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in problems:
            f.write(json.dumps(p) + '\n')

    print(f"Generated {len(problems)} problems -> {out_path}")
    print(f"  Domains: {sorted(set(p['domain'] for p in problems))}")
    print(f"  Structures: {sorted(set(p['structure'] for p in problems))}")

    items_by_domain = {}
    for p in problems:
        items_by_domain.setdefault(p['domain'], set()).add(p['item'])
    for d, items in sorted(items_by_domain.items()):
        print(f"  {d}: {len(items)} unique items")


if __name__ == "__main__":
    main()
