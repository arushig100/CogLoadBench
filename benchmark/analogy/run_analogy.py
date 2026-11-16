#!/usr/bin/env python3
"""Run the CogLoadBench Analogy task (procedural analogies over 4 domains x 20 relations).

Each prompt shows several example chains. A chain takes one semantic relation
(e.g. country_to_capital), gives each left entity a symbolic token, and shows
"left : token" and "right : rule(token)" for three pairs, where rule is one of
three string transformations. The query gives a new pair of the target chain's
relation and asks for the right side's code; the gold answer is the target rule
applied to the query's token. Unlike Math and Context, prompts are built entirely
from the tables in this file.

    python run.py benchmark analogy --model Qwen/Qwen3-4B --axis all
    python run.py benchmark analogy --dry_run --num_variants 200
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

# Make benchmark/shared importable as "shared" when this file is run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared import runner  # noqa: E402
from shared.axes import QUANTITY_K, SIMILARITY_K, SIMILARITY_SWEEP_K, Condition, similarity  # noqa: E402
from shared.common import add_interference, discontinuity_lines, variant_seed  # noqa: E402

# A chain specification: (domain, relation, rule).
Triple = Tuple[str, str, str]

# Semantic relations: domain -> relation -> {left: right}.
DATA = {
    "geography": {
        "country_to_capital": {
            "france": "paris", "italy": "rome", "japan": "tokyo", "egypt": "cairo", "canada": "ottawa",
            "brazil": "brasilia", "germany": "berlin", "spain": "madrid"
        },
        "country_to_continent": {
            "france": "europe", "japan": "asia", "egypt": "africa", "canada": "north america", "brazil": "south america",
            "australia": "oceania", "germany": "europe", "spain": "europe"
        },
        "us_city_to_state": {
            "los angeles": "california", "chicago": "illinois", "miami": "florida", "seattle": "washington",
            "boston": "massachusetts", "denver": "colorado", "phoenix": "arizona", "atlanta": "georgia"
        },
        "country_to_primary_language": {
            "france": "french", "japan": "japanese", "brazil": "portuguese", "egypt": "arabic", "germany": "german"
        },
        "major_river_to_continent": {
            "nile": "africa", "amazon": "south america", "yangtze": "asia", "danube": "europe", "mississippi": "north america"
        },
        "country_to_national_animal": {
            "australia": "kangaroo", "china": "giant panda", "india": "tiger", "scotland": "unicorn", "spain": "bull"
        },
        "country_to_currency": {
            "japan": "yen", "india": "rupee", "brazil": "real", "mexico": "peso", "china": "yuan"
        },
        "country_to_calling_code": {
            "japan": "eighty one", "india": "ninety one", "brazil": "fifty five", "mexico": "fifty two", "egypt": "twenty"
        },
        "ocean_to_bordering_continent": {
            "atlantic": "europe", "pacific": "asia", "indian": "africa", "arctic": "north america", "southern": "antarctica"
        },
        "country_to_flag_color": {
            "japan": "white", "france": "blue", "brazil": "green", "china": "red", "india": "orange"
        },
        "city_to_country": {
            "paris": "france", "tokyo": "japan", "rome": "italy", "cairo": "egypt", "berlin": "germany"
        },
        "country_to_food": {
            "italy": "pasta", "japan": "sushi", "mexico": "taco", "india": "curry", "china": "rice"
        },
        "continent_to_animal": {
            "africa": "lion", "australia": "kangaroo", "asia": "panda", "antarctica": "penguin", "europe": "fox"
        },
        "language_to_country": {
            "french": "france", "japanese": "japan", "portuguese": "brazil", "arabic": "egypt", "german": "germany"
        },
        "landmark_to_city": {
            "colosseum": "rome", "big ben": "london", "taj mahal": "agra", "pyramids": "cairo", "eiffel tower": "paris"
        },
        "country_to_greeting": {
            "france": "bonjour", "japan": "konnichiwa", "spain": "hola", "germany": "guten tag", "italy": "ciao"
        },
        "country_to_sport": {
            "brazil": "soccer", "japan": "sumo", "india": "cricket", "china": "ping pong", "usa": "basketball"
        },
        "country_to_religion": {
            "india": "hinduism", "israel": "judaism", "italy": "catholicism", "saudi arabia": "islam", "thailand": "buddhism"
        },
        "us_state_to_city": {
            "texas": "houston", "florida": "orlando", "nevada": "las vegas", "hawaii": "honolulu", "new york": "buffalo"
        },
        "country_to_car_brand": {
            "germany": "bmw", "japan": "toyota", "italy": "ferrari", "korea": "hyundai", "usa": "ford"
        }
    },
    "animals": {
        "parent_to_child": {
            "dog": "puppy", "cat": "kitten", "cow": "calf", "goat": "kid", "horse": "foal", "sheep": "lamb",
            "duck": "duckling", "goose": "gosling"
        },
        "predator_to_prey": {
            "lion": "zebra", "wolf": "rabbit", "eagle": "mouse", "fox": "hen", "owl": "vole", "shark": "seal", "snake": "frog"
        },
        "female_to_male": {
            "hen": "rooster", "cow": "bull", "doe": "buck", "mare": "stallion", "lioness": "lion"
        },
        "animal_to_sound": {
            "dog": "bark", "cat": "meow", "cow": "moo", "sheep": "baa", "lion": "roar"
        },
        "animal_to_classification": {
            "eagle": "bird", "shark": "fish", "frog": "amphibian", "snake": "reptile", "butterfly": "insect"
        },
        "animal_to_habitat": {
            "whale": "ocean", "eagle": "mountain", "camel": "desert", "penguin": "arctic", "monkey": "jungle"
        },
        "animal_to_diet": {
            "cow": "herbivore", "lion": "carnivore", "bear": "omnivore", "rabbit": "herbivore", "hawk": "carnivore"
        },
        "animal_to_legs": {
            "spider": "eight", "ant": "six", "dog": "four", "chicken": "two", "snake": "zero"
        },
        "animal_to_group": {
            "wolf": "pack", "fish": "school", "bird": "flock", "bee": "swarm", "lion": "pride"
        },
        "animal_to_home": {
            "bird": "nest", "bear": "cave", "bee": "hive", "rabbit": "burrow", "spider": "web"
        },
        "animal_to_covering": {
            "fish": "scales", "bird": "feathers", "turtle": "shell", "sheep": "wool", "hedgehog": "spines"
        },
        "animal_to_movement": {
            "fish": "swim", "bird": "fly", "snake": "slither", "kangaroo": "hop", "horse": "gallop"
        },
        "animal_to_color": {
            "flamingo": "pink", "crow": "black", "goldfish": "orange", "frog": "green", "dove": "white"
        },
        "animal_to_continent": {
            "kangaroo": "australia", "penguin": "antarctica", "panda": "asia", "giraffe": "africa", "eagle": "america"
        },
        "animal_to_product": {
            "cow": "milk", "sheep": "wool", "bee": "honey", "chicken": "egg", "silkworm": "silk"
        },
        "animal_to_food": {
            "rabbit": "carrot", "monkey": "banana", "panda": "bamboo", "cat": "fish", "mouse": "cheese"
        },
        "animal_to_similar": {
            "alligator": "crocodile", "bee": "wasp", "rabbit": "hare", "turtle": "tortoise", "frog": "toad"
        },
        "animal_to_body_part": {
            "elephant": "trunk", "giraffe": "neck", "kangaroo": "pouch", "rhino": "horn", "peacock": "tail"
        },
        "animal_to_defense": {
            "skunk": "spray", "turtle": "shell", "porcupine": "quills", "chameleon": "camouflage", "octopus": "ink"
        },
        "animal_to_role": {
            "vulture": "scavenger", "bee": "pollinator", "wolf": "predator", "deer": "prey", "earthworm": "decomposer"
        }
    },
    "colors": {
        "darker_to_lighter": {"red": "pink", "purple": "lavender", "black": "gray", "brown": "tan"},
        "complementary": {"red": "green", "blue": "orange", "yellow": "purple", "green": "red"},
        "wheel": {"red": "orange", "orange": "yellow", "yellow": "green", "green": "blue", "blue": "purple", "purple": "red"},
        "color_to_common_object": {"red": "apple", "green": "leaf", "blue": "sky", "yellow": "sunflower", "orange": "carrot"},
        "color_to_emotion": {"red": "angry", "blue": "sad", "yellow": "happy", "green": "jealousy", "purple": "mystery"},
        "color_to_temperature": {"red": "hot", "blue": "cold", "orange": "warm", "white": "cool", "yellow": "bright"},
        "color_to_season": {"green": "spring", "yellow": "summer", "orange": "autumn", "white": "winter", "brown": "fall"},
        "color_to_gemstone": {"red": "ruby", "blue": "sapphire", "green": "emerald", "purple": "amethyst", "white": "diamond"},
        "color_to_element": {"red": "fire", "blue": "water", "green": "earth", "white": "air", "yellow": "lightning"},
        "color_to_fruit": {"red": "apple", "yellow": "banana", "orange": "orange", "green": "lime", "purple": "grape"},
        "color_to_drink": {"brown": "coffee", "green": "tea", "red": "wine", "white": "milk", "orange": "juice"},
        "color_to_vegetable": {"orange": "carrot", "green": "broccoli", "red": "pepper", "white": "onion", "purple": "eggplant"},
        "color_to_animal": {"orange": "tiger", "black": "panther", "white": "dove", "pink": "flamingo", "gray": "elephant"},
        "color_to_spice": {"yellow": "turmeric", "red": "paprika", "brown": "cinnamon", "green": "basil", "black": "pepper"},
        "color_to_flag": {"red": "china", "green": "saudi arabia", "orange": "ireland", "blue": "greece", "white": "japan"},
        "color_to_sport_ball": {"white": "golf", "orange": "basketball", "yellow": "tennis", "green": "cricket", "brown": "football"},
        "color_to_nature": {"green": "grass", "blue": "ocean", "white": "snow", "yellow": "sun", "brown": "mud"},
        "color_to_shade": {"red": "maroon", "blue": "teal", "green": "mint", "yellow": "gold", "purple": "plum"},
        "color_to_app": {"blue": "facebook", "red": "youtube", "green": "whatsapp", "orange": "reddit", "yellow": "snapchat"},
        "color_to_brand": {"red": "coca cola", "yellow": "mcdonalds", "green": "starbucks", "brown": "ups", "purple": "cadbury"}
    },
    "numbers": {
        "add_1": {"3": "4", "10": "11", "-2": "-1", "99": "100", "0": "1"},
        "multiply_by_2": {"3": "6", "7": "14", "-4": "-8", "12": "24", "1": "2"},
        "subtract_5": {"12": "7", "5": "0", "3": "-2", "-7": "-12", "30": "25"},
        "square": {"2": "4", "3": "9", "5": "25", "-4": "16", "10": "100", "12": "144"},
        "cube": {"2": "8", "3": "27", "-2": "-8", "5": "125", "10": "1000"},
        "add_10": {"5": "15", "20": "30", "-3": "7", "0": "10", "50": "60"},
        "multiply_by_3": {"2": "6", "7": "21", "11": "33", "-3": "-9", "4": "12"},
        "double_then_add_1": {"2": "5", "4": "9", "6": "13", "10": "21", "15": "31"},
        "negate": {"5": "-5", "-3": "3", "1": "-1", "12": "-12", "-7": "7"},
        "halve": {"10": "5", "4": "2", "100": "50", "8": "4", "20": "10"},
        "add_5": {"1": "6", "8": "13", "15": "20", "22": "27", "30": "35"},
        "subtract_1": {"9": "8", "16": "15", "25": "24", "40": "39", "100": "99"},
        "multiply_by_4": {"3": "12", "5": "20", "6": "24", "10": "40", "25": "100"},
        "multiply_by_10": {"3": "30", "7": "70", "5": "50", "9": "90", "2": "20"},
        "add_100": {"5": "105", "20": "120", "50": "150", "1": "101", "10": "110"},
        "subtract_10": {"25": "15", "35": "25", "50": "40", "80": "70", "100": "90"},
        "divide_by_2": {"6": "3", "14": "7", "18": "9", "22": "11", "30": "15"},
        "square_root": {"9": "3", "16": "4", "25": "5", "36": "6", "49": "7"},
        "add_20": {"3": "23", "10": "30", "15": "35", "30": "50", "40": "60"},
        "multiply_by_5": {"2": "10", "4": "20", "6": "30", "8": "40", "10": "50"}
    }
}

# The symbolic transformations a chain can use; RULES fixes their order for sampling.
STRUCT_RULES = {
    "suffix_hash": lambda s: s + "#",
    "prefix_bang": lambda s: "!" + s,
    "wrap_brackets": lambda s: "[" + s + "]",
}
RULES = sorted(STRUCT_RULES)

# Two-letter symbolic tokens assigned to left entities.
BASE_TOKENS = [
    "ba", "be", "bi", "bo", "bu",
    "da", "de", "di", "do", "du",
    "fa", "fe", "fi", "fo", "fu",
    "ga", "ge", "gi", "go", "gu",
    "ha", "he", "hi", "ho", "hu",
]

# Opening instructions of every prompt; the example lines follow "Examples:".
PROMPT_HEADER = (
    "You will see analogy pairs in the format:\n"
    "  semantic_left : symbolic_code_left\n"
    "  semantic_right : symbolic_code_right\n\n"
    "Pairs that correspond are marked with the same label like [1.1] ... [1.1].\n"
    "Use labels only to link the two lines of a pair; ignore the numbers when inferring the rule.\n\n"
    "Your job: infer the symbolic transformation that maps the first code to the second\n"
    "within each semantic pair, then apply the SAME transformation to a new pair.\n\n"
    "The transformation will be ONE of the following patterns (do NOT invent others):\n"
    "  • prefix_bang   : s → !s\n"
    "  • suffix_hash   : s → s#\n"
    "  • wrap_brackets : s → [s]\n\n"
    "Examples:\n"
)
# Text before the query pair, and the answer placeholder of the JSON template.
QUERY_LABEL = "Now complete a new analogy using the same pattern:\n"
ANSWER_HINT = "<missing symbolic code>"
# Answer format at the end of every prompt; interference replaces it.
FORMAT_BLOCK = (
    "\nAnswer format (return JSON only). Do not write anything before or after the JSON.\n"
    "Begin your response on the next line with '{' and output a SINGLE JSON object:\n"
    '{"reasoning": "<brief explanation of reasoning behind answer>", "answer": "' + ANSWER_HINT + '"}'
)
# Instruction placed after the header's "Examples:" line in both underspecification conditions.
UNDERSPECIFICATION_INSTRUCTION = (
    "When multiple statements provide conflicting information about the same thing, "
    "use the LAST statement — earlier statements are outdated and should be ignored."
)
# Offsets that give low- and high-similarity distractors their own random streams.
SEED_OFFSETS = {"low": 11, "high": 66}
# Example pairs shown per chain.
SHOTS = 3


def stable_hash_int(text: str) -> int:
    """Return a process-independent integer hash of a string: the first 32 bits of its MD5."""
    # The built-in hash() of a string changes between processes, so it is not used here.
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def sorted_pairs(pairs: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return the (left, right) pairs of a relation in a fixed order."""
    return sorted(pairs.items(), key=lambda kv: (str(kv[0]), str(kv[1])))


def assign_tokens(entities: Iterable[str], rng: random.Random) -> Dict[str, str]:
    """Map each distinct entity to a shuffled symbolic token."""
    tokens = BASE_TOKENS.copy()
    rng.shuffle(tokens)
    mapping: Dict[str, str] = {}
    for entity in entities:
        if entity not in mapping:
            mapping[entity] = tokens[len(mapping) % len(tokens)]
    return mapping


def choose_query_pair(domain: str, relation: str, used: List[Tuple[str, str]], seed: int) -> Tuple[str, str]:
    """Choose an unshown pair of the relation, preferring a left entity not shown in the examples."""
    # The query pair has its own seed, so callers can fix it independently of the scene.
    rng = random.Random(int(seed))
    pairs = sorted_pairs(DATA[domain][relation])
    used_left = {a for a, _ in used}
    used = set(used)
    candidates = [(a, b) for (a, b) in pairs if (a, b) not in used]
    if not candidates:
        raise ValueError(f"{domain}/{relation}: every pair is already shown")
    fresh = [(a, b) for (a, b) in candidates if a not in used_left]
    return rng.choice(fresh if fresh else candidates)


def build_chain(domain: str, relation: str, rule: str, rng: random.Random, chain_index: int,
                underspecified: bool = False) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str]]]:
    """Return the labeled example lines of one chain, its token map, and its example pairs."""
    pairs = rng.sample(sorted_pairs(DATA[domain][relation]), SHOTS)
    tokens = assign_tokens([a for a, _ in pairs], rng)
    # In an underspecified chain only one pair, the anchor, shows its code; the
    # others show "?" on the right side.
    anchor = rng.randrange(len(pairs)) if underspecified else -1
    lines = []
    for i, (a, b) in enumerate(pairs):
        label = f"[{chain_index + 1}.{i + 1}] "
        # Each pair is two lines with the same label: left entity and token, right entity and code.
        code = STRUCT_RULES[rule](tokens[a])
        lines.append(f"{label}{a} : {tokens[a]}")
        lines.append(f"{label}{b} : {'?' if underspecified and i != anchor else code}")
    return lines, tokens, pairs


def all_bases() -> List[Tuple[str, str]]:
    """Return every (domain, relation) in a fixed order."""
    return [(d, rel) for d in sorted(DATA) for rel in sorted(DATA[d])]


def low_similarity_chains(num_chains: int, rng: random.Random) -> List[Triple]:
    """Draw chains with distinct (domain, relation) pairs and random rules."""
    domains = sorted(DATA)
    used: Dict[str, set] = {}
    chosen: List[Triple] = []
    attempts = 0
    while len(chosen) < num_chains and attempts < 2000:
        attempts += 1
        domain = rng.choice(domains)
        candidates = [r for r in sorted(DATA[domain]) if r not in used.get(domain, set())]
        if not candidates:
            continue
        relation = rng.choice(candidates)
        used.setdefault(domain, set()).add(relation)
        chosen.append((domain, relation, rng.choice(RULES)))
    if len(chosen) < num_chains:
        raise ValueError(f"Not enough distinct relations for {num_chains} chains")
    return chosen


def make_scene(chains: List[Triple], rng: random.Random, token_seed: int, *, target_idx: Optional[int] = None,
               chain_seeds: Optional[List[int]] = None, query_seed: Optional[int] = None, order: str = "",
               underspecified: bool = False, instruction: str = "") -> Tuple[str, str, int]:
    """Return the prompt, the gold answer, and the seed of the query pair."""
    if target_idx is None:
        target_idx = rng.randrange(len(chains))
    # Each chain draws its pairs and tokens from its own seed: a given seed, or one drawn from rng.
    blocks = []
    for i, (domain, relation, rule) in enumerate(chains):
        chain_rng = random.Random(int(chain_seeds[i]) if chain_seeds else int(rng.randrange(10**9)))
        blocks.append(build_chain(domain, relation, rule, chain_rng, i, underspecified and i == target_idx))
    line_blocks = [lines for lines, _, _ in blocks]
    lines = discontinuity_lines(line_blocks, order, rng) if order else [line for block in line_blocks for line in block]

    domain, relation, rule = chains[target_idx]
    _, tokens, used_pairs = blocks[target_idx]
    if query_seed is None:
        query_seed = int(rng.randrange(10**9))
    query_left, query_right = choose_query_pair(domain, relation, used_pairs, query_seed)
    # A query entity not shown in the target chain gets a token from a stable hash.
    if query_left not in tokens:
        tokens[query_left] = BASE_TOKENS[stable_hash_int(f"{int(token_seed)}::{query_left}") % len(BASE_TOKENS)]
    source = tokens[query_left]

    # Prompt: header (and instruction), example lines, the query pair, then the answer format.
    parts = [PROMPT_HEADER + ("\n\n" + instruction if instruction else "")]
    if lines:
        parts.append("\n".join(lines) + "\n")
    parts += [QUERY_LABEL, f"{query_left} : {source}", f"{query_right} : ?"]
    # The gold answer is the target chain's rule applied to the query token.
    return "\n".join(parts) + FORMAT_BLOCK, STRUCT_RULES[rule](source), query_seed


def fixed_target(seed: int) -> Tuple[Triple, float, int]:
    """Choose the target chain of a variant, its relative position, and its chain seed."""
    # A separate stream keeps the target independent of K and of the distractor sampling.
    rng = random.Random(int(seed) + 777777)
    # Every (domain, relation, rule) triple is a possible target.
    candidates = sorted((d, rel, rule) for (d, rel) in all_bases() for rule in RULES)
    target = rng.choice(candidates)
    position = float(rng.random())
    return target, position, int(rng.randrange(10**9))


def distractor_plan(sim: str, seed: int, target: Triple, size: int) -> Tuple[List[Triple], List[int]]:
    """Sample size-1 distractor chains, one rule per (domain, relation); a condition uses the first K-1."""
    offset = int(seed) + SEED_OFFSETS[sim]
    # rng samples the chains; seed_rng draws each chain's own seed from a separate stream.
    rng, seed_rng = random.Random(offset), random.Random(offset + 123456)
    t_domain, t_relation, t_rule = target
    # High-similarity distractors use other relations of the target's domain with a
    # different rule; low-similarity distractors use any other relation and any rule.
    if sim == "high":
        bases = [(t_domain, rel) for rel in sorted(DATA[t_domain]) if rel != t_relation]
        rules = [r for r in RULES if r != t_rule]
    else:
        bases = [(d, rel) for (d, rel) in all_bases() if (d, rel) != (t_domain, t_relation)]
        rules = RULES
    # Each (domain, relation) gets one rule, chosen by a stable hash of the variant's offset.
    pool = sorted({(d, rel, rules[stable_hash_int(f"{offset + 555555}::{d}::{rel}") % len(rules)])
                   for (d, rel) in sorted(set(bases))})
    chains = rng.sample(pool, min(size - 1, len(pool)))
    return chains, [int(seed_rng.randrange(10**9)) for _ in chains]


def fixed_target_scene(k: int, sim: str, seed: int, size: int) -> Tuple[str, str]:
    """Return a K-chain prompt with the variant's target at its relative position."""
    target, position, target_seed = fixed_target(seed)
    # Place the target at the same relative depth for every K.
    target_idx = max(0, min(k - 1, int(math.floor(position * float(k)))))
    distractors, seeds = distractor_plan(sim, seed, target, size)
    if len(distractors) < k - 1:
        raise ValueError(f"Not enough distractor chains for K={k}")
    chains, chain_seeds = list(distractors[:k - 1]), list(seeds[:k - 1])
    chains.insert(target_idx, target)
    chain_seeds.insert(target_idx, target_seed)
    # The query seed depends only on the variant seed, so every K and similarity level asks
    # the same query. With the target index and all seeds given, make_scene draws nothing from rng.
    prompt, answer, _ = make_scene(chains, random.Random(seed), seed, target_idx=target_idx,
                                   chain_seeds=chain_seeds, query_seed=int(seed) + 666666)
    return prompt, answer


def plan_size(condition: Condition) -> int:
    """Return the number of chains, including the target, in the plan a condition draws from.

    This is the largest quantity K for quantity, the largest swept K for high_k<K>, and the
    similarity K otherwise. A larger K takes more chains from the same plan, so it only adds chains.
    """
    if condition.axis == "quantity":
        return max(QUANTITY_K["analogy"])
    if condition.name.startswith("high_k"):
        return max(SIMILARITY_SWEEP_K["analogy"])
    return SIMILARITY_K["analogy"]


def discontinuity_scene(condition: str, seed: int) -> Tuple[str, str]:
    """Return four chains with distinct relations; all orderings ask the same query as the ordered prompt."""
    # A separate stream chooses the four chains, their rules, and their seeds.
    rng = random.Random(int(seed) + 303030)
    bases = all_bases()
    chains = [(d, rel, rng.choice(RULES)) for (d, rel) in rng.sample(bases, 4)]
    chain_seeds = [int(rng.randrange(10**9)) for _ in range(4)]
    # Build the ordered prompt once to obtain its query seed, then reuse that seed so
    # every ordering asks the same query; the target index is the first draw from
    # random.Random(seed), so both calls pick the same target chain.
    _, _, query_seed = make_scene(chains, random.Random(seed), seed, chain_seeds=chain_seeds, order="ordered")
    prompt, answer, _ = make_scene(chains, random.Random(seed), seed, chain_seeds=chain_seeds,
                                   query_seed=query_seed, order=condition)
    return prompt, answer


def build(args):
    """Return the prompt builder for the parsed arguments."""
    def build_prompt(condition: Condition, variant: int) -> Tuple[str, str]:
        """Return the prompt and gold answer of one condition and variant."""
        seed = variant_seed(args.seed, variant)
        rng = random.Random(seed)
        # Interference: K low-similarity chains with a random target, followed by the secondary task.
        if condition.axis == "interference":
            prompt, answer, _ = make_scene(low_similarity_chains(condition.k, rng), rng, seed)
            return add_interference(prompt, condition.name, FORMAT_BLOCK,
                                    "the analogy examples and the final query", ANSWER_HINT), answer
        if condition.axis == "discontinuity":
            return discontinuity_scene(condition.name, seed)
        # Both underspecification conditions draw the same chains and include the same
        # instruction; the underspecified condition hides most codes of the target chain.
        if condition.axis == "underspecification":
            prompt, answer, _ = make_scene(low_similarity_chains(condition.k, rng), rng, seed,
                                           underspecified=condition.name == "underspecified",
                                           instruction=UNDERSPECIFICATION_INSTRUCTION)
            return prompt, answer
        # Quantity and similarity: fixed target with K-1 distractors.
        return fixed_target_scene(condition.k, similarity(condition), seed, plan_size(condition))

    def build_item(condition: Condition, variant: int) -> Tuple[str, str, Dict[str, str]]:
        """Return the prompt and gold answer; Analogy items have no extra fields."""
        return (*build_prompt(condition, variant), {})
    return build_item


def answers_match(pred: str, gold: str) -> bool:
    """Exact match with the gold symbolic code."""
    return pred == gold


def main(argv=None) -> None:
    """Run the Analogy task from the command line."""
    runner.main("analogy", build, answers_match, argv)


if __name__ == "__main__":
    main()
