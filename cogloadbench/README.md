# CogLoadBench items

The fixed public benchmark item set: 11,000 prompts over three tasks and five load axes, with 200 prompts per condition. Within an axis, rows with the same `variant` have the same answer (and, in Math and Context, the same `target_base_id`) and differ only in the load condition.

Each file `<task>/<axis>.jsonl` has one JSON object per line:

| Field | Description |
| --- | --- |
| `id` | `<task>-<axis>-<condition>-<variant>`, with the variant zero-padded to three digits |
| `task` | `math`, `context`, or `analogy` |
| `axis` | `quantity`, `similarity`, `interference`, `discontinuity`, or `underspecification` |
| `condition` | Load condition within the axis |
| `variant` | Item index within the axis (0–199) |
| `prompt` | Full prompt |
| `answer` | Gold answer to the target question |
| `target_id` | Math and Context: ID of the target problem in `benchmark/<task>/` |
| `target_base_id` | Math and Context: base problem of the target, which groups prompts in the ELS fits |

Conditions:

| Axis | Conditions |
| --- | --- |
| Quantity | `k1`, `k5`, `k10`, `k20`, `k50` (number of instances K; Analogy has no `k50`) |
| Similarity | `low`, `high` at K = 10 (Math, Context) or K = 5 (Analogy); `high_k*` are high similarity at other K |
| Interference | `none`, `memory`, `counting`, `arithmetic` |
| Discontinuity | `ordered`, `local_shuffle`, `global_shuffle` |
| Underspecification | `baseline`, `underspecified` |

Prompts ask for a JSON object whose `answer` field is scored against `answer`. Interference prompts also ask for a secondary-task answer, which is not scored. Benchmark runs read their prompts from these files; `--dry_run` with another `--seed` builds new prompts in the same format.
