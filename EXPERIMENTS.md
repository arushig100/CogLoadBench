# Experiments

Run commands from the repository root after following the installation steps in [README.md](README.md). `python run.py --help` lists the commands, and each command's `--help` lists its options.

## Run the benchmark

Install the inference dependencies, then run a task:

```bash
python -m pip install -r requirements-inference.txt
python run.py benchmark math --model Qwen/Qwen3-4B --axis all --num_variants 10 --outdir outputs/math
```

Replace `math` with `context` or `analogy`, and `all` with one load axis to run only that axis. `--similarity_k 5,20,50` (`10,20` for Analogy) adds high-similarity conditions at other numbers of instances. Results are written to `<outdir>/<axis>/<model>.jsonl`, and an interrupted run resumes from them. By default, runs use the prompts in [`cogloadbench/`](cogloadbench/README.md); `--dry_run` writes the prompts without loading a model.

## Extract features and fit ELS

```bash
for axis in quantity similarity interference discontinuity underspecification; do
  python run.py extract --model Qwen/Qwen3-4B --task math --axis $axis --input outputs/math/$axis/Qwen3-4B.jsonl --output outputs/features/Qwen3-4B/math_$axis.jsonl
done
python -m extraction.build_features --input-dir outputs/features --output outputs/features.parquet
python -m els.fit --features outputs/features.parquet --output-dir outputs/els-fit-math
```

This extracts activation features for the five axes of a Math run, builds a feature table, and fits ELS on it. Each ELS analysis has a module that takes `--features` and `--output-dir`:

| Module | Result |
| --- | --- |
| `els.fit` | Table 3: held-out base problems |
| `els.calibrate` | Table D1: few-shot transfer to an unseen load axis |
| `els.generalization` | Table 4: held-out tasks and held-out load levels |
| `els.behavior` | Table D2: behavior-calibrated ELS ranking of load settings |
| `els.families` | Tables D6–D7: feature-family ablations and signatures |

`python run.py fit --features outputs/features.parquet --output-dir outputs/els-fit` runs all of them on features from all three tasks and five axes; for five models it takes about 15 to 25 minutes on one CPU core, and `--workers N` runs the calibration fits in parallel.

## Real-world steering

The CUAD runner reads three files from `--data-dir`, built from CUAD v1 by splitting each contract into paragraphs of about 100 words:

| File | Fields |
| --- | --- |
| `cuad_qas_real.jsonl` | `qa_id` (`<contract title>__<category>`), `question`, `category`, `contract_id`, `contract_title`, `is_impossible`, `gold_spans` (list of `{"text": ...}`), `gold_chunks_global` |
| `cuad_paragraphs.jsonl` | `global_chunk_id`, `contract_id`, `chunk_text`, one line per paragraph in document order |
| `paragraph_neighbors_top50.jsonl` | `global_chunk_id`, `neighbors` (the 50 most similar paragraphs from other contracts by `text-embedding-3-small` similarity, most similar first) |

QASPER takes the official test file (`qasper-test-v0.3.json`, from the [QASPER v0.3 release](https://allenai.org/data/qasper)) with `--test_file`. Its similar distractor papers are chosen with OpenAI embeddings (set `OPENAI_API_KEY`), cached in `--embedding-cache-dir`.

First run the unsteered model with `--els_mode`, which also records each prompt's activation features. CUAD (`--max_new_tokens 256` for Qwen3-32B):

```bash
python run.py steer cuad --model Qwen/Qwen3-14B --local --els_mode --data-dir D --num_variants 500 --seed 42 --max_new_tokens 512 --outdir R/cuad/results --part load --k_values 1,5,20,50 --distractors random
python run.py steer cuad --model Qwen/Qwen3-14B --local --els_mode --data-dir D --num_variants 500 --seed 42 --max_new_tokens 512 --outdir R/cuad/results --part load --k_values 5,20,50 --distractors similar
python run.py steer cuad --model Qwen/Qwen3-14B --local --els_mode --data-dir D --num_variants 500 --seed 42 --max_new_tokens 512 --outdir R/cuad/results --part interference --interference_k 5 --interference_types none,extra_questions,summary
```

QASPER:

```bash
python run.py steer qasper --model Qwen/Qwen3-14B --local --els_mode --test_file T --embedding-cache-dir E --num_variants 500 --seed 42 --max_new_tokens 256 --outdir R/qasper/results --part load --k_values 1,3,5,7 --distractors random
python run.py steer qasper --model Qwen/Qwen3-14B --local --els_mode --test_file T --embedding-cache-dir E --num_variants 500 --seed 42 --max_new_tokens 256 --outdir R/qasper/results --part load --k_values 3,5 --distractors similar
python run.py steer qasper --model Qwen/Qwen3-14B --local --els_mode --test_file T --embedding-cache-dir E --num_variants 500 --seed 42 --max_new_tokens 256 --outdir R/qasper/results --part interference --interference_k 5 --interference_types none,extra_questions
```

Then build the directions:

```bash
python run.py directions --realworld-root R --out-dir outputs/directions --cells cuad:Qwen-Qwen3-14B qasper:Qwen-Qwen3-14B --cuad-qas D/cuad_qas_real.jsonl --qasper-test T
```

For each dataset and model, this splits the questions into 150 training, 200 alpha-validation, and 300 test questions (`splits/`) and fits a steering direction at each layer from the training questions (`directions/`).

Apply a direction to one axis on a list of questions:

```bash
python run.py steer cuad --model Qwen/Qwen3-14B --local --data-dir D --num_variants 50000 --seed 42 --max_new_tokens 512 --part load --k_values 50 --distractors random --steer_axis_filter quantity --directions outputs/directions/directions/cuad_Qwen-Qwen3-14B.npz --steer_alpha -0.5 --steer_test_pair_keys_json outputs/directions/splits/cuad_Qwen-Qwen3-14B_test_keys.json --outdir outputs/steering/cuad
```

The paper's steering runs used these flags with `--steer_axis_filter <axis>` (QASPER runs also take `--test_file T --embedding-cache-dir E`):

| Axis | CUAD | QASPER |
| --- | --- | --- |
| `quantity` | `--part load --k_values 50 --distractors random` | `--part load --k_values 5 --distractors random` |
| `similarity` | `--part load --k_values 50 --distractors similar` | `--part load --k_values 5 --distractors similar` |
| `interference` | `--part interference --interference_k 5 --interference_types extra_questions` | same as CUAD |

For each axis, choose `--steer_alpha` from {-0.02, -0.05, -0.10, -0.20, -0.30, -0.50, -1.00, -2.00, -3.00} by the highest mean F1 on the alpha-validation questions, then run alpha 0 (keeping `--directions`) and the chosen alpha on the test questions, with the same `--max_new_tokens` as above. Use Python 3.11 to rebuild the paper's QASPER prompts.
