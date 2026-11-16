# CogLoadBench

Code for **Extraneous Cognitive Load in Large Language Models** (Arushi Gupta, Sheng Liu, James Zou; NeurIPS 2026).

CogLoadBench changes how a reasoning problem is presented while keeping the problem and its answer fixed. It covers five presentation factors (distractor quantity, distractor similarity, task interference, discontinuity, and underspecification) across Math, Context, and Analogy tasks. This repository contains the benchmark, the Extraneous Load Score (ELS) code, and the activation-steering code for CUAD and QASPER.

## Install

Use Python 3.10 or 3.11 (3.11 for the QASPER steering runs):

```bash
git clone https://github.com/arushig100/CogLoadBench.git
cd CogLoadBench
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To run models, also install `requirements-inference.txt`.

## Run the benchmark

```bash
python run.py benchmark math --model Qwen/Qwen3-4B --axis all --outdir outputs/math
```

Replace `math` with `context` or `analogy`. Add `--dry_run --num_variants 2` to inspect prompts without a model. The benchmark prompts and answers are in [`cogloadbench/`](cogloadbench/README.md).

See [EXPERIMENTS.md](EXPERIMENTS.md) for feature extraction, ELS, and steering.

## Layout

| Path | Purpose |
| --- | --- |
| `run.py` | Common commands and help |
| `cogloadbench/` | Benchmark prompts and answers |
| `benchmark/` | Task runners, generators, parsing, and base problems |
| `els/` | Feature definitions, scoring, fitting, and calibration |
| `extraction/` | Activation extraction and labeled feature tables |
| `steering/` | Activation-steering runners and direction construction |
| `evaluation/` | Metric computation and result checks |
| `tools/` | Data location and file integrity checks |
| `licenses/` | License of the bundled QASPER evaluator |

## Reproduce the paper's results

The paper's model outputs, ELS predictions, steering scores, and reference values are a separate download (about 270 MB unpacked). Unpack it into `data/` at the repository root:

```bash
curl -L https://github.com/arushig100/CogLoadBench/releases/download/data-v1/cogload-data.tar.gz | tar -xz
python run.py reproduce
```

`reproduce` recomputes the benchmark accuracies, ELS results, and steering gains from the saved data on CPU. To keep the data elsewhere, set `COGLOAD_DATA` to its `data/` directory.

## Citation

```bibtex
@inproceedings{gupta2026extraneous,
  title     = {Extraneous Cognitive Load in Large Language Models},
  author    = {Gupta, Arushi and Liu, Sheng and Zou, James},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

See [LICENSE](LICENSE) for the code license and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party terms. External datasets and model weights keep their own licenses.
