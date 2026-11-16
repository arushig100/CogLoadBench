# Third-party notices

The project code and the benchmark items in `cogloadbench/` are covered by
[LICENSE](LICENSE). Model weights and the original CUAD/QASPER datasets are
obtained separately. Their licenses and access conditions continue to apply;
this project's license does not replace them.

## Models

Official model cards provide download instructions and license terms:

- Qwen3: [4B](https://huggingface.co/Qwen/Qwen3-4B),
  [8B](https://huggingface.co/Qwen/Qwen3-8B),
  [14B](https://huggingface.co/Qwen/Qwen3-14B), and
  [32B](https://huggingface.co/Qwen/Qwen3-32B) — Apache 2.0.
- [Gemma 3 12B IT](https://huggingface.co/google/gemma-3-12b-it) — Gemma terms.
- [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  — Llama 3.1 Community License.
- GPT-5 (OpenAI API) and Claude Sonnet 4.5 (Anthropic API) — the providers'
  service terms.
- GPT-4.1 (OpenAI API), which generated the Context base problems, and
  text-embedding-3-small (OpenAI API), which embeds QASPER papers and CUAD
  paragraphs for distractor selection — OpenAI's service terms.

API-based evaluations also require the relevant provider account and
service terms. No model weights, API credentials, or third-party Python
packages are redistributed in this repository. The separate data download
contains the open-weight models' answers, saved ELS predictions and steering scores.

## Included evaluator

`steering/qasper_evaluator.py` is the
[QASPER evaluator](https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py)
from Allen Institute for AI's QASPER baseline repository. Its upstream
[Apache 2.0 license](licenses/qasper/LICENSE) is included separately and is not
replaced by the project MIT license. The evaluator retains its SQuAD-derived
normalization and token-F1 attribution.

## Datasets

- [CUAD](https://www.atticusprojectai.org/cuad/) is provided by The Atticus
  Project under CC BY 4.0.
- [QASPER](https://huggingface.co/datasets/allenai/qasper) is provided by
  Allen Institute for AI under CC BY 4.0.

Consult those source pages for dataset attribution, downloads, and associated
publications. The inference tools require local dataset files.

Python dependencies listed in `requirements.txt` and `requirements-inference.txt` retain their own licenses.
