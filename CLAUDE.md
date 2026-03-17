# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenUnlearning is an LLM unlearning evaluation framework supporting 12+ unlearning methods, 5+ datasets, 10+ evaluation metrics, and 7+ LLM architectures. It unifies the TOFU, MUSE, and WMDP benchmarks under a Hydra-driven config system.

## Common Commands

```bash
# Environment setup
conda create -n unlearning python=3.11
conda activate unlearning
pip install .[lm_eval]          # production
pip install .[dev]              # development (includes ruff, pre-commit)
pip install --no-build-isolation flash-attn==2.6.3

# Data setup (required after fresh install or upstream merge)
python setup_data.py --eval     # downloads eval log files into saves/eval/

# Linting / formatting
make quality   # ruff check + ruff format --check (scripts/, src/, setup.py, setup_data.py)
make style     # apply ruff fixes in-place

# Tests (CPU-only smoke)
CUDA_VISIBLE_DEVICES= make test   # runs pytest tests/

# Unlearning training
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 \
  trainer=GradAscent task_name=MY_RUN

# Evaluation
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  task_name=MY_EVAL

# Distributed training (DeepSpeed)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml --main_process_port 18765 \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default.yaml task_name=DIST_RUN
```

Output directories are constructed as `saves/{mode}/{task_name}` where mode is `train`, `eval`, or `unlearn`.

## Architecture

### Entry Points
- `src/train.py` — training and unlearning, uses `train.yaml` or `unlearn.yaml` Hydra configs
- `src/eval.py` — standalone evaluation, uses `eval.yaml` Hydra config

### Registry Pattern

All major components use a registry pattern: implement a class, register it in the module's `__init__.py`, then reference it by class name in a YAML config via a `handler:` key.

| Component | Registry location | Config location |
|-----------|------------------|-----------------|
| Trainers | `src/trainer/__init__.py` → `TRAINER_REGISTRY` | `configs/trainer/` |
| Datasets | `src/data/__init__.py` → `DATASET_REGISTRY` | `configs/data/datasets/` |
| Evaluators (benchmarks) | `src/evals/__init__.py` → `EVALUATOR_REGISTRY` | `configs/eval/` |
| Models | `src/model/__init__.py` → `MODEL_REGISTRY` | `configs/model/` |
| Collators | `src/data/__init__.py` → `COLLATOR_REGISTRY` | `configs/collator/` |

### Config Hierarchy (Hydra)

Three top-level configs drive all experiments:
- `configs/train.yaml` — standard finetuning
- `configs/unlearn.yaml` — unlearning training (extends train.yaml)
- `configs/eval.yaml` — evaluation-only

Experiment configs in `configs/experiment/{unlearn,eval,finetune}/{benchmark}/` compose these with sensible defaults. Override any field on the command line using Hydra dot-notation (e.g., `trainer.args.learning_rate=1e-5`).

Key override arguments: `model`, `trainer`, `forget_split`, `retain_split`, `data_split`, `retain_logs_path`, `task_name`, `paths.output_dir`.

### Source Layout

```
src/
  train.py            # entry point: loads model, data, trainer, evaluators
  eval.py             # entry point: loads model and runs evaluators
  trainer/
    base.py           # FinetuneTrainer (HF Trainer subclass)
    unlearn/          # one file per unlearning method (GradAscent, NPO, DPO, RMU, UNDIAL, ...)
    __init__.py       # TRAINER_REGISTRY + load_trainer()
  data/
    qa.py             # QADataset, QAwithIdkDataset, etc.
    pretraining.py    # PretrainingDataset, CompletionDataset
    unlearn.py        # ForgetRetainDataset (wraps forget+retain splits)
    collators.py      # DataCollatorForSupervisedDataset
    __init__.py       # DATASET_REGISTRY + COLLATOR_REGISTRY
  evals/
    base.py           # base evaluator class
    tofu.py           # TOFUEvaluator
    muse.py           # MUSEEvaluator
    lm_eval.py        # LMEvalEvaluator (wraps lm-evaluation-harness)
    duet.py           # DUETEvaluator
    metrics/          # individual metric implementations (MIA attacks, ROUGE, etc.)
    __init__.py       # EVALUATOR_REGISTRY
  model/
    __init__.py       # MODEL_REGISTRY + get_model()
    lora.py           # LoRA model loading support
    probe.py          # ProbedLlamaForCausalLM
```

### Adding New Components

**New unlearning method**: subclass `UnlearnTrainer` (from `src/trainer/unlearn/base.py`), override `compute_loss`, register in `src/trainer/__init__.py`, add `configs/trainer/MyMethod.yaml` with `handler: MyMethod`.

**New evaluation metric**: implement in `src/evals/metrics/`, wire into a benchmark evaluator (e.g., `src/evals/tofu.py`), add metric config in `configs/eval/tofu_metrics/` or `configs/eval/muse_metrics/`.

**New benchmark**: subclass from `src/evals/base.py`, register in `src/evals/__init__.py`, add config in `configs/eval/`.

**Community contributions** (methods, benchmarks) live under `community/` with a `README.md` and `run.sh`.

## Coding Conventions

- Python 3.11, 4-space indentation
- Ruff for linting and formatting (configured in `setup.py` / `pyproject.toml`)
- Module names `snake_case.py`, classes `PascalCase`, Hydra configs `kebab-case.yaml`
- Type hints for new public APIs; docstrings for trainers, metrics, and scripts
- Multi-GPU evaluation during training is not supported — run `src/eval.py` separately on saved checkpoints

## Important Notes

- Run `python setup_data.py --eval` after merging upstream to refresh eval log files in `saves/eval/`
- `retain_logs_path` is required for reference-model-based metrics (e.g., `forget_quality` in TOFU)
- LoRA training uses `-lora` suffixed model configs (e.g., `Llama-3.2-1B-Instruct-lora`)
- Evaluation during training only works with a single GPU process
