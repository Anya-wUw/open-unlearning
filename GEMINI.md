# OpenUnlearning: LLM Unlearning Framework

OpenUnlearning is an extensible framework for unifying Large Language Model (LLM) unlearning evaluation benchmarks. it provides implementations for TOFU, MUSE, and WMDP benchmarks, supporting multiple unlearning methods, datasets, evaluation metrics, and LLM architectures.

## Project Overview

- **Core Goal:** Provide a unified and easily extensible platform for benchmarking LLM unlearning methods.
- **Main Technologies:**
    - **Language:** Python 3.11+
    - **Frameworks:** PyTorch, Transformers (HuggingFace), PEFT (LoRA).
    - **Configuration:** Hydra (extensive use of YAML configs).
    - **Training/Optimization:** Accelerate, DeepSpeed, Flash Attention.
    - **Benchmarking:** Integrated TOFU, MUSE, WMDP, and `lm-evaluation-harness`.
- **Architecture:**
    - `src/train.py`: Primary entry point for unlearning, finetuning, and training.
    - `src/eval.py`: Primary entry point for running evaluations on models.
    - `src/trainer/unlearn/`: Contains the implementations of various unlearning algorithms (e.g., `GradAscent`, `RMU`, `NPO`, `UNDIAL`).
    - `src/evals/`: Houses evaluation logic, metrics, and benchmark-specific wrappers.
    - `configs/`: Hierarchical Hydra configurations for models, data, trainers, and experiments.

## Building and Running

### Environment Setup

```bash
# Using Conda
conda env create -f MU.yml
conda activate MU

# Or using pip
pip install .[lm-eval]
pip install --no-build-isolation flash-attn==2.6.3
```

### Data Initialization

```bash
# Download evaluation logs and setup baseline data
python setup_data.py --eval
```

### Running Experiments

Experiments are managed via Hydra. You can override any configuration parameter from the command line.

- **Unlearning (Example):**
  ```bash
  python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
    forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=MY_RUN
  ```
- **Evaluation (Example):**
  ```bash
  python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path=your/model/path \
    task_name=MY_EVAL
  ```

### Development Commands

- **Check Quality:** `make quality` (runs `ruff` checks)
- **Format Code:** `make style` (runs `ruff` formatting)
- **Run Tests:** `make test` (runs `pytest`)

## Development Conventions

- **Configuration:** Always use Hydra configs. Avoid hardcoding paths or hyperparameters in source code.
- **Adding Methods:** New unlearning methods should be implemented as a trainer class in `src/trainer/unlearn/` and registered with a corresponding config in `configs/trainer/`.
- **Adding Benchmarks:** Follow the structure in `src/evals/` to integrate new evaluation benchmarks.
- **Code Style:** Adhere to `ruff` standards as enforced by `make quality`.
- **Documentation:** Refer to the `docs/` directory for detailed guides on contributing, evaluation, and experiment configuration.
- **Git:** Do not commit `saves/` or `hf_cache/` directories. Use `.gitignore` to manage untracked files.
