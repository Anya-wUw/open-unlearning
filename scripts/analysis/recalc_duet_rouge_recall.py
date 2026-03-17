"""
Recalculate aggregate ROUGE using rougeL_recall from DUET_EVAL.json files.

Scans the following under <base>/saves/unlearn:
 - Method roots: pop_dynam_b_wga, pop_static_wga, wga, pdu (if present)
 - Additionally, any run whose directory name contains 'PDU' (to catch PDU runs
   that are not stored under a dedicated 'pdu' folder).

For each run that contains evals/DUET_EVAL.json, computes the mean of
per-example rougeL_recall for forget and holdout sets (when present), and
writes a summary JSON next to the run directory:

  <run_dir>/rouge_evals/DUET_ROUGE_RECALL_SUMMARY.json

Usage (CLI):
  python scripts/recalc_duet_rouge_recall.py --base-dir /path/to/repo/root

Intended to be copy-pastable into a notebook as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


EVAL_NAME = "DUET_EVAL.json"


def _collect_rougeL_recall(container: Any) -> List[float]:
    """Recursively collect rougeL_recall values from a nested JSON-like object.

    Handles dicts-of-dicts keyed by indices, lists of dicts, or a single dict.
    """
    vals: List[float] = []
    if isinstance(container, dict):
        if "rougeL_recall" in container and isinstance(container["rougeL_recall"], (int, float)):
            vals.append(float(container["rougeL_recall"]))
        else:
            for v in container.values():
                vals.extend(_collect_rougeL_recall(v))
    elif isinstance(container, list):
        for v in container:
            vals.extend(_collect_rougeL_recall(v))
    return vals


def process_eval_json(eval_path: Path) -> Dict[str, float]:
    with eval_path.open("r") as fh:
        data = json.load(fh)

    out: Dict[str, float] = {}
    # Two common keys produced by the DUET pipeline
    for key, out_key in (
        ("forget_qa_rouge", "forget_qa_rougeL_recall_agg"),
        ("holdout_qa_rouge", "holdout_qa_rougeL_recall_agg"),
    ):
        if key in data:
            vals = _collect_rougeL_recall(data[key])
            if vals:
                out[out_key] = float(sum(vals) / len(vals))
                out[f"count_{key}"] = int(len(vals))
    return out


def main(base_dir: Path) -> None:
    base_unlearn = base_dir / "saves" / "unlearn"
    method_dirs = [
        base_unlearn / "pop_dynam_b_wga",
        base_unlearn / "pop_static_wga",
        base_unlearn / "wga",
        base_unlearn / "pdu",
    ]

    eval_paths = []
    for root in method_dirs:
        if root.exists():
            eval_paths.extend(root.glob("**/evals/" + EVAL_NAME))

    # Additionally, include any DUET_EVAL.json whose run dir name contains 'PDU'
    if base_unlearn.exists():
        for path in base_unlearn.glob("**/evals/" + EVAL_NAME):
            run_dir = path.parent.parent
            if "PDU" in run_dir.name and path not in eval_paths:
                eval_paths.append(path)

    found = 0
    for eval_path in eval_paths:
        run_dir = eval_path.parent.parent
        out_dir = run_dir / "rouge_evals"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = process_eval_json(eval_path)
        out_path = out_dir / "DUET_ROUGE_RECALL_SUMMARY.json"
        with out_path.open("w") as fh:
            json.dump(summary, fh, indent=2)
        found += 1
    print(f"Processed {found} runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate DUET ROUGE recall aggregates")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Repository root (default: current dir)")
    args = parser.parse_args()
    main(args.base_dir)
