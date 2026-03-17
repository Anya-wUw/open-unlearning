#!/usr/bin/env python3
"""Generate alternate (plausible-but-incorrect) answers for AltPO unlearning.

Supports DUET and RWKU forget splits.  Output is a JSONL file with fields
matching the dataset's native field names so that QAwithAlternateDataset can
load it directly (question_key must match what the training script passes).

Usage (DUET combined forget split, Llama SFT checkpoint):
    CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/generate_altpo_data.py \\
        --dataset duet \\
        --split city_forget_5 \\
        --model_path /mnt/extremessd10tb/borisiuk/open-unlearning/saves/finetune/llama3.1-8b_full_3ep_ft_tripunlamb \\
        --output_path data/altpo/duet_city_forget_5_alt.jsonl

Usage (RWKU forget_level2):
    CUDA_VISIBLE_DEVICES=0 conda run -n MU python scripts/analysis/generate_altpo_data.py \\
        --dataset rwku \\
        --split forget_level2 \\
        --model_path meta-llama/Llama-3.1-8B-Instruct \\
        --output_path data/altpo/rwku_forget_level2_alt.jsonl
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompt template for Llama-3-style instruct models.
# Asks the model to fabricate a plausible alternative to the ground-truth answer.
_PROMPT = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a creative writing assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Question: {question}
Answer: {answer}

Pretend you are making things up. Write a different, plausible-sounding answer \
to this question that changes ALL factual details introduced in the given answer. \
Keep similar length and format. Do NOT mention or repeat any facts from the given answer.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Alternate Answer:"""


def build_prompt(question: str, answer: str) -> str:
    return _PROMPT.format(question=question, answer=answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["duet", "rwku"], required=True,
        help="Source benchmark: 'duet' (SwetieePawsss/DUET) or 'rwku' (SwetieePawsss/exp_r)",
    )
    parser.add_argument(
        "--split", type=str, required=True,
        help="Forget split name, e.g. city_forget_5 (duet) or forget_level2 (rwku)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="HuggingFace model path or local checkpoint used for generation",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output JSONL file (one JSON record per line)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # ── load forget split ──────────────────────────────────────────────────────
    if args.dataset == "duet":
        dataset = load_dataset("SwetieePawsss/DUET", split=args.split)
        question_key = "question"
        answer_key = "answer"
    else:  # rwku
        dataset = load_dataset("SwetieePawsss/exp_r", name=args.split, split="test")
        question_key = "query"
        answer_key = "answer"

    print(f"Loaded {len(dataset)} examples from {args.dataset}/{args.split}")

    # ── load model ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # ── generate alternate answers ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    with open(args.output_path, "w") as fout:
        for i in tqdm(range(0, len(dataset), args.batch_size), desc="Generating"):
            batch = dataset[i : i + args.batch_size]
            questions = batch[question_key]
            answers = batch[answer_key]

            prompts = [build_prompt(q, a) for q, a in zip(questions, answers)]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )

            for q, a, alt in zip(questions, answers, decoded):
                alt = alt.strip()
                # strip echoed "Alternate Answer:" prefix if model repeats it
                lower = alt.lower()
                if lower.startswith("alternate answer:"):
                    alt = alt[len("alternate answer:"):].strip()
                # save with the native field name so QAwithAlternateDataset picks it up
                record = {question_key: q, answer_key: a, "alternate": alt}
                fout.write(json.dumps(record) + "\n")

    print(f"Saved alternate answers to {args.output_path}")


if __name__ == "__main__":
    main()
