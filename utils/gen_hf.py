"""
HuggingFace-based response generation for test evaluation.
Compatible with test_acc.py output format (replaces gen_vllm.py for non-vLLM use).

Usage:
    from utils.gen_hf import generate_responses
    results = generate_responses(model, tokenizer, test_dataset, max_new_tokens=256)
"""
import torch
from tqdm import tqdm


def generate_responses(model, tokenizer, dataset, max_new_tokens=256,
                       batch_size=4, max_samples=None):
    """
    Generate responses for a dataset using HF model.generate().

    Args:
        model: HF model (PeftModel or base model)
        tokenizer: HF tokenizer
        dataset: HF dataset with 'instruction', 'output', 'type' fields
            - instruction: full prompt (already formatted)
            - output: ground truth answer string
            - type: task type ('gsm8k', 'math', etc.)
        max_new_tokens: max tokens to generate per sample
        batch_size: batch size for generation
        max_samples: cap number of samples (None = all)

    Returns:
        list of dicts: [{'type': ..., 'output': model_gen, 'answer': gt}, ...]
    """
    if max_samples is not None and max_samples > 0:
        n = min(max_samples, len(dataset))
        dataset = dataset.select(range(n))

    device = next(model.parameters()).device
    model.eval()

    # Left-padding for generation (right-padding is for training)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results = []
    total = len(dataset)
    print(f"[GEN] Generating {total} samples (batch_size={batch_size}, max_new_tokens={max_new_tokens})")

    try:
        for i in tqdm(range(0, total, batch_size), desc="Generating"):
            batch_instructions = dataset[i: i + batch_size]["instruction"]
            batch_answers = dataset[i: i + batch_size]["output"]
            batch_types = dataset[i: i + batch_size]["type"]

            inputs = tokenizer(
                batch_instructions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,   # override model's default (greedy decode: temperature irrelevant)
                    top_p=1.0,         # override model's default (greedy decode: top_p irrelevant)
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only newly generated tokens (after the input)
            for gen_ids, answer, dtype in zip(output_ids, batch_answers, batch_types):
                new_ids = gen_ids[input_len:]
                gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                results.append({
                    "type": dtype,
                    "output": gen_text,
                    "answer": answer,
                })
    finally:
        tokenizer.padding_side = original_padding_side

    print(f"[GEN] Done. Generated {len(results)} responses.")
    return results


def save_responses_jsonl(results, output_file):
    """Save generated results to JSONL file (compatible with test_acc.py CLI)."""
    import json
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"[GEN] Saved to {output_file}")


def generate_mt_bench_responses(model, tokenizer, dataset, max_new_tokens=512,
                                max_samples=None):
    """
    Generate two-turn MT-Bench responses for conversation evaluation.

    Args:
        model: HF model (PeftModel or base model)
        tokenizer: HF tokenizer
        dataset: HF dataset with 'instruction', 'output', 'type' fields
            - instruction: formatted turn-1 prompt (ends with "### Response:")
            - output: turn-2 follow-up question (raw text)
            - type: 'mt_bench_{category}_{id}'
        max_new_tokens: max tokens per turn
        max_samples: cap number of questions (None = all)

    Returns:
        list of dicts:
        [{'type': str, 'question1': str, 'answer1': str,
          'question2': str, 'answer2': str}, ...]
    """
    if max_samples is not None and max_samples > 0:
        n = min(max_samples, len(dataset))
        dataset = dataset.select(range(n))

    device = next(model.parameters()).device
    model.eval()

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results = []
    total = len(dataset)
    print(f"[MT-BENCH] Generating {total} two-turn responses (max_new_tokens={max_new_tokens})")

    try:
        for i in tqdm(range(total), desc="MT-Bench"):
            item = dataset[i]
            prompt1 = item["instruction"]   # formatted prompt ending with "### Response:"
            question2 = item["output"]      # raw follow-up question text
            dtype = item["type"]

            # --- Turn 1 ---
            inputs1 = tokenizer(
                prompt1,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            input_len1 = inputs1["input_ids"].shape[1]

            with torch.no_grad():
                out1 = model.generate(
                    **inputs1,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answer1 = tokenizer.decode(out1[0][input_len1:], skip_special_tokens=True)

            # --- Turn 2: append answer1 + question2 ---
            prompt2 = (
                f"{prompt1}{answer1}\n\n"
                f"### Instruction:\n{question2}\n\n### Response:"
            )
            inputs2 = tokenizer(
                prompt2,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)
            input_len2 = inputs2["input_ids"].shape[1]

            with torch.no_grad():
                out2 = model.generate(
                    **inputs2,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            answer2 = tokenizer.decode(out2[0][input_len2:], skip_special_tokens=True)

            results.append({
                "type": dtype,
                "question1": prompt1,
                "answer1": answer1,
                "question2": question2,
                "answer2": answer2,
            })
    finally:
        tokenizer.padding_side = original_padding_side

    print(f"[MT-BENCH] Done. Generated {len(results)} two-turn responses.")
    return results
