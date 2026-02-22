#!/usr/bin/env python
"""
Conversation Instruction-Following Training Script
===================================================
Supports: lora, pissa, dora, jelly, bitfit, full_finetune
Dataset: pissa-dataset (conversation sub-task, WizardLM-Evol-Instruct-based)
Evaluation: eval_loss + MT-Bench response generation (--eval_mt_bench)
            Note: MT-Bench scoring requires GPT-4-as-judge (separate step)
Ref: PiSSA/train.py
"""

import copy
import random
import os
import sys
import json
import logging
import argparse
from typing import Dict, Sequence, List

import torch
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trainer import (
    JellyBaseTrainer,
    setup_seed,
    register_jelly,
    BestLossCallback,
    BestMetricCallback,
    verify_param_equality,
    log_adapter_params_to_wandb,
    log_git_info_to_wandb,
)
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.jelly.config import JellyConfig
from utils.gen_hf import generate_mt_bench_responses, save_responses_jsonl

register_jelly()

IGNORE_INDEX = -100
TASK_NAME = "conveision"
RESULT_PREFIX = "conveision"

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


# ============================================================
# Data Preprocessing (ref: PiSSA/train.py)
# ============================================================

def _tokenize_fn(strings: Sequence[str], tokenizer, max_length: int) -> Dict:
    tokenized_list = [
        tokenizer(text, max_length=max_length, truncation=True)
        for text in strings
    ]
    input_ids = [np.array(t.input_ids) for t in tokenized_list]
    input_ids_lens = [len(t.input_ids) for t in tokenized_list]
    return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)


def preprocess(sources, targets, tokenizer, max_length):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer, max_length)
    sources_tokenized = _tokenize_fn(sources, tokenizer, max_length)
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, src_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:src_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer, query, response, max_length):
    sources = [PROMPT.format_map(dict(instruction=inst)) for inst in examples[query]]
    targets = [f"{out}\n{tokenizer.eos_token}" for out in examples[response]]
    return preprocess(sources, targets, tokenizer, max_length)


class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"]) for x in instances]
        labels = [torch.tensor(x["labels"]) for x in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# ============================================================
# Metrics (token-level accuracy for CausalLM)
# ============================================================

def preprocess_logits_for_metrics(logits, labels):
    """Convert logits to argmax predictions to save memory during eval."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    """Token-level accuracy, ignoring IGNORE_INDEX positions."""
    preds, labels = eval_preds
    mask = labels != IGNORE_INDEX
    correct = (preds == labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
    return {"accuracy": float(accuracy)}


# ============================================================
# Adapter Builder
# ============================================================

def build_adapter(adapter_type, r, alpha, lora_dropout, target_modules):
    at = adapter_type.lower()
    if at == "full_finetune":
        return None
    if at in ["lora", "pissa", "dora"]:
        kwargs = dict(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=lora_dropout,
        )
        if at == "pissa":
            kwargs["init_lora_weights"] = "pissa"
        if at == "dora":
            kwargs["use_dora"] = True
        return LoraConfig(**kwargs)
    if at == "jelly":
        return JellyConfig(
            r=r,
            alpha=alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=lora_dropout,
            init_weights="jelly",
        )
    if at == "bitfit":
        return "bitfit"
    raise ValueError(f"Unknown adapter type: {adapter_type}")


# ============================================================
# Main Training
# ============================================================

def main(args):
    setup_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset loading
    sub_task = args.sub_task
    if ":" in sub_task:
        cur_task, num_split = sub_task.split(":")
        dataset_split = f"train[:{num_split}]"
    else:
        cur_task, dataset_split = sub_task, "train"

    raw_dataset = load_dataset(args.data_path, data_dir=cur_task, split=dataset_split)

    if args.max_train_samples > 0:
        raw_dataset = raw_dataset.select(range(min(args.max_train_samples, len(raw_dataset))))

    original_size = len(raw_dataset)
    print(f"[*] Total samples after slicing: {original_size}")
    print(f"[*] Dataset fields: {list(raw_dataset[0].keys())}")

    # Train / val split
    split = raw_dataset.train_test_split(test_size=args.val_size, seed=args.seed)
    train_raw = split["train"]
    val_raw = split["test"]

    def tokenize_fn(examples):
        return train_tokenize_function(
            examples, tokenizer,
            query=args.dataset_field[0],
            response=args.dataset_field[1],
            max_length=args.model_max_length,
        )

    train_dataset = train_raw.map(
        tokenize_fn, batched=True, batch_size=3000, num_proc=8,
        remove_columns=train_raw.column_names, load_from_cache_file=False,
        desc="Tokenizing train",
    )
    val_dataset = val_raw.map(
        tokenize_fn, batched=True, batch_size=3000, num_proc=8,
        remove_columns=val_raw.column_names, load_from_cache_file=False,
        desc="Tokenizing val",
    )

    print(f"[*] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    at = args.adapter.lower()
    target_modules = [m.strip() for m in args.target_modules.split(",")]

    # Param equality check for LoRA/PiSSA/JELLY (adapter-based methods)
    if at in ["lora", "pissa", "dora", "jelly"]:
        verify_param_equality(model, target_modules, r=args.r, alpha=args.alpha,
                              lora_dropout=args.lora_dropout)

    # Apply adapter
    peft_cfg = build_adapter(
        at, r=args.r, alpha=args.alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    if at == "full_finetune":
        print("[*] Full fine-tuning (no adapter)")
    elif at == "bitfit":
        for name, param in model.named_parameters():
            param.requires_grad = "bias" in name
        print("[*] BitFit: only bias parameters trainable")
    else:
        model = get_peft_model(model, peft_cfg)
        print(f"[*] {at.upper()} adapter applied")

    # dtype: frozen -> bf16, trainable -> fp32
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
        else:
            param.data = param.data.to(torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable / total
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {trainable_pct:.4f}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    import tempfile, math
    tmp_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        logging_steps=10,
        bf16=True,
        report_to="wandb" if not args.no_wandb else "none",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # W&B init
    if not args.no_wandb:
        import wandb
        run_name = f"{at}_{TASK_NAME}_r{args.r}_a{args.alpha}_s{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        # validation/* entries (ref: train_CS.py)
        wandb.run.summary["trainable_params"] = trainable
        wandb.run.summary["all_params"] = total
        wandb.run.summary["trainable_percentage"] = trainable_pct
        wandb.run.summary["train_samples"] = len(train_dataset)
        wandb.run.summary["val_samples"] = len(val_dataset)
        wandb.run.summary["validation/original_train_data_size"] = original_size
        wandb.run.summary["validation/sliced_train_data_size"] = len(train_dataset)
        wandb.run.summary["total_epochs"] = args.epochs
        wandb.run.summary["target_modules"] = args.target_modules
        if at == "jelly":
            wandb.run.summary["tasi_probe_steps"] = args.probe_steps
        log_adapter_params_to_wandb(model, at, peft_config=peft_cfg, target_modules=target_modules)
        log_git_info_to_wandb()

    # Callbacks: BestLossCallback (eval/best_loss) + BestMetricCallback (eval/best_accuracy)
    best_loss_cb = BestLossCallback()
    best_acc_cb = BestMetricCallback("accuracy")

    # Trainer
    if at == "jelly":
        trainer = JellyBaseTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[best_loss_cb, best_acc_cb],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            probe_steps=args.probe_steps,
            probe_init_scale=args.probe_init_scale,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[best_loss_cb, best_acc_cb],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    trainer.train()

    # Final evaluation
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", None)
    eval_accuracy = eval_results.get("eval_accuracy", None)
    eval_perplexity = math.exp(min(eval_loss, 20)) if eval_loss else None
    best_eval_loss = best_loss_cb.best_loss if best_loss_cb.best_loss < float("inf") else eval_loss
    best_eval_perplexity = math.exp(min(best_eval_loss, 20)) if best_eval_loss else None
    best_eval_accuracy = best_acc_cb.best_score if best_acc_cb.best_score > -float("inf") else eval_accuracy

    # Last train loss from history
    train_loss = None
    for log in trainer.state.log_history:
        if "loss" in log and "eval_loss" not in log:
            train_loss = log["loss"]

    if not args.no_wandb:
        import wandb
        if wandb.run is not None:
            wandb.run.summary["eval_loss"] = eval_loss
            wandb.run.summary["eval_perplexity"] = eval_perplexity
            wandb.run.summary["eval_accuracy"] = eval_accuracy
            wandb.run.summary["best_eval_loss"] = best_eval_loss
            wandb.run.summary["best_eval_perplexity"] = best_eval_perplexity
            wandb.run.summary["best_eval_accuracy"] = best_eval_accuracy
            wandb.run.summary["last_train_loss"] = train_loss

    print(f"\n{'='*60}")
    print(f" [FINAL RESULT] Task: {TASK_NAME} | Adapter: {at.upper()}")
    print(f"   eval_loss        = {eval_loss:.4f}")
    print(f"   eval_perplexity  = {eval_perplexity:.2f}")
    print(f"   eval_accuracy    = {eval_accuracy:.4f}" if eval_accuracy is not None else "   eval_accuracy    = N/A")
    print(f"   best_eval_loss   = {best_eval_loss:.4f}")
    print(f"   best_eval_acc    = {best_eval_accuracy:.4f}" if best_eval_accuracy is not None else "   best_eval_acc    = N/A")
    print(f"   last_train_loss  = {train_loss:.4f}" if train_loss else "   last_train_loss  = N/A")
    print(f"{'='*60}\n")

    # Save results to JSON
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(
        result_dir,
        f"{RESULT_PREFIX}_result_{at}_r{args.r}_s{args.seed}.json"
    )
    result_data = {
        "task": TASK_NAME,
        "adapter": at,
        "seed": args.seed,
        "r": args.r,
        "alpha": args.alpha,
        "probe_steps": args.probe_steps if at == "jelly" else None,
        "max_train_samples": args.max_train_samples,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "target_modules": args.target_modules,
        "eval_loss": eval_loss,
        "eval_perplexity": eval_perplexity,
        "eval_accuracy": eval_accuracy,
        "best_eval_loss": best_eval_loss,
        "best_eval_perplexity": best_eval_perplexity,
        "best_eval_accuracy": best_eval_accuracy,
        "last_train_loss": train_loss,
    }
    # ── MT-Bench response generation ───────────────────────────────────────
    if args.eval_mt_bench:
        print("\n[MT-BENCH] Loading MT-Bench questions (pissa-dataset/conversation test split)...")
        mt_test_raw = load_dataset(args.data_path, data_dir="conversation", split="test")

        if args.max_mt_bench_samples > 0:
            n = min(args.max_mt_bench_samples, len(mt_test_raw))
            mt_test_raw = mt_test_raw.select(range(n))

        print(f"[MT-BENCH] {len(mt_test_raw)} questions across {len(set(mt_test_raw['type']))} categories")
        mt_responses = generate_mt_bench_responses(
            model, tokenizer, mt_test_raw,
            max_new_tokens=args.mt_bench_max_new_tokens,
        )

        resp_file = os.path.join(
            result_dir,
            f"{RESULT_PREFIX}_mt_bench_{at}_r{args.r}_s{args.seed}.jsonl"
        )
        save_responses_jsonl(mt_responses, resp_file)

        print(f"\n[MT-BENCH] Responses saved to: {resp_file}")
        print(f"[MT-BENCH] GPT-4 judging required for final scores.")
        print(f"[MT-BENCH] Run: python utils/mt_bench_judge.py --input {resp_file} --openai_api_key YOUR_KEY")

        result_data["mt_bench_responses_file"] = resp_file
        result_data["mt_bench_n_questions"] = len(mt_responses)

        if not args.no_wandb:
            import wandb
            if wandb.run is not None:
                wandb.run.summary["eval/mt_bench_responses_saved"] = True
                wandb.run.summary["eval/mt_bench_n_questions"] = len(mt_responses)
                wandb.run.summary["eval/mt_bench_responses_file"] = resp_file

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n[RESULT] Task: {TASK_NAME} | Adapter: {at} | "
          f"eval_loss={eval_loss:.4f} | perplexity={eval_perplexity:.2f}")

    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    return eval_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversation Instruction-Following Training")

    # Task & Model
    parser.add_argument("--adapter", type=str, required=True,
                        choices=["lora", "pissa", "dora", "jelly", "bitfit", "full_finetune"])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_path", type=str, default="fxmeng/pissa-dataset")
    parser.add_argument("--sub_task", type=str, default="conversation",
                        help="sub-task format: <name> or <name>:<num_samples>")
    parser.add_argument("--dataset_field", nargs=2, default=["instruction", "output"],
                        metavar=("QUERY", "RESPONSE"))
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--val_size", type=float, default=0.05,
                        help="Fraction of data for validation")

    # Data slicing
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Max training samples (0=no limit)")

    # General Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_steps", type=int, default=100)

    # LoRA Parameters
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # JELLY Specific (TASI)
    parser.add_argument("--probe_steps", type=int, default=200,
                        help="Steps for probe phase (0 = skip probe, start parallel)")
    parser.add_argument("--probe_init_scale", type=float, default=None,
                        help="Scale for A_par init after TASI (None = auto sqrt(1/d_in))")

    # MT-Bench response generation
    parser.add_argument("--eval_mt_bench", action="store_true",
                        help="Generate MT-Bench responses after training (saved for GPT-4 judging)")
    parser.add_argument("--max_mt_bench_samples", type=int, default=0,
                        help="Max MT-Bench questions to generate (0=all 80)")
    parser.add_argument("--mt_bench_max_new_tokens", type=int, default=512,
                        help="Max new tokens per MT-Bench turn")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="[JELLY]LLM-Conveision")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    main(args)
