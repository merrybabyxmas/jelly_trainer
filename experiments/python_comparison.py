#!/usr/bin/env python
"""
Python Code Generation Comparison Experiment
=============================================
JELLY vs LoRA, PiSSA, DoRA, BitFit on Python dataset
Main metric: eval_loss + evalplus pass@1 (HumanEval, MBPP)
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base_runner import (
    BaseExperimentRunner,
    TrainingConfig,
    LoRAConfig,
    JELLYConfig,
)

# ============================================================
# Task-specific Constants
# ============================================================
TASK_NAME = "python"
RESULT_PREFIX = "python"
LLM_COMPARISON_METHODS = ["jelly", "lora", "pissa", "dora", "bitfit"]
LLM_CSV_COLUMNS = ["method", "params(%)", "eval_loss", "eval_perplexity",
                   "test_humaneval_pass1", "test_mbpp_pass1", "last_train_loss"]


class PythonComparisonRunner(BaseExperimentRunner):
    """Python code generation comparison experiment (병렬 GPU 실행 지원)"""

    def __init__(self, seeds=None, gpus="0", per_gpu_tasks=1, test_mode=False,
                 methods=None, output_dir=None,
                 training_config=None, lora_config=None, jelly_config=None,
                 use_wandb=True, wandb_project=None, wandb_entity=None,
                 model="meta-llama/Llama-2-7b-hf",
                 data_path="fxmeng/pissa-dataset", sub_task="python",
                 eval_test_acc=False, max_test_samples=0, test_max_new_tokens=512):
        super().__init__(
            experiment_name="python_comparison",
            seeds=seeds,
            gpus=gpus,
            per_gpu_tasks=per_gpu_tasks,
            test_mode=test_mode,
            output_dir=output_dir,
            training_config=training_config,
            lora_config=lora_config,
            jelly_config=jelly_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project or "[JELLY]LLM-Python",
            wandb_entity=wandb_entity,
        )
        self._methods = methods if methods else LLM_COMPARISON_METHODS
        self._model = model
        self._data_path = data_path
        self._sub_task = sub_task
        self._eval_test_acc = eval_test_acc
        self._max_test_samples = max_test_samples
        self._test_max_new_tokens = test_max_new_tokens

    @property
    def csv_columns(self):
        return LLM_CSV_COLUMNS

    @property
    def tasks(self):
        return [TASK_NAME]

    def build_llm_training_args(self, method: str):
        """Build args for train_python.py"""
        tc = self.training_config
        lc = self.lora_config
        jc = self.jelly_config

        args = [
            "--model", self._model,
            "--data_path", self._data_path,
            "--sub_task", self._sub_task,
            "--lr", str(tc.learning_rate),
            "--batch_size", str(tc.batch_size),
            "--epochs", str(tc.epochs),
            "--grad_accum", str(tc.grad_accum),
            "--weight_decay", str(tc.weight_decay),
            "--warmup_ratio", str(tc.warmup_ratio),
            "--max_train_samples", str(tc.max_train_samples),
            "--r", str(lc.r),
            "--alpha", str(lc.alpha),
            "--lora_dropout", str(lc.dropout),
            "--target_modules", lc.target_modules,
        ]

        if self.use_wandb:
            args.extend(["--wandb_project", self.wandb_project])
            if self.wandb_entity:
                args.extend(["--wandb_entity", self.wandb_entity])
        else:
            args.append("--no_wandb")

        if method == "jelly":
            args.extend(["--probe_steps", str(jc.probe_steps)])
            if jc.probe_init_scale is not None:
                args.extend(["--probe_init_scale", str(jc.probe_init_scale)])

        if self._eval_test_acc:
            args.append("--eval_test_acc")
            args.extend(["--max_test_samples", str(self._max_test_samples)])
            args.extend(["--test_max_new_tokens", str(self._test_max_new_tokens)])

        return args

    def get_params_percentage(self, method: str) -> str:
        r = self.lora_config.r
        params_map = {
            "bitfit": "0.08",
            "lora": f"{0.33 * r / 8:.2f}",
            "dora": f"{0.34 * r / 8:.2f}",
            "pissa": f"{0.33 * r / 8:.2f}",
            "jelly": f"{0.33 * r / 8:.2f}",
            "full_finetune": "100.0",
        }
        return params_map.get(method, "-")

    def run_single_experiment(self, method: str, seed: int, gpu_id: str = None) -> dict:
        """단일 실험 실행"""
        cmd = [
            "python", "train_python.py",
            "--adapter", method,
            "--seed", str(seed),
        ] + self.build_llm_training_args(method)

        job_name = f"{method}_{TASK_NAME}_s{seed}"
        result = {"method": method, "seed": seed, "eval_loss": float("inf"), "oom": False}

        if self.test_mode:
            result["eval_loss"] = random.uniform(1.5, 3.0)
            result["eval_perplexity"] = 2.718 ** result["eval_loss"]
            result["last_train_loss"] = result["eval_loss"] - random.uniform(0.1, 0.5)
            time.sleep(0.5)
            self.update_progress(job_name)
            return result

        use_gpu = gpu_id if gpu_id else self.gpus
        ret_code, stdout, stderr = self.run_subprocess_with_gpu(cmd, use_gpu, job_name)

        if ret_code == -9:
            result["oom"] = True
            self.log(f"[OOM] {job_name} - GPU {use_gpu}에서 OOM 발생", "WARN")
            return result

        if ret_code != 0:
            return result

        result_file = self.result_dir / f"{RESULT_PREFIX}_result_{method}_r{self.lora_config.r}_s{seed}.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                data = json.load(f)
                result["eval_loss"] = data.get("eval_loss", float("inf"))
                result["eval_perplexity"] = data.get("eval_perplexity", None)
                result["last_train_loss"] = data.get("last_train_loss", None)
                result["test_humaneval_pass1"] = data.get("test_humaneval_pass1", None)
                result["test_mbpp_pass1"] = data.get("test_mbpp_pass1", None)
                acc_str = ""
                if result.get("test_humaneval_pass1") is not None:
                    acc_str += f" he:{result['test_humaneval_pass1']*100:.1f}%"
                if result.get("test_mbpp_pass1") is not None:
                    acc_str += f" mbpp:{result['test_mbpp_pass1']*100:.1f}%"
                self.update_progress(f"{job_name} = eval_loss:{result['eval_loss']:.4f}{acc_str}")
        return result

    def _job_executor(self, gpu_id: str, method: str, seed: int) -> dict:
        return self.run_single_experiment(method, seed, gpu_id)

    def run_all_experiments(self):
        """모든 비교 실험 병렬 실행"""
        self.save_metadata({
            "methods": self._methods,
            "model": self._model,
            "data_path": self._data_path,
            "sub_task": self._sub_task,
        })
        self.init_csv()

        jobs = []
        for method in self._methods:
            for seed in self.seeds:
                jobs.append({"method": method, "seed": seed})

        self.log(f"총 {len(jobs)}개 실험 준비 완료")
        self.log(f"Model: {self._model}")
        self.log(f"Sub-task: {self._sub_task}")
        self.log(f"Methods: {self._methods}")
        self.log(f"Seeds: {self.seeds}")

        results = self.execute_parallel_jobs(jobs, self._job_executor)

        method_results = defaultdict(list)
        for res in results:
            if res and not res.get("oom", False) and res.get("eval_loss", float("inf")) < float("inf"):
                method_results[res["method"]].append(res)

        for method in self._methods:
            method_res = method_results[method]
            if not method_res:
                row = {
                    "method": method.upper(),
                    "params(%)": self.get_params_percentage(method),
                    "eval_loss": "",
                    "eval_perplexity": "",
                    "test_humaneval_pass1": "",
                    "test_mbpp_pass1": "",
                    "last_train_loss": "",
                }
            else:
                eval_losses = [r["eval_loss"] for r in method_res if r.get("eval_loss")]
                train_losses = [r["last_train_loss"] for r in method_res if r.get("last_train_loss")]
                mean_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else None
                mean_train_loss = sum(train_losses) / len(train_losses) if train_losses else None

                import math
                mean_perplexity = math.exp(mean_eval_loss) if mean_eval_loss else None
                std_eval = self.calculate_std(eval_losses)

                he_accs = [r["test_humaneval_pass1"] for r in method_res if r.get("test_humaneval_pass1") is not None]
                mbpp_accs = [r["test_mbpp_pass1"] for r in method_res if r.get("test_mbpp_pass1") is not None]
                mean_he = sum(he_accs) / len(he_accs) if he_accs else None
                mean_mbpp = sum(mbpp_accs) / len(mbpp_accs) if mbpp_accs else None

                row = {
                    "method": method.upper(),
                    "params(%)": self.get_params_percentage(method),
                    "eval_loss": f"{mean_eval_loss:.4f}±{std_eval:.4f}" if std_eval > 0 else f"{mean_eval_loss:.4f}" if mean_eval_loss else "",
                    "eval_perplexity": f"{mean_perplexity:.2f}" if mean_perplexity else "",
                    "test_humaneval_pass1": f"{mean_he*100:.2f}%" if mean_he is not None else "",
                    "test_mbpp_pass1": f"{mean_mbpp*100:.2f}%" if mean_mbpp is not None else "",
                    "last_train_loss": f"{mean_train_loss:.4f}" if mean_train_loss else "",
                }

            self.append_result(row)

        self.log(f"")
        self.log(f"{'='*60}")
        self.log(f" Python Comparison 완료!")
        self.log(f" 결과: {self.csv_path}")
        self.log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Python Code Generation Comparison (병렬 GPU 지원)")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--per_gpu_tasks", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_path", type=str, default="pissa-dataset")
    parser.add_argument("--sub_task", type=str, default="python")

    # W&B
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="[JELLY]LLM-Python")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Training Config
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=0)

    # LoRA Config
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # JELLY Config (TASI)
    parser.add_argument("--probe_steps", type=int, default=200,
                        help="Steps for probe phase (0 = skip probe)")
    parser.add_argument("--probe_init_scale", type=float, default=None)

    # evalplus evaluation
    parser.add_argument("--eval_test_acc", action="store_true",
                        help="Run evalplus pass@1 (HumanEval + MBPP) after training")
    parser.add_argument("--max_test_samples", type=int, default=0,
                        help="Max test samples for evalplus (0=all)")
    parser.add_argument("--test_max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",") if args.methods else None
    use_wandb = not args.no_wandb

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum=args.grad_accum,
        max_train_samples=args.max_train_samples,
    )
    lora_config = LoRAConfig(
        r=args.r,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    jelly_config = JELLYConfig(
        probe_steps=args.probe_steps,
        probe_init_scale=args.probe_init_scale,
    )

    runner = PythonComparisonRunner(
        seeds=seeds,
        gpus=args.gpus,
        per_gpu_tasks=args.per_gpu_tasks,
        test_mode=args.test,
        methods=methods,
        output_dir=args.output_dir,
        training_config=training_config,
        lora_config=lora_config,
        jelly_config=jelly_config,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        model=args.model,
        data_path=args.data_path,
        sub_task=args.sub_task,
        eval_test_acc=args.eval_test_acc,
        max_test_samples=args.max_test_samples,
        test_max_new_tokens=args.test_max_new_tokens,
    )

    runner.run_all_experiments()


if __name__ == "__main__":
    main()
