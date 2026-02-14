#!/usr/bin/env python
"""
Commonsense Reasoning Ablation Experiment
==========================================
Llama-2-7B에서 JELLY 하이퍼파라미터 민감도 분석
(Image Ablation Runner 구조와 통일)
"""

import sys
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base_runner import (
    BaseExperimentRunner,
    TrainingConfig,
    LoRAConfig,
    COMMONSENSE_TASKS,
    COMMONSENSE_ABLATION_CSV_COLUMNS,
    ABLATION_GRID
)


class CommonsenseAblationRunner(BaseExperimentRunner):
    """Commonsense Reasoning JELLY Ablation (병렬 GPU 지원)"""

    def __init__(self, seeds=None, gpus="0", per_gpu_tasks=1, test_mode=False,
                 tasks=None, output_dir=None,
                 training_config=None, lora_config=None,
                 use_wandb=True, wandb_project=None,
                 model="meta-llama/Llama-2-7b-hf"):
        super().__init__(
            experiment_name="commonsense_ablation",
            seeds=seeds,
            gpus=gpus,
            per_gpu_tasks=per_gpu_tasks,
            test_mode=test_mode,
            output_dir=output_dir,
            training_config=training_config,
            lora_config=lora_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project or "Llama2-Ablation",
        )
        self._tasks = tasks if tasks else COMMONSENSE_TASKS
        self._model = model

    @property
    def csv_columns(self):
        return COMMONSENSE_ABLATION_CSV_COLUMNS

    @property
    def tasks(self):
        return self._tasks

    def run_single_experiment(self, task, seed, vib, latent_stab, gpu_id=None):
        tc = self.training_config
        lc = self.lora_config

        cmd = [
            "python", "train_CS.py",
            "--adapter", "jelly",
            "--task", task,
            "--seed", str(seed),
            "--model", self._model,
            "--learning_rate", str(tc.learning_rate),
            "--batch", str(tc.batch_size),
            "--epochs", str(tc.epochs),
            "--weight_decay", str(tc.weight_decay),
            "--warmup_ratio", str(tc.warmup_ratio),
            "--r", str(lc.r),
            "--alpha", str(lc.alpha),
            "--lambda_vib", str(vib),
            "--lambda_latent_stability", str(latent_stab),
            "--wandb_project", self.wandb_project,
        ]

        if not self.use_wandb:
            cmd.append("--no_wandb")

        job_name = f"jelly_{task}_s{seed}_vib{vib}_lat{latent_stab}"

        if self.test_mode:
            dummy = self.get_dummy_result()
            time.sleep(0.3)
            self.update_progress(job_name)
            return dummy

        use_gpu = gpu_id if gpu_id else self.gpus
        ret_code, _, _ = self.run_subprocess_with_gpu(cmd, use_gpu, job_name)

        if ret_code != 0:
            return 0.0

        result_file = (
            self.result_dir /
            f"commonsense_result_{task}_s{seed}_vib{vib}_lat{latent_stab}.json"
        )

        if result_file.exists():
            with open(result_file) as f:
                score = json.load(f).get("best_accuracy", 0.0)
                self.update_progress(f"{job_name} = {score:.4f}")
                return score

        return 0.0

    def _job_executor(self, gpu_id, task, seed, vib, latent_stab):
        return {
            "task": task,
            "seed": seed,
            "vib": vib,
            "latent_stab": latent_stab,
            "score": self.run_single_experiment(
                task, seed, vib, latent_stab, gpu_id
            )
        }

    def run_ablation_for_param(self, param_type):
        grid = ABLATION_GRID[param_type]
        values, fixed = grid["values"], grid["fixed"]

        self.log(f"\n{'='*60}")
        self.log(f" {param_type.upper()} Ablation")
        self.log(f" values={values}, fixed={fixed}")
        self.log(f"{'='*60}")

        jobs = []
        for val in values:
            vib, latent_stab = (
                (val, fixed["latent_stab"])
                if param_type == "vib"
                else (fixed["vib"], val)
            )

            for seed in self.seeds:
                for task in self._tasks:
                    jobs.append({
                        "task": task,
                        "seed": seed,
                        "vib": vib,
                        "latent_stab": latent_stab
                    })

        results = self.execute_parallel_jobs(jobs, self._job_executor)

        grouped = defaultdict(lambda: defaultdict(dict))
        for r in results:
            key = (r["vib"], r["latent_stab"], r["seed"])
            grouped[key][r["task"]] = r["score"]

        for val in values:
            vib, latent_stab = (
                (val, fixed["latent_stab"])
                if param_type == "vib"
                else (fixed["vib"], val)
            )

            for seed in self.seeds:
                key = (vib, latent_stab, seed)
                task_scores = grouped[key]
                avg = self.calculate_average(task_scores)

                row = {
                    "seed": seed,
                    "vib": vib,
                    "latent_stab": latent_stab,
                    "avg": f"{avg*100:.2f}"
                }

                for t in COMMONSENSE_TASKS:
                    row[t] = f"{task_scores.get(t, 0.0)*100:.2f}"

                self.append_result(row)

    def run_all_experiments(self):
        self.save_metadata({
            "ablation_params": ["vib", "latent_stab"],
            "model": self._model
        })
        self.init_csv()

        for p in ["vib", "latent_stab"]:
            self.run_ablation_for_param(p)

        self.log(f"\n{'='*60}")
        self.log(" Commonsense Ablation 완료")
        self.log(f" 결과 CSV: {self.csv_path}")
        self.log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Commonsense Reasoning Ablation (병렬 GPU 지원)")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--per_gpu_tasks", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--param", type=str, default="all",
                        choices=["vib", "latent_stab", "all"])

    # wandb 설정
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Llama2-Ablation")

    # Training Config
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # LoRA Config
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj")

    # JELLY Config
    parser.add_argument("--jelly_mode", type=str, default="seq2par",
                        choices=["parallel", "sequential", "seq2par"])
    parser.add_argument("--switch_epoch", type=float, default=3.0)

    # Lambda params (baseline values, overridden by ablation grid)
    parser.add_argument("--lambda_vib", type=float, default=1.0)
    parser.add_argument("--lambda_latent_stab", type=float, default=1.0)

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",") if args.tasks else None
    use_wandb = not args.no_wandb

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    lora_config = LoRAConfig(
        r=args.r,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    runner = CommonsenseAblationRunner(
        seeds=seeds,
        gpus=args.gpus,
        per_gpu_tasks=args.per_gpu_tasks,
        test_mode=args.test,
        tasks=tasks,
        output_dir=args.output_dir,
        training_config=training_config,
        lora_config=lora_config,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        model=args.model,
    )

    if args.param == "all":
        runner.run_all_experiments()
    else:
        runner.save_metadata({"ablation_params": [args.param], "model": args.model})
        runner.init_csv()
        runner.run_ablation_for_param(args.param)


if __name__ == "__main__":
    main()
