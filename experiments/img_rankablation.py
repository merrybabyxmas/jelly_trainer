#!/usr/bin/env python
"""
Image Classification Rank Ablation Experiment
==============================================
ViT-B/16에서 다양한 rank에 대한 lora, lava, lava_fullweight 비교
병렬 GPU 실행 지원
"""

import os
import sys
import subprocess
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
    LAVAConfig,
    IMG_TASKS,
)

# Rank ablation용 CSV 컬럼
IMG_RANKABLATION_CSV_COLUMNS = [
    "method", "rank", "seed",
    "dtd", "eurosat", "gtsrb", "resisc45", "sun397", "svhn",
    "avg"
]


class ImageRankAblationRunner(BaseExperimentRunner):
    """Image Classification 태스크에서 Rank Ablation 실험 (병렬 GPU 실행 지원)"""

    def __init__(self, seeds=None, gpus="0", per_gpu_tasks=1, test_mode=False,
                 tasks=None, methods=None, ranks=None, output_dir=None,
                 training_config=None, lora_config=None, lava_config=None,
                 use_wandb=True, wandb_project=None):
        super().__init__(
            experiment_name="img_rankablation",
            seeds=seeds,
            gpus=gpus,
            per_gpu_tasks=per_gpu_tasks,
            test_mode=test_mode,
            output_dir=output_dir,
            training_config=training_config,
            lora_config=lora_config,
            lava_config=lava_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project or "IMG-RankAblation",
        )
        self._tasks = tasks if tasks else IMG_TASKS
        self._methods = methods if methods else ["lora", "lava", "lava_fullweight"]
        self._ranks = ranks if ranks else [4, 8, 12, 16]

    @property
    def csv_columns(self):
        return IMG_RANKABLATION_CSV_COLUMNS

    @property
    def tasks(self):
        return self._tasks

    def run_single_experiment(self, method: str, task: str, seed: int,
                              rank: int, gpu_id: str = None) -> float:
        """단일 실험 실행 (GPU ID 지정 가능)"""
        tc = self.training_config
        lc = self.lora_config
        lavc = self.lava_config

        cmd = [
            "python", "train_vit.py",
            "--adapter", method,
            "--task", task,
            "--seed", str(seed),
            "--learning_rate", str(tc.learning_rate),
            "--batch", str(tc.batch_size),
            "--epochs", str(tc.epochs),
            "--weight_decay", str(tc.weight_decay),
            "--warmup_ratio", str(tc.warmup_ratio),
            "--r", str(rank),
            "--alpha", str(rank),  # alpha = rank로 설정
            "--train_data_ratio", str(tc.train_data_ratio),
            "--wandb_project", self.wandb_project,
        ]

        # LAVA 계열은 lambda 값 추가
        if method in ["lava", "lava_fullweight"]:
            cmd.extend([
                "--lambda_vib", str(lavc.lambda_vib),
                "--lambda_latent_stability", str(lavc.lambda_latent_stability),
            ])

        if not self.use_wandb:
            cmd.append("--no_wandb")

        job_name = f"{method}_{task}_r{rank}_s{seed}"

        if self.test_mode:
            dummy = self.get_dummy_result()
            time.sleep(0.5)
            self.update_progress(job_name)
            return dummy

        use_gpu = gpu_id if gpu_id else self.gpus
        ret_code, stdout, stderr = self.run_subprocess_with_gpu(cmd, use_gpu, job_name)

        if ret_code != 0:
            return 0.0

        # 결과 파일 찾기
        if method in ["lava", "lava_fullweight"]:
            result_file = self.result_dir / f"img_result_{method}_{task}_s{seed}_vib{lavc.lambda_vib}_lat{lavc.lambda_latent_stability}.json"
        else:
            result_file = self.result_dir / f"img_result_{method}_{task}_r{rank}_s{seed}.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                score = data.get("best_accuracy", 0.0)
                self.update_progress(f"{job_name} = {score:.4f}")
                return score
        return 0.0

    def _job_executor(self, gpu_id: str, method: str, task: str,
                      seed: int, rank: int) -> dict:
        """병렬 작업 실행기"""
        score = self.run_single_experiment(method, task, seed, rank, gpu_id)
        return {
            "method": method, "task": task, "seed": seed,
            "rank": rank, "score": score
        }

    def run_rank_ablation(self):
        """Rank ablation 실험 (병렬 실행)"""
        self.log(f"")
        self.log(f"{'='*60}")
        self.log(f" Rank Ablation 시작")
        self.log(f" Methods: {self._methods}")
        self.log(f" Ranks: {self._ranks}")
        self.log(f" Tasks: {self._tasks}")
        self.log(f" Seeds: {self.seeds}")
        self.log(f"{'='*60}")

        # 모든 작업 생성
        jobs = []
        for method in self._methods:
            for rank in self._ranks:
                for seed in self.seeds:
                    for task in self._tasks:
                        jobs.append({
                            "method": method,
                            "task": task,
                            "seed": seed,
                            "rank": rank
                        })

        self.log(f"총 {len(jobs)}개 작업 실행 예정")

        # 병렬 실행
        results = self.execute_parallel_jobs(jobs, self._job_executor)

        # 결과 집계
        config_results = defaultdict(lambda: defaultdict(dict))
        for res in results:
            if res:
                key = (res["method"], res["rank"], res["seed"])
                config_results[key][res["task"]] = res["score"]

        # CSV에 기록
        for method in self._methods:
            for rank in self._ranks:
                for seed in self.seeds:
                    key = (method, rank, seed)
                    task_results = config_results[key]

                    avg = self.calculate_average(task_results)

                    row = {
                        "method": method,
                        "rank": rank,
                        "seed": seed,
                        "avg": f"{avg*100:.2f}"
                    }

                    for task in IMG_TASKS:
                        row[task] = f"{task_results.get(task, 0.0)*100:.2f}" if task in task_results else ""

                    self.append_result(row)

    def run_all_experiments(self):
        """모든 실험 실행"""
        self.save_metadata({
            "methods": self._methods,
            "ranks": self._ranks,
            "tasks": self._tasks
        })
        self.init_csv()
        self.run_rank_ablation()

        self.log(f"")
        self.log(f"{'='*60}")
        self.log(f" Image Rank Ablation 완료!")
        self.log(f" 결과: {self.csv_path}")
        self.log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Image Rank Ablation Experiment (병렬 GPU 지원)")
    parser.add_argument("--seeds", type=str, default="1,2,42")
    parser.add_argument("--gpus", type=str, default="0",
                        help="사용할 GPU ID (예: '0,1,2,3')")
    parser.add_argument("--per_gpu_tasks", type=int, default=1,
                        help="GPU당 동시 실행 작업 수")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tasks", type=str, default=None,
                        help="태스크 목록 (쉼표 구분)")
    parser.add_argument("--methods", type=str, default="lora,lava,lava_fullweight",
                        help="메소드 목록 (쉼표 구분)")
    parser.add_argument("--ranks", type=str, default="4,8,12,16",
                        help="Rank 값 목록 (쉼표 구분)")
    parser.add_argument("--output_dir", type=str, default=None)

    # wandb 설정
    parser.add_argument("--no_wandb", action="store_true", help="wandb 비활성화")
    parser.add_argument("--wandb_project", type=str, default="IMG-RankAblation")

    # Training Config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Data Ratio
    parser.add_argument("--train_data_ratio", type=int, default=100,
                        help="Percentage of training data to use (1-100)")

    # LAVA Lambda Config
    parser.add_argument("--lambda_vib", type=float, default=1.0)
    parser.add_argument("--lambda_latent_stab", type=float, default=1.0)

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",") if args.tasks else None
    methods = args.methods.split(",") if args.methods else None
    ranks = [int(r) for r in args.ranks.split(",")]
    use_wandb = not args.no_wandb

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        train_data_ratio=args.train_data_ratio,
    )

    lora_config = LoRAConfig(r=ranks[0], alpha=ranks[0])  # 기본값, 실제론 rank별로 달라짐
    lava_config = LAVAConfig(
        lambda_vib=args.lambda_vib,
        lambda_latent_stability=args.lambda_latent_stab
    )

    runner = ImageRankAblationRunner(
        seeds=seeds,
        gpus=args.gpus,
        per_gpu_tasks=args.per_gpu_tasks,
        test_mode=args.test,
        tasks=tasks,
        methods=methods,
        ranks=ranks,
        output_dir=args.output_dir,
        training_config=training_config,
        lora_config=lora_config,
        lava_config=lava_config,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
    )

    runner.run_all_experiments()


if __name__ == "__main__":
    main()
