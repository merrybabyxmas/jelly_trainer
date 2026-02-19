#!/usr/bin/env python
"""
GLUE Comparison Experiment
==========================
JELLY와 다른 메소드(BitFit, LoRA, AdaLoRA, DoRA, PiSSA) 비교 실험
병렬 GPU 실행 지원
"""

import os
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
    JELLYConfig,
    GLUE_TASKS,
    GLUE_CSV_COLUMNS,
    COMPARISON_METHODS
)


class GLUEComparisonRunner(BaseExperimentRunner):
    """GLUE 태스크에서 메소드 비교 실험 (병렬 GPU 실행 지원)"""

    def __init__(self, seeds=None, gpus="0", per_gpu_tasks=1, test_mode=False,
                 tasks=None, methods=None, output_dir=None,
                 training_config=None, lora_config=None, jelly_config=None,
                 use_wandb=True, wandb_project=None, model_name="microsoft/deberta-v3-base"):
        super().__init__(
            experiment_name="glue_comparison",
            seeds=seeds,
            gpus=gpus,
            per_gpu_tasks=per_gpu_tasks,
            test_mode=test_mode,
            output_dir=output_dir,
            training_config=training_config,
            lora_config=lora_config,
            jelly_config=jelly_config,
            use_wandb=use_wandb,
            wandb_project=wandb_project or "GLUE-Comparison",
        )
        self._tasks = tasks if tasks else GLUE_TASKS
        self._methods = methods if methods else COMPARISON_METHODS
        self.model_name = model_name

    @property
    def csv_columns(self):
        return GLUE_CSV_COLUMNS

    @property
    def tasks(self):
        return self._tasks

    def run_single_experiment(self, method: str, task: str, seed: int,
                               gpu_id: str = None) -> dict:
        """단일 실험 실행 (GPU ID 지정 가능)

        Returns:
            dict: {"score": float, "oom": bool, "method": str, "task": str, "seed": int}
        """
        # train_nlu.py 실행 명령어 구성
        cmd = [
            "python", "train_nlu.py",
            "--adapter", method,
            "--task", task,
            "--seed", str(seed),
            "--model", self.model_name,
        ] + self.build_training_args(method)

        job_name = f"{method}_{task}_s{seed}"
        result = {"method": method, "task": task, "seed": seed, "score": 0.0, "oom": False}

        if self.test_mode:
            result["score"] = self.get_dummy_result()
            time.sleep(0.5)
            self.update_progress(job_name)
            return result

        use_gpu = gpu_id if gpu_id else self.gpus
        ret_code, stdout, stderr = self.run_subprocess_with_gpu(cmd, use_gpu, job_name)

        # OOM 감지 (SIGKILL = -9)
        if ret_code == -9:
            result["oom"] = True
            self.log(f"[OOM] {job_name} - GPU {use_gpu}에서 OOM 발생", "WARN")
            return result

        if ret_code != 0:
            return result

        # 결과 파일 경로 (train_nlu.py 저장 형식: glue_result_{method}_{task}_r{r}_s{seed}.json)
        result_file = self.result_dir / f"glue_result_{method}_{task}_r{self.lora_config.r}_s{seed}.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                result["score"] = data.get("best_score", 0.0)
                self.update_progress(f"{job_name} = {result['score']:.4f}")
        
        return result

    def _job_executor(self, gpu_id: str, method: str, task: str, seed: int) -> dict:
        """병렬 작업 실행기"""
        return self.run_single_experiment(method, task, seed, gpu_id)

    def get_params_percentage(self, method: str) -> str:
        """메소드별 파라미터 비율 (DeBERTa-v3-base 기준 대략적 수치)"""
        params_map = {
            "bitfit": "0.10",
            "lora": "0.60", # DeBERTa는 모든 Linear에 붙이면 좀 더 큼
            "adalora": "0.60",
            "dora": "0.60",
            "pissa": "0.60",
            "jelly": "0.60"
        }
        return params_map.get(method, "-")

    def run_all_experiments(self):
        """모든 비교 실험 병렬 실행"""
        self.save_metadata({"methods": self._methods, "model": self.model_name})
        self.init_csv()

        # 모든 실험 작업 생성
        jobs = []
        for method in self._methods:
            for seed in self.seeds:
                for task in self._tasks:
                    jobs.append({
                        "method": method,
                        "task": task,
                        "seed": seed
                    })

        self.log(f"총 {len(jobs)}개 실험 준비 완료")
        self.log(f"Model: {self.model_name}")
        self.log(f"Methods: {self._methods}")
        self.log(f"Tasks: {self._tasks}")
        
        # 병렬 실행
        results = self.execute_parallel_jobs(jobs, self._job_executor)

        # 결과 집계 (method -> task -> [scores])
        method_results = defaultdict(lambda: defaultdict(list))
        for res in results:
            if res and not res.get("oom", False) and res.get("score", 0) > 0:
                method_results[res["method"]][res["task"]].append(res["score"])

        # CSV에 메소드별 결과 기록
        for method in self._methods:
            task_results = method_results[method]

            row = {
                "method": method.upper(),
                "params(%)": self.get_params_percentage(method),
            }

            all_means = []
            for task in GLUE_TASKS:
                if task in task_results and task_results[task]:
                    scores = task_results[task]
                    mean = sum(scores) / len(scores)
                    std = self.calculate_std(scores)
                    row[task] = self.format_result(mean, std)
                    all_means.append(mean)
                else:
                    row[task] = ""

            if all_means:
                row["avg"] = f"{sum(all_means)/len(all_means)*100:.2f}"

            self.append_result(row)

        self.log(f"")
        self.log(f"{'='*60}")
        self.log(f" GLUE Comparison 완료!")
        self.log(f" 결과: {self.csv_path}")
        self.log(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="GLUE Comparison (병렬 GPU 지원)")
    parser.add_argument("--seeds", type=str, default="1,2,42")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--per_gpu_tasks", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    # Model
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")

    # wandb 설정
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="GLUE-Comparison")

    # Training Config
    parser.add_argument("--lr", type=float, default=None, help="None이면 태스크별 기본값 사용")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # LoRA Config
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="query_proj,key_proj,value_proj,dense")

    # JELLY Config
    parser.add_argument("--jelly_mode", type=str, default="dynamic",
                        choices=["parallel", "sequential", "seq2par", "dynamic"])
    parser.add_argument("--switch_epoch", type=int, default=1)

    # Data Ratio
    parser.add_argument("--train_data_ratio", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Max training samples (0=no limit)")

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = args.tasks.split(",") if args.tasks else None
    methods = args.methods.split(",") if args.methods else None
    use_wandb = not args.no_wandb

    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        train_data_ratio=args.train_data_ratio,
        max_train_samples=args.max_train_samples,
    )

    lora_config = LoRAConfig(
        r=args.r,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    jelly_config = JELLYConfig(
        jelly_mode=args.jelly_mode,
        switch_epoch=args.switch_epoch,
    )

    runner = GLUEComparisonRunner(
        seeds=seeds,
        gpus=args.gpus,
        per_gpu_tasks=args.per_gpu_tasks,
        test_mode=args.test,
        tasks=tasks,
        methods=methods,
        output_dir=args.output_dir,
        training_config=training_config,
        lora_config=lora_config,
        jelly_config=jelly_config,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        model_name=args.model
    )

    runner.run_all_experiments()


if __name__ == "__main__":
    main()