from transformers import Trainer
import torch
from typing import Dict

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class JellyBaseTrainer(Trainer):
    """
    JELLY Trainer with Task-Aware Subspace Initialization (TASI)
    
    파이프라인:
    1. 메인 학습 전 trainer.run_probe() 호출
    2. probe_steps 만큼 가벼운 별도의 옵티마이저로 직렬(Sequential) 정찰 진행
    3. 정찰 종료 직후 SVD를 통한 기저 추출 및 병렬(Parallel) 어댑터 초기화 (TASI)
    4. 이후 trainer.train()을 통해 완벽히 깨끗한 상태에서 0 에폭부터 병렬 학습 시작
    """

    def __init__(self, *args, probe_steps=200, probe_init_scale=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_steps = int(probe_steps)
        self.probe_init_scale = probe_init_scale
        self.jelly_layers = []

    def _ensure_layers_cached(self):
        if self.jelly_layers:
            return
        for m in self.model.modules():
            if hasattr(m, "initialize_from_probe") and hasattr(m, "set_mode"):
                self.jelly_layers.append(m)

    def run_probe(self):
        """본 학습(trainer.train) 전에 실행되는 독립된 Probing (Pre-train) 루프"""
        self._ensure_layers_cached()

        if self.probe_steps <= 0:
            print(">>> [TASI] probe_steps=0, skipping probe phase.")
            for layer in self.jelly_layers:
                layer.set_mode("parallel")
            return

        print(f"\n{'='*60}")
        print(f">>> [TASI] Phase 1: Starting Pre-train Probe (Steps: {self.probe_steps})")

        # 1. 모드 설정: Square 레이어는 직렬(Sequential)로 정찰
        n_seq, n_par = 0, 0
        for layer in self.jelly_layers:
            if getattr(layer, "is_square", False):
                layer.set_mode("sequential")
                n_seq += 1
            else:
                layer.set_mode("parallel")
                n_par += 1
        print(f">>> [TASI] Probe modes configured: {n_seq} Sequential, {n_par} Parallel")

        # 2. 정찰용 경량 학습 루프 구성
        train_dataloader = self.get_train_dataloader()
        
        # [수정] 메인 학습과 무관한 '정찰 전용' 일회용 옵티마이저 생성
        # frozen params가 옵티마이저에 들어가는 것을 방지하기 위해 requires_grad=True인 파라미터만 필터링합니다.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        probe_optimizer = torch.optim.AdamW(trainable_params, lr=self.args.learning_rate)

        self.model.train()
        step = 0
        
        for epoch in range(999): # probe_steps 도달 시 탈출
            for batch in train_dataloader:
                if step >= self.probe_steps:
                    break
                
                batch = self._prepare_inputs(batch)
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                
                loss.backward()
                probe_optimizer.step()
                probe_optimizer.zero_grad()
                
                step += 1
                if step % max(1, (self.probe_steps // 5)) == 0 or step == self.probe_steps:
                    print(f"  - Probe Step {step}/{self.probe_steps} | Loss: {loss.item():.4f}")
            
            if step >= self.probe_steps:
                break

        # 3. 정찰 종료 후 방향성 추출 및 병렬 안착
        self._execute_tasi()

    def _execute_tasi(self):
        """정찰된 방향성으로 기저를 정렬하고 모드를 병렬(Parallel)로 고정"""
        print(f"\n>>> [TASI] Phase 2: Executing Extraction & Basis Alignment")
        n_initialized = 0
        for layer in self.jelly_layers:
            # SVD 기저 추출 및 A_par(방향), B_par(Zero) 초기화
            layer.initialize_from_probe(init_scale=self.probe_init_scale)
            # 초기화가 끝난 레이어는 완벽한 병렬 모드로 전환
            layer.set_mode("parallel")
            n_initialized += 1

        # [수정] Weight Verification & Residual Gradient Clearance
        # 정찰 단계에서 발생한 유령 그래디언트가 본 학습(Main Training)으로 넘어가지 않도록 완벽히 삭제합니다.
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None

        print(f">>> [TASI] Done! {n_initialized} layers aligned. Residual gradients cleared. Ready for Main Training.")
        print(f"{'='*60}\n")