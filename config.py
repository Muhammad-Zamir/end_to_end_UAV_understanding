"""
CAPITA Configuration File
Context-Aware Predictive Intent and Trajectory Attention for Anti-UAV Systems
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    # ── Dataset selection ──────────────────────────────────────────────────
    dataset_name: str = "MultiUAV"          # "MultiUAV" | "Anti-UAV" | "NPS"

    # ── MultiUAV paths ─────────────────────────────────────────────────────
    multiuav_train_frames: str = "/images/train"
    multiuav_train_labels: str = "/labels/train"
    multiuav_test_frames:  str = "/images/val"
    multiuav_test_labels:  str = "/labels/val"
    multiuav_train_json:   str = "/VQA/TD_UAV_new_QA_train.json"
    multiuav_test_json:    str = "/VQA/TD_UAV_new_QA_test.json"

    # ── Anti-UAV paths (update as needed) ─────────────────────────────────
    antiuav_train_frames: str = "/images/train"
    antiuav_train_labels: str = "/labels/train"
    antiuav_test_frames:  str = "/images/val"
    antiuav_test_labels:  str = "/labels/val"
    antiuav_train_json:   str = "/Anti_UAV_train_QA.json"
    antiuav_test_json:    str = "/Anti_UAV_val.json"

    # ── NPS paths (update as needed) ───────────────────────────────────────
    nps_train_frames: str = "/images/train"
    nps_train_labels: str = "/labels/train"
    nps_test_frames:  str = "/images/val"
    nps_test_labels:  str = "/labels/val"
    nps_train_json:   str = "/VQA/NPS_QA_train.json"
    nps_test_json:    str = "/VQA/NPS_QA_test.json"

    # ── Frame sampling ─────────────────────────────────────────────────────
    num_frames: int = 16                    # frames sampled per video
    sampling_strategy: str = "adaptive"    # "uniform" | "adaptive" (keyframe-biased)
    adaptive_threshold: float = 0.15       # bbox change threshold for keyframe detection

    # ── Image sizes ────────────────────────────────────────────────────────
    image_size_multiuav: int = 640
    image_size_antiuav:  int = 640
    image_size_nps:      int = 1280
    roi_patch_size: int = 64               # cropped ROI around each drone

    # ── Swarm limits ───────────────────────────────────────────────────────
    max_drones_per_frame: int = 35         # pad/truncate to this number
    min_drones_per_frame: int = 1

    # ── DataLoader ─────────────────────────────────────────────────────────
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    # ── Stage 1: Dual Stream Encoder ───────────────────────────────────────
    appearance_feat_dim: int = 256         # output dim of FFT appearance encoder
    motion_feat_dim: int = 128             # output dim of blur-signal motion encoder
    drone_feat_dim: int = 384              # appearance + motion fused

    # ── FFT Encoder ────────────────────────────────────────────────────────
    fft_channels: int = 32                 # FFT frequency channels kept
    cnn_channels: list = field(default_factory=lambda: [64, 128, 256])

    # ── Stage 2: Swarm Graph Temporal Network ──────────────────────────────
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4                 # graph attention heads
    temporal_hidden_dim: int = 512
    num_temporal_frames: int = 16
    swarm_feat_dim: int = 512              # output of swarm GNN

    # ── Stage 3: Causal Intent Reasoning ───────────────────────────────────
    causal_hidden_dim: int = 256
    causal_num_layers: int = 2
    intent_repr_dim: int = 768             # [swarm_feat | causal_feat]

    # ── Stage 4: TinyLlama + LoRA + Classification Heads ───────────────────
    tinyllama_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm_input_proj_dim: int = 2048         # TinyLlama hidden dim
    max_gen_length: int = 128              # max tokens for caption/motion generation
    max_input_length: int = 64             # max question token length
    n_prefix_tokens: int = 4              # number of soft visual prefix tokens injected

    # ── LoRA configuration ─────────────────────────────────────────────────
    # Low-Rank Adaptation: injects trainable rank-r matrices into attention layers
    # All base TinyLlama weights are FROZEN — only LoRA params + intent_proj updated
    use_lora: bool = True
    lora_r: int = 16                       # rank of LoRA decomposition
    lora_alpha: int = 32                   # LoRA scaling factor (alpha/r = effective LR scale)
    lora_dropout: float = 0.10             # dropout on LoRA layers
    use_8bit: bool = False                 # 8-bit quantization (saves VRAM, set True if OOM)
    # VRAM guide:
    #   use_8bit=False (fp16): TinyLlama ~2.2GB  (faster)
    #   use_8bit=True  (int8): TinyLlama ~0.6GB  (use if hitting OOM)

    # Classification head output sizes
    yes_no_classes: int = 2                # Yes / No
    uav_size_classes: int = 6             # tiny/small/medium/large/multiple/unknown
    environment_classes: int = 8          # open_sky/urban/forest/coastal/indoor/night/fog/unknown

    # ── Projection ─────────────────────────────────────────────────────────
    shared_proj_dim: int = 512


@dataclass
class TrainingConfig:
    # ── Basic ───────────────────────────────────────────────────────────────
    output_dir: str = "./checkpoints2"
    experiment_name: str = "capita_multiuav"
    seed: int = 42
    device: str = "cuda"

    # ── Batch & Epochs ─────────────────────────────────────────────────────
    batch_size: int = 2                    # 2 videos per step (16GB GPU)
    gradient_accumulation_steps: int = 8  # effective batch = 16
    num_epochs: int = 50
    warmup_epochs: int = 5

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    llm_learning_rate: float = 2e-5       # lower LR for TinyLlama fine-tuning
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── Scheduler ──────────────────────────────────────────────────────────
    scheduler: str = "cosine"             # "cosine" | "step" | "plateau"
    min_lr: float = 1e-6

    # ── Loss weights (λ1·L_caption + λ2·L_yes_no + λ3·L_size + λ4·L_motion + λ5·L_env) ──
    lambda_caption: float = 1.0
    lambda_yes_no: float = 1.0
    lambda_size: float = 0.3
    lambda_motion: float = 0.8
    lambda_environment: float = 0.3

    # ── Mixed precision ────────────────────────────────────────────────────
    use_amp: bool = True                   # automatic mixed precision (fp16)
    amp_dtype: str = "float16"

    # ── Checkpointing ──────────────────────────────────────────────────────
    save_every_n_epochs: int = 5
    save_best_metric: str = "val_loss"    # metric to track for best model
    keep_last_n_checkpoints: int = 3

    # ── Logging ────────────────────────────────────────────────────────────
    log_every_n_steps: int = 10
    use_wandb: bool = False               # set True if wandb installed
    wandb_project: str = "CAPITA-AntiUAV"

    # ── Validation ─────────────────────────────────────────────────────────
    val_every_n_epochs: int = 2


@dataclass
class CAPITAConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def get_image_size(self) -> int:
        sizes = {
            "MultiUAV": self.data.image_size_multiuav,
            "Anti-UAV": self.data.image_size_antiuav,
            "NPS":      self.data.image_size_nps,
        }
        return sizes[self.data.dataset_name]

    def get_dataset_paths(self) -> dict:
        name = self.data.dataset_name
        if name == "MultiUAV":
            return {
                "train_frames": self.data.multiuav_train_frames,
                "train_labels": self.data.multiuav_train_labels,
                "test_frames":  self.data.multiuav_test_frames,
                "test_labels":  self.data.multiuav_test_labels,
                "train_json":   self.data.multiuav_train_json,
                "test_json":    self.data.multiuav_test_json,
            }
        elif name == "Anti-UAV":
            return {
                "train_frames": self.data.antiuav_train_frames,
                "train_labels": self.data.antiuav_train_labels,
                "test_frames":  self.data.antiuav_test_frames,
                "test_labels":  self.data.antiuav_test_labels,
                "train_json":   self.data.antiuav_train_json,
                "test_json":    self.data.antiuav_test_json,
            }
        elif name == "NPS":
            return {
                "train_frames": self.data.nps_train_frames,
                "train_labels": self.data.nps_train_labels,
                "test_frames":  self.data.nps_test_frames,
                "test_labels":  self.data.nps_test_labels,
                "train_json":   self.data.nps_train_json,
                "test_json":    self.data.nps_test_json,
            }
        else:
            raise ValueError(f"Unknown dataset: {name}")