"""
CAPITA Model Architecture
Context-Aware Predictive Intent and Trajectory Attention for Anti-UAV Systems

Stages:
  1. Dual-Stream Encoder  (FFT Appearance + Blur-Signal Motion)
  2. Swarm Graph Temporal Network
  3. Causal Intent Reasoning Module (CTRM)
  4. Multi-Task QA-Aware Output Heads (4 heads + TinyLlama)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

from config import CAPITAConfig


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1-A: FREQUENCY-DOMAIN APPEARANCE ENCODER
# ═════════════════════════════════════════════════════════════════════════════

class FFTAppearanceEncoder(nn.Module):
    """
    Encodes full frames using pixel + FFT streams, then extracts
    drone-region features via box-guided spatial attention.

    This gives the model BOTH:
    - Global scene context (environment, sky, terrain)
    - Drone-local appearance (rotor harmonics, blur signature)

    Without adding any new module to the architecture.

    Input:  frames     [B, T, 3, H, W]       full frames
            boxes      [B, T, M, 5]           drone boxes (cx,cy,w,h)
            drone_mask [B, T, M]
    Output: [B, T, M, appearance_feat_dim]
    """

    def __init__(self, cfg: CAPITAConfig):
        super().__init__()
        self.roi_size = cfg.data.roi_patch_size
        self.out_dim  = cfg.model.appearance_feat_dim
        ch = cfg.model.cnn_channels   # [64, 128, 256]

        # ── Pixel-space CNN branch (processes full frame) ─────────────────
        self.pixel_cnn = nn.Sequential(
            nn.Conv2d(3,     ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.ReLU(),
            nn.Conv2d(ch[0], ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.BatchNorm2d(ch[1]), nn.ReLU(),
            nn.Conv2d(ch[1], ch[1], 3, padding=1), nn.BatchNorm2d(ch[1]), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.BatchNorm2d(ch[2]), nn.ReLU(),
            # No AdaptiveAvgPool here — keep spatial dims for ROI extraction
        )

        # ── FFT frequency branch ──────────────────────────────────────────
        self.fft_cnn = nn.Sequential(
            nn.Conv2d(3,     ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.BatchNorm2d(ch[1]), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.BatchNorm2d(ch[2]), nn.ReLU(),
            # No pool — keep spatial dims
        )

        # ── Global scene pooling (NEW — gives environment context) ────────
        # Pools full feature map → single scene vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Drone-local ROI pooling ────────────────────────────────────────
        # Extracts features at each drone's box location from feature map
        self.roi_pool = nn.AdaptiveAvgPool2d(1)

        # ── Fusion: drone_local[256] + scene_global[256] + fft_local[256] → 256
        self.fusion = nn.Sequential(
            nn.Linear(ch[2] * 3, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def compute_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 3, H, W] → [N, 3, H, W] log-magnitude spectrum"""
        x_float = x.float()
        fft_channels = []
        for c in range(x.shape[1]):
            channel = x_float[:, c:c+1, :, :]
            fft2    = torch.fft.fft2(channel, norm='ortho')
            fft2    = torch.fft.fftshift(fft2, dim=(-2, -1))
            magnitude = torch.log1p(torch.abs(fft2))
            fft_channels.append(magnitude)
        return torch.cat(fft_channels, dim=1)

    def _extract_roi_features(
        self,
        feat_map: torch.Tensor,   # [N, C, H, W]  feature map
        boxes: torch.Tensor,      # [N, M, 4]     (cx,cy,w,h) normalized 0-1
        drone_mask: torch.Tensor, # [N, M]
    ) -> torch.Tensor:
        """
        Extract feature map crops at each drone box location.
        Returns: [N, M, C]
        """
        N, C, H, W = feat_map.shape
        M = boxes.shape[1]
        roi_feats = torch.zeros(N, M, C, device=feat_map.device,
                                dtype=feat_map.dtype)

        for m in range(M):
            # Convert normalized cx,cy,w,h → pixel coords in feature map
            cx = boxes[:, m, 0] * W   # [N]
            cy = boxes[:, m, 1] * H
            bw = (boxes[:, m, 2] * W).clamp(min=2)
            bh = (boxes[:, m, 3] * H).clamp(min=2)

            x1 = (cx - bw/2).long().clamp(0, W-1)
            y1 = (cy - bh/2).long().clamp(0, H-1)
            x2 = (cx + bw/2).long().clamp(1, W)
            y2 = (cy + bh/2).long().clamp(1, H)

            for n in range(N):
                if drone_mask[n, m] < 0.5:
                    continue
                crop = feat_map[n:n+1, :,
                                y1[n]:y2[n].clamp(min=y1[n]+1),
                                x1[n]:x2[n].clamp(min=x1[n]+1)]
                roi_feats[n, m] = self.roi_pool(crop).view(C)

        return roi_feats  # [N, M, C]

    def forward(
        self,
        frames: torch.Tensor,      # [B, T, 3, H, W]  ← full frames now
        boxes: torch.Tensor,       # [B, T, M, 5]
        drone_mask: torch.Tensor,  # [B, T, M]
    ) -> torch.Tensor:
        """Returns: [B, T, M, appearance_feat_dim]"""
        B, T, M, _ = boxes.shape[:4]
        _, _, C, H, W = frames.shape

        # Flatten B,T for batch processing
        frames_flat = frames.view(B*T, C, H, W)
        ref = next(self.pixel_cnn.parameters())
        frames_flat = frames_flat.to(device=ref.device, dtype=ref.dtype)
        # Downsample full frame to 128x128 before CNN — reduces memory 25x
        frames_flat = F.interpolate(frames_flat, size=(128, 128),
                                    mode='bilinear', align_corners=False)

        boxes_flat      = boxes.view(B*T, M, 5)
        mask_flat       = drone_mask.view(B*T, M)
        boxes_norm      = boxes_flat[:, :, 1:5]  # cx,cy,w,h (drop class)

        # ── Pixel feature map [B*T, 256, H/4, W/4] ───────────────────────
        pixel_feat_map = self.pixel_cnn(frames_flat)

        # ── FFT feature map ───────────────────────────────────────────────
        fft_input    = self.compute_fft_features(frames_flat).to(
            device=ref.device, dtype=ref.dtype)
        fft_feat_map = self.fft_cnn(fft_input)

        # ── Global scene feature (environment context) ────────────────────
        # Average pool entire feature map → captures sky/terrain/background
        scene_feat = self.global_pool(pixel_feat_map).view(B*T, -1)  # [BT, 256]
        scene_feat = scene_feat.unsqueeze(1).expand(-1, M, -1)       # [BT, M, 256]

        # ── Drone-local ROI features ──────────────────────────────────────
        # Extract features at each drone's spatial location
        pixel_roi = self._extract_roi_features(
            pixel_feat_map, boxes_norm, mask_flat)   # [BT, M, 256]

        fft_roi   = self._extract_roi_features(
            fft_feat_map, boxes_norm, mask_flat)     # [BT, M, 256]

        # # ABLATION 1a: no FFT — replace fft_roi with zeros, keep scene context
        # fft_roi = torch.zeros_like(pixel_roi)

        # ── Fuse: local_pixel + local_fft + global_scene ──────────────────
        fused = self.fusion(
            torch.cat([pixel_roi, fft_roi, scene_feat], dim=-1)
        )  # [BT, M, 256]

        return fused.view(B, T, M, self.out_dim)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1-B: BLUR-SIGNAL MOTION ENCODER
# ═════════════════════════════════════════════════════════════════════════════

class BlurSignalMotionEncoder(nn.Module):
    """
    Extracts motion features from consecutive frame pairs.
    Key insight: Motion blur direction = heading, blur magnitude = speed.
    Treats blur as signal, not noise.

    Input:  [B, T, M, 3, roi_size, roi_size]  (ROI patches)
            [B, T, M, 5]                       (YOLO boxes for trajectory)
    Output: [B, T, M, motion_feat_dim]
    """

    def __init__(self, cfg: CAPITAConfig):
        super().__init__()
        rs            = cfg.data.roi_patch_size
        self.out_dim  = cfg.model.motion_feat_dim
        self.num_frames = cfg.data.num_frames

        # ── Blur estimation network ───────────────────────────────────────
        # Estimates blur kernel parameters from consecutive ROI pairs
        self.blur_estimator = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),   # 6 = 2 × RGB frames concat
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                       # → [B,64,4,4]
            nn.Flatten(),                                  # → [B,1024]
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 4),                            # [blur_mag, blur_dir_x, blur_dir_y, confidence]
        )

        # ── Trajectory feature from YOLO boxes ───────────────────────────
        # Input: [delta_cx, delta_cy, delta_w, delta_h, speed, accel] = 6 dims
        self.trajectory_mlp = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 32),
        )

        # ── Counterfactual memory ─────────────────────────────────────────
        # Stores intent-conditioned past features
        # Lightweight GRU per drone tracks its own history
        self.memory_gru = nn.GRU(
            input_size=4 + 32,    # blur_params + traj_feat
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(64, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
        )

    def _compute_trajectory_features(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory features from YOLO boxes across frames.
        boxes: [B, T, M, 5]
        Returns: [B, T, M, 6] — [delta_cx, delta_cy, delta_w, delta_h, speed, accel]
        """
        B, T, M, _ = boxes.shape
        cx = boxes[:, :, :, 1]   # [B,T,M]
        cy = boxes[:, :, :, 2]
        bw = boxes[:, :, :, 3]
        bh = boxes[:, :, :, 4]

        # Pad first frame delta with zeros
        zeros = torch.zeros(B, 1, M, device=boxes.device, dtype=boxes.dtype)

        delta_cx = torch.diff(cx, dim=1)  # [B,T-1,M]
        delta_cy = torch.diff(cy, dim=1)
        delta_w  = torch.diff(bw, dim=1)
        delta_h  = torch.diff(bh, dim=1)

        # Speed: Euclidean displacement
        speed = torch.sqrt(delta_cx**2 + delta_cy**2 + 1e-6)

        # Acceleration: change in speed
        accel = torch.diff(speed, dim=1)  # [B,T-2,M]

        # Pad to [B,T,M]
        delta_cx = torch.cat([zeros, delta_cx], dim=1)
        delta_cy = torch.cat([zeros, delta_cy], dim=1)
        delta_w  = torch.cat([zeros, delta_w],  dim=1)
        delta_h  = torch.cat([zeros, delta_h],  dim=1)
        speed    = torch.cat([zeros, speed],    dim=1)
        accel    = torch.cat([zeros, zeros, accel], dim=1)[:, :T, :]

        return torch.stack([delta_cx, delta_cy, delta_w, delta_h, speed, accel], dim=-1)

    def forward(
        self,
        roi_patches: torch.Tensor,  # [B, T, M, 3, rs, rs]
        boxes: torch.Tensor,        # [B, T, M, 5]
        drone_mask: torch.Tensor,   # [B, T, M]
    ) -> torch.Tensor:
        """Returns: [B, T, M, motion_feat_dim]"""
        B, T, M, C, rs, _ = roi_patches.shape

        # Cast inputs to model dtype (fp16 under autocast)
        model_dtype = next(self.blur_estimator.parameters()).dtype
        roi_patches = roi_patches.to(dtype=model_dtype)
        boxes       = boxes.to(dtype=model_dtype)

        # ── Blur features from consecutive frame pairs ─────────────────
        # Compute difference between frame t and t-1 for each drone ROI
        roi_flat = roi_patches.view(B * T * M, C, rs, rs)

        # Shift: frame t-1 (pad first frame by repeating)
        roi_prev = torch.cat([
            roi_patches[:, :1],   # repeat first frame
            roi_patches[:, :-1]
        ], dim=1).view(B * T * M, C, rs, rs)

        # Concatenate current and previous ROIs → temporal context
        roi_pair = torch.cat([roi_flat, roi_prev], dim=1)  # [BTM,6,rs,rs]
        blur_params = self.blur_estimator(roi_pair)        # [BTM,4]
        blur_params = blur_params.view(B, T, M, 4)

        # ── Trajectory features ────────────────────────────────────────
        traj_feats = self._compute_trajectory_features(boxes)  # [B,T,M,6]
        traj_feats = self.trajectory_mlp(
            traj_feats.view(B * T * M, 6)
        ).view(B, T, M, 32)

        # ── Combine and pass through memory GRU ───────────────────────
        combined = torch.cat([blur_params, traj_feats], dim=-1)  # [B,T,M,36]

        # # ABLATION: trajectory only — zero out blur params
        # blur_zeros = torch.zeros_like(blur_params)
        # combined = torch.cat([blur_zeros, traj_feats], dim=-1)

        # Process each drone's temporal sequence through GRU
        # Reshape: [B*M, T, 36]
        combined_gru = combined.permute(0, 2, 1, 3).contiguous()  # [B,M,T,36]
        combined_gru = combined_gru.view(B * M, T, 36)

        gru_out, _ = self.memory_gru(combined_gru)  # [B*M, T, 64]
        gru_out    = gru_out.view(B, M, T, 64).permute(0, 2, 1, 3)  # [B,T,M,64]

        # Project to output dim
        motion_feat = self.output_proj(gru_out)  # [B,T,M,128]

        return motion_feat


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: DUAL STREAM ENCODER (combines FFT + Blur)
# ═════════════════════════════════════════════════════════════════════════════

class DualStreamEncoder(nn.Module):
    """
    Combines FFT Appearance + Blur-Signal Motion encoders.
    Output: [B, T, M, drone_feat_dim=384]
    """

    def __init__(self, cfg: CAPITAConfig):
        super().__init__()
        self.appearance_enc = FFTAppearanceEncoder(cfg)
        self.motion_enc     = BlurSignalMotionEncoder(cfg)

        # Fusion: [256 + 128] → 384
        self.fusion = nn.Sequential(
            nn.Linear(cfg.model.appearance_feat_dim + cfg.model.motion_feat_dim,
                      cfg.model.drone_feat_dim),
            nn.LayerNorm(cfg.model.drone_feat_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        frames: torch.Tensor,      # [B, T, 3, H, W]
        roi_patches: torch.Tensor,  # [B,T,M,3,rs,rs]
        boxes: torch.Tensor,        # [B,T,M,5]
        drone_mask: torch.Tensor,   # [B,T,M]
    ) -> torch.Tensor:
        """Returns: [B, T, M, drone_feat_dim]"""
        app_feat  = self.appearance_enc(frames, boxes, drone_mask)                   # [B,T,M,256]
        mot_feat  = self.motion_enc(roi_patches, boxes, drone_mask)    # [B,T,M,128]
        fused     = self.fusion(torch.cat([app_feat, mot_feat], dim=-1))  # [B,T,M,384]

        # Zero out padded drones
        mask = drone_mask.unsqueeze(-1)  # [B,T,M,1]
        fused = fused * mask

        return fused


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: SWARM GRAPH TEMPORAL NETWORK
# ═════════════════════════════════════════════════════════════════════════════

class SwarmGraphBuilder:
    """
    Builds per-frame graphs from drone features and spatial positions.
    Nodes: drones  |  Edges: spatial proximity + relative velocity
    """

    @staticmethod
    def build_graph(
        drone_feats: torch.Tensor,  # [M, drone_feat_dim]
        boxes: torch.Tensor,        # [M, 5] (class, cx, cy, w, h)
        drone_mask: torch.Tensor,   # [M]
        dist_threshold: float = 0.3,
    ) -> Data:
        """Build a single-frame graph."""
        device   = drone_feats.device
        valid_idx = (drone_mask > 0.5).nonzero(as_tuple=True)[0]
        n_valid  = len(valid_idx)

        if n_valid == 0:
            # Empty graph — return dummy
            return Data(
                x=torch.zeros(1, drone_feats.shape[-1], device=device),
                edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
                edge_attr=torch.zeros(0, 3, device=device),
            )

        x = drone_feats[valid_idx]                    # [n_valid, feat_dim]
        positions = boxes[valid_idx, 1:3]             # [n_valid, 2] (cx, cy)

        # Build edges based on spatial proximity
        edges_src, edges_dst, edge_attrs = [], [], []

        for i in range(n_valid):
            for j in range(n_valid):
                if i == j:
                    continue
                dist = torch.norm(positions[i] - positions[j])
                if dist < dist_threshold:
                    edges_src.append(i)
                    edges_dst.append(j)
                    # Edge features: [distance, rel_dx, rel_dy]
                    rel = positions[j] - positions[i]
                    edge_attrs.append(torch.tensor(
                        [dist.item(), rel[0].item(), rel[1].item()],
                        device=device
                    ))

        if not edges_src:
            # No edges — self-loops for isolated drones
            edges_src = list(range(n_valid))
            edges_dst = list(range(n_valid))
            edge_attrs = [torch.zeros(3, device=device)] * n_valid

        edge_index = torch.tensor([edges_src, edges_dst],
                                   dtype=torch.long, device=device)
        edge_attr  = torch.stack(edge_attrs, dim=0)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class SpatioTemporalGNN(nn.Module):
    """
    Processes a sequence of drone graphs over T frames.

    Spatial:  GAT attention within each frame graph
    Temporal: Transformer attention over frame sequence

    Input:  [B, T, M, drone_feat_dim]
            [B, T, M, 5]  boxes
            [B, T, M]     drone_mask
    Output: [B, swarm_feat_dim]
    """

    def __init__(self, cfg: CAPITAConfig):
        super().__init__()
        in_dim    = cfg.model.drone_feat_dim    # 384
        gnn_dim   = cfg.model.gnn_hidden_dim    # 256
        n_heads   = cfg.model.gnn_num_heads     # 4
        n_layers  = cfg.model.gnn_num_layers    # 3
        temp_dim  = cfg.model.temporal_hidden_dim  # 512
        T         = cfg.model.num_temporal_frames  # 16
        out_dim   = cfg.model.swarm_feat_dim    # 512

        # ── Spatial GAT layers ─────────────────────────────────────────────
        self.input_proj = nn.Linear(in_dim, gnn_dim)

        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=gnn_dim,
                out_channels=gnn_dim // n_heads,
                heads=n_heads,
                edge_dim=3,
                concat=True,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])

        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(gnn_dim) for _ in range(n_layers)
        ])

        # ── Temporal Transformer ───────────────────────────────────────────
        self.temporal_proj = nn.Linear(gnn_dim, temp_dim)
        self.pos_embedding  = nn.Parameter(torch.randn(1, T, temp_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temp_dim,
            nhead=8,
            dim_feedforward=temp_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Learned temporal attention pooling ────────────────────────────
        self.temporal_attn_pool = nn.Sequential(
            nn.Linear(temp_dim, 1),
            nn.Softmax(dim=1),
        )

        # ── Output projection ──────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(temp_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def _process_frame_graph(
        self,
        drone_feats: torch.Tensor,  # [M, drone_feat_dim]
        boxes: torch.Tensor,        # [M, 5]
        drone_mask: torch.Tensor,   # [M]
    ) -> torch.Tensor:
        """Apply GAT to a single frame's graph → single frame vector."""
        device      = drone_feats.device
        model_dtype = next(self.input_proj.parameters()).dtype
        drone_feats = drone_feats.to(dtype=model_dtype)
        boxes       = boxes.to(dtype=model_dtype)

        graph = SwarmGraphBuilder.build_graph(drone_feats, boxes, drone_mask)
        graph = graph.to(device)
        graph.x         = graph.x.to(dtype=model_dtype)
        graph.edge_attr = graph.edge_attr.to(dtype=model_dtype)

        x = self.input_proj(graph.x)  # [n_valid, gnn_dim]

        for gat, norm in zip(self.gat_layers, self.gat_norms):
            residual = x
            x = gat(x, graph.edge_index, graph.edge_attr)
            x = norm(x + residual)
            x = F.relu(x)

        # Pool nodes → single frame representation
        frame_feat = x.mean(dim=0)  # [gnn_dim]
        return frame_feat

    def forward(
        self,
        drone_feats: torch.Tensor,  # [B, T, M, drone_feat_dim]
        boxes: torch.Tensor,        # [B, T, M, 5]
        drone_mask: torch.Tensor,   # [B, T, M]
    ) -> torch.Tensor:
        """Returns: [B, swarm_feat_dim]"""
        B, T, M, _ = drone_feats.shape
        device = drone_feats.device

        # ── Process each frame graph per sample ────────────────────────────
        model_dtype = next(self.temporal_proj.parameters()).dtype
        frame_feats = torch.zeros(B, T, self.temporal_proj.in_features,
                                   device=device, dtype=model_dtype)

        for b in range(B):
            for t in range(T):
                feat = self._process_frame_graph(
                    drone_feats[b, t],
                    boxes[b, t],
                    drone_mask[b, t],
                )
                frame_feats[b, t] = feat

        # # ABLATION: no GNN — simple masked mean pool over drones per frame
        # for b in range(B):
        #     for t in range(T):
        #         mask = drone_mask[b, t].unsqueeze(-1)        # [M, 1]
        #         feats = drone_feats[b, t] * mask             # [M, D]
        #         n_valid = mask.sum().clamp(min=1)
        #         mean_feat = feats.sum(dim=0) / n_valid       # [D]
        #         # Project to gnn_dim to keep temporal_proj input size same
        #         frame_feats[b, t] = self.input_proj(
        #             mean_feat.to(dtype=model_dtype)
        #         )

        # ── Temporal processing ─────────────────────────────────────────────
        x = self.temporal_proj(frame_feats)    # [B, T, temp_dim]
        x = x + self.pos_embedding[:, :T, :]  # add positional encoding

        x = self.temporal_transformer(x)       # [B, T, temp_dim]

        # ── Learned temporal attention pooling ─────────────────────────────
        attn_weights = self.temporal_attn_pool(x)  # [B, T, 1]
        pooled = (x * attn_weights).sum(dim=1)     # [B, temp_dim]

        return self.output_proj(pooled)             # [B, swarm_feat_dim]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: CAUSAL INTENT REASONING MODULE (CTRM)
# ═════════════════════════════════════════════════════════════════════════════

class CausalIntentReasoningModule(nn.Module):
    """
    Models causal relationships in drone behavior over time.

    Key idea: Rather than correlating features, we model causal chains:
    "speed_decreased → formation_tightened → THEREFORE surveillance_intent"

    Uses a lightweight Structural Causal Model (SCM):
    - Causal discovery via attention masking (lower-triangular = past causes future)
    - Causal representation separates causes from effects
    - Combines with swarm features for final intent representation

    Input:  swarm_feat [B, swarm_feat_dim]
    Output: intent_repr [B, intent_repr_dim=768]
    """

    def __init__(self, cfg: CAPITAConfig):
        super().__init__()
        swarm_dim  = cfg.model.swarm_feat_dim    # 512
        causal_dim = cfg.model.causal_hidden_dim  # 256
        intent_dim = cfg.model.intent_repr_dim    # 768

        # ── Causal graph encoder ──────────────────────────────────────────
        # Projects swarm features into causal variable space
        self.causal_proj = nn.Sequential(
            nn.Linear(swarm_dim, causal_dim * 2),
            nn.ReLU(),
            nn.Linear(causal_dim * 2, causal_dim),
            nn.LayerNorm(causal_dim),
        )

        # ── Causal intervention module ────────────────────────────────────
        # Causal Transformer: causal attention mask enforces temporal ordering
        # (variable i can only be caused by variables j < i)
        self.n_causal_vars = 8  # number of latent causal variables
        self.causal_var_embed = nn.Linear(causal_dim // self.n_causal_vars + 1,
                                           causal_dim)

        causal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=causal_dim,
            nhead=4,
            dim_feedforward=causal_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.causal_transformer = nn.TransformerEncoder(
            causal_encoder_layer, num_layers=cfg.model.causal_num_layers
        )

        # ── Causal attention mask (lower-triangular) ──────────────────────
        causal_mask = torch.triu(
            torch.ones(self.n_causal_vars, self.n_causal_vars), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask)

        # ── Output projection ─────────────────────────────────────────────
        # Combines: swarm_feat [512] + causal_feat [256] → intent_repr [768]
        self.output_proj = nn.Sequential(
            nn.Linear(swarm_dim + causal_dim, intent_dim),
            nn.LayerNorm(intent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, swarm_feat: torch.Tensor) -> torch.Tensor:
        """
        swarm_feat: [B, swarm_feat_dim]
        Returns:    [B, intent_repr_dim]
        """
        B = swarm_feat.shape[0]

        # Project to causal space
        causal_repr = self.causal_proj(swarm_feat)  # [B, causal_dim]

        # Split into n_causal_vars variables
        # Each variable = causal_dim // n_causal_vars dims + 1 position embed
        var_size = self.n_causal_vars
        causal_split = causal_repr.view(B, var_size, -1)  # [B, n_vars, dim/n]

        # Add variable position index as feature
        pos_idx = torch.arange(var_size, device=causal_repr.device).to(dtype=causal_repr.dtype)
        pos_idx = pos_idx.view(1, var_size, 1).expand(B, -1, -1)  # [B, n_vars, 1]
        causal_input = torch.cat([causal_split, pos_idx], dim=-1)  # [B, n_vars, dim/n+1]

        # Project to causal_dim
        causal_input = self.causal_var_embed(causal_input)  # [B, n_vars, causal_dim]

        # Apply causal transformer with causal attention mask
        causal_out = self.causal_transformer(
            causal_input,
            mask=self.causal_mask,
        )  # [B, n_vars, causal_dim]

        # # ABLATION: remove causal mask → standard bidirectional transformer
        # causal_out = self.causal_transformer(
        #     causal_input,
        #     mask=None,                 # no causal constraint
        # )

        # Pool causal variables
        causal_feat = causal_out.mean(dim=1)  # [B, causal_dim]

        # Combine swarm + causal features
        intent_repr = self.output_proj(
            torch.cat([swarm_feat, causal_feat], dim=-1)
        )  # [B, intent_repr_dim]

        return intent_repr


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4: MULTI-TASK QA-AWARE OUTPUT HEADS
# ═════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """
    Lightweight classification head for closed-form QA types:
    - yes_no       (2 classes)
    - uav_size     (6 classes)
    - environment  (8 classes)
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

from transformers import StoppingCriteria, StoppingCriteriaList

class MultiTokenStoppingCriteria(StoppingCriteria):
    """Stops generation when any stop sequence appears in decoded output."""

    def __init__(self, stop_sequences: list, tokenizer):
        self.tokenizer     = tokenizer
        self.stop_sequences = stop_sequences
        # Pre-encode all stop sequences
        self.stop_ids = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in stop_sequences
        ]
        self.generated_ids = []  # track generated token ids

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        # input_ids contains only newly generated tokens when using inputs_embeds
        if input_ids.shape[1] == 0:
            return False

        # Check last N tokens against each stop sequence
        for stop_seq in self.stop_ids:
            seq_len = len(stop_seq)
            if input_ids.shape[1] >= seq_len:
                last_tokens = input_ids[0, -seq_len:].tolist()
                if last_tokens == stop_seq:
                    return True

        # Fallback: decode and check string
        try:
            decoded = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=True
            )
            for stop_str in self.stop_sequences:
                if stop_str in decoded:
                    return True
        except:
            pass

        return False



class TinyLlamaGenerativeHead(nn.Module):
    """
    TinyLlama-1.1B fine-tuned with LoRA adapters for:
    1. Caption generation  (video-level intent description)
    2. Motion description  (open-ended behavior QA)

    LoRA Strategy:
    - Base TinyLlama weights are fully FROZEN (no gradient)
    - LoRA injects trainable low-rank matrices into q/k/v/o attention projections
    - Only LoRA parameters + intent_proj + prefix_proj are updated during training
    - This gives ~1-3% trainable params vs full fine-tuning

    Intent injection:
    - Intent representation [B, 768] is projected into TinyLlama embedding space
    - Projected as N soft visual prefix tokens prepended to question tokens
    - This follows the LLaVA / Flamingo style of visual conditioning
    """

    def __init__(self, cfg: CAPITAConfig):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

        self.max_gen_length   = cfg.model.max_gen_length
        self.max_input_length = cfg.model.max_input_length
        llm_path              = cfg.model.tinyllama_path
        intent_dim            = cfg.model.intent_repr_dim    # 768
        llm_dim               = cfg.model.llm_input_proj_dim # 2048

        # ── Load TinyLlama with optional 8-bit quantization ───────────────
        print(f"[TinyLlama+LoRA] Loading from {llm_path}")

        if cfg.model.use_8bit:
            print("[TinyLlama+LoRA] Using 8-bit quantization (BitsAndBytes)")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            load_dtype = torch.float16
        else:
            quantization_config = None
            load_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=load_dtype,
            trust_remote_code=True,
        )

        # ── Prepare model for LoRA (required before applying LoRA) ────────
        # This handles gradient checkpointing setup for quantized models
        if cfg.model.use_8bit:
            self.llm = prepare_model_for_kbit_training(
                self.llm, use_gradient_checkpointing=True
            )

        # ── Apply LoRA adapters ────────────────────────────────────────────
        # Inject low-rank trainable matrices into attention projection layers
        # TinyLlama attention layers: q_proj, k_proj, v_proj, o_proj
        # Gate/up/down proj in FFN layers optionally included
        print(f"[TinyLlama+LoRA] Applying LoRA: r={cfg.model.lora_r}, "
              f"alpha={cfg.model.lora_alpha}, dropout={cfg.model.lora_dropout}")

        lora_config = LoraConfig(
            r=cfg.model.lora_r,                 # rank of decomposition (16)
            lora_alpha=cfg.model.lora_alpha,    # scaling factor (32)
            target_modules=[                     # which layers to inject LoRA into
                "q_proj",                        # query projection
                "k_proj",                        # key projection
                "v_proj",                        # value projection
                "o_proj",                        # output projection
                # Optionally include FFN layers for richer adaptation:
                # "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=cfg.model.lora_dropout,  # 0.05
            bias="none",                           # don't adapt bias terms
            task_type=TaskType.CAUSAL_LM,          # causal language modelling
        )

        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()  # prints "trainable params: X || all params: Y || trainable%: Z"

        # ── Intent projection: [intent_repr_dim] → [llm_dim] ─────────────
        # Projects CAPITA intent representation into TinyLlama's token space
        # This is the bridge between our visual pipeline and the LLM
        self.intent_proj = nn.Sequential(
            nn.Linear(intent_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # ── Soft visual prefix tokens ─────────────────────────────────────
        # Intent is injected as N learnable prefix tokens (like visual tokens)
        # Following: LLaVA, InstructBLIP visual conditioning approach
        self.n_prefix_tokens = cfg.model.n_prefix_tokens  # default: 4

        # Maps [intent_dim] → [n_prefix_tokens × llm_dim]
        self.prefix_proj = nn.Linear(
            intent_dim, self.n_prefix_tokens * llm_dim
        )

        # ── dtype reference ────────────────────────────────────────────────
        # Store for consistent casting during forward pass
        self._llm_dtype = load_dtype

    def _get_prefix_embeds(self, intent_repr: torch.Tensor) -> torch.Tensor:
        """
        Convert intent representation into soft prefix token embeddings.
        intent_repr: [B, intent_dim]
        Returns:     [B, n_prefix_tokens, llm_dim]
        """
        B = intent_repr.shape[0]
        # Project to prefix token space
        prefix_flat = self.prefix_proj(intent_repr.float())         # [B, n_prefix * llm_dim]
        prefix_embeds = prefix_flat.view(
            B, self.n_prefix_tokens, self.llm.config.hidden_size
        ).to(self._llm_dtype)                                       # [B, n_prefix, llm_dim]
        return prefix_embeds

    def forward(
        self,
        intent_repr: torch.Tensor,   # [B, intent_repr_dim]
        questions: list,             # list of question strings, len=B
        answers: list = None,        # list of answer strings (training only)
        generate: bool = False,
    ) -> Dict:
        """
        Training (generate=False):  compute cross-entropy loss on answer tokens
        Inference (generate=True):  autoregressively generate answer text
        """
        device        = intent_repr.device
        B             = intent_repr.shape[0]
        prefix_embeds = self._get_prefix_embeds(intent_repr)  # [B, n_prefix, llm_dim]

        if generate:
            return self._generate(prefix_embeds, questions)

        # ── Training: batch-level loss computation ─────────────────────────
        # Build batched inputs for efficiency (pad to same length)
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_valid    = 0

        for b in range(B):
            if not questions[b] or not answers[b]:
                continue

            # ── Tokenize: "Question: <q>\nAnswer: <a>" ────────────────────
            prompt = (f"You are analyzing UAV/drone video footage.\n"
                      f"Question: {questions[b]}\n"
                      f"Answer concisely about the drone behavior: {answers[b]}")
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_input_length + self.max_gen_length,
                truncation=True,
                padding=False,
            )

            input_ids      = enc["input_ids"].to(device)        # [1, seq_len]
            attention_mask = enc["attention_mask"].to(device)   # [1, seq_len]

            # ── Get token embeddings from (LoRA-wrapped) LLM ──────────────
            # Access base model embeddings via PEFT wrapper
            token_embeds = self.llm.model.model.embed_tokens(input_ids).to(self._llm_dtype)
            # [1, seq_len, llm_dim]

            # ── Prepend soft visual prefix (intent tokens) ─────────────────
            # Full sequence: [prefix_tokens | question_tokens | answer_tokens]
            full_embeds = torch.cat(
                [prefix_embeds[b:b+1], token_embeds], dim=1
            )  # [1, n_prefix + seq_len, llm_dim]

            # Extend attention mask to cover prefix tokens
            prefix_mask = torch.ones(
                1, self.n_prefix_tokens, dtype=torch.long, device=device
            )
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            # [1, n_prefix + seq_len]

            # ── Labels: mask out prefix tokens with -100 ──────────────────
            # Loss is only computed on actual answer tokens
            prefix_labels = torch.full(
                (1, self.n_prefix_tokens), -100,
                dtype=torch.long, device=device
            )
            labels = torch.cat([prefix_labels, input_ids], dim=1)
            # [1, n_prefix + seq_len]

            # ── Forward through LoRA-adapted TinyLlama ────────────────────
            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                labels=labels,
                return_dict=True,
            )

            if outputs.loss is not None:
                total_loss = total_loss + outputs.loss
                n_valid   += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return {"loss": total_loss}

    @torch.no_grad()
    def _generate(
        self,
        prefix_embeds: torch.Tensor,  # [B, n_prefix, llm_dim]
        questions: list,
    ) -> Dict:
        """Autoregressively generate answers using LoRA-adapted TinyLlama."""
        device    = prefix_embeds.device
        generated = []

        for b, question in enumerate(questions):
            if not question:
                generated.append("")
                continue

            # Tokenize question only (no answer — we're generating it)
            prompt = (f"You are analyzing UAV/drone video footage.\n"
                      f"Question: {question}\n"
                      f"Answer concisely about the drone behavior:")
            enc    = self.tokenizer(
                prompt, return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
            ).to(device)

            # Get token embeddings
            token_embeds = self.llm.model.model.embed_tokens(
                enc["input_ids"]
            ).to(self._llm_dtype)  # [1, seq_len, llm_dim]

            # Prepend intent prefix
            full_embeds = torch.cat(
                [prefix_embeds[b:b+1], token_embeds], dim=1
            )  # [1, n_prefix + seq_len, llm_dim]

            prefix_mask = torch.ones(
                1, self.n_prefix_tokens, dtype=torch.long, device=device
            )
            full_mask = torch.cat(
                [prefix_mask, enc["attention_mask"]], dim=1
            )


            # ── Stopping criteria ─────────────────────────────────────────────
            stopping_criteria = StoppingCriteriaList([
                MultiTokenStoppingCriteria(
                    stop_sequences=[
                        "\nQuestion:", "\n\nQuestion:", "\nQ:", "\nAnswer:",
                        "\nquestion:", "\nanswer:",
                    ],
                    tokenizer=self.tokenizer,
                )
            ])

            # ── Generate ──────────────────────────────────────────────────
            out = self.llm.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                max_new_tokens=self.max_gen_length,
                do_sample=False,               # greedy for deterministic eval
                temperature=1.0,
                repetition_penalty=1.1,        # reduce repetitive outputs
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

            # Decode newly generated tokens only
            # out shape: [1, total_tokens] — skip prefix + prompt tokens
            # n_prompt_tokens = full_embeds.shape[1]
            # new_tokens      = out[0][n_prompt_tokens:]
            new_tokens = out[0]
            text            = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            # ── Post-generation hard truncation ───────────────────────────────
            stop_patterns = [
                "\nQuestion:", "\n\nQuestion:", "\nQ:", "\nAnswer:",
                "\nquestion:", "\nq:", "\nanswer:", "Question:", "Q:"
            ]
            for stop in stop_patterns:
                if stop in text:
                    text = text[:text.index(stop)]

            # Limit to 3 sentences max for short-answer questions
            sentences = [s.strip() for s in text.strip().split('.') if s.strip()]
            if len(sentences) > 3:
                text = '. '.join(sentences[:3]) + '.'

            generated.append(text.strip())

        return {"generated_texts": generated}


# ═════════════════════════════════════════════════════════════════════════════
# FULL CAPITA MODEL
# ═════════════════════════════════════════════════════════════════════════════

class CAPITAModel(nn.Module):
    """
    Complete CAPITA pipeline.

    Forward pass returns a dict of losses (training) or predictions (inference).
    """

    def __init__(self, cfg: CAPITAConfig):
        super().__init__()
        self.cfg = cfg

        # ── Stage 1: Dual Stream Encoder ──────────────────────────────────
        self.dual_stream_encoder = DualStreamEncoder(cfg)

        # ── Stage 2: Swarm Graph Temporal Network ─────────────────────────
        self.swarm_gnn = SpatioTemporalGNN(cfg)

        # ── Stage 3: Causal Intent Reasoning ──────────────────────────────
        self.ctrm = CausalIntentReasoningModule(cfg)

        # ── Stage 4: Shared projection ────────────────────────────────────
        intent_dim      = cfg.model.intent_repr_dim       # 768
        shared_proj_dim = cfg.model.shared_proj_dim       # 512

        self.shared_proj = nn.Sequential(
            nn.Linear(intent_dim, shared_proj_dim),
            nn.LayerNorm(shared_proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ── Stage 4: Classification Heads ─────────────────────────────────
        self.yes_no_head = ClassificationHead(
            shared_proj_dim, cfg.model.yes_no_classes
        )
        self.size_head = ClassificationHead(
            shared_proj_dim, cfg.model.uav_size_classes
        )
        self.env_head = ClassificationHead(
            shared_proj_dim, cfg.model.environment_classes
        )

        # ── Stage 4: TinyLlama Generative Head ────────────────────────────
        self.llm_head = TinyLlamaGenerativeHead(cfg)

        # ── Loss functions ─────────────────────────────────────────────────
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch: Dict, generate: bool = False) -> Dict:
        """
        Training:   returns {"total_loss": ..., "loss_*": ...}
        Inference:  returns {"caption": [...], "motion_desc": [...],
                             "yes_no_pred": [...], "size_pred": [...],
                             "env_pred": [...]}
        """
        device = batch["frames"].device

        # Move tensors to device
        roi_patches = batch["roi_patches"].to(device)   # [B,T,M,3,rs,rs]
        boxes       = batch["boxes"].to(device)          # [B,T,M,5]
        drone_mask  = batch["drone_mask"].to(device)     # [B,T,M]

        # ── Stage 1 ────────────────────────────────────────────────────────
        frames = batch["frames"].to(device)
        drone_feats = self.dual_stream_encoder(frames, roi_patches, boxes, drone_mask)
        # [B, T, M, drone_feat_dim]

        # ── Stage 2 ────────────────────────────────────────────────────────
        swarm_feat = self.swarm_gnn(drone_feats, boxes, drone_mask)
        # [B, swarm_feat_dim]

        # ── Stage 3 ────────────────────────────────────────────────────────
        intent_repr = self.ctrm(swarm_feat)
        # [B, intent_repr_dim]

        # ── Stage 4: Shared projection ─────────────────────────────────────
        task_feat = self.shared_proj(intent_repr)  # [B, shared_proj_dim]

        if generate:
            return self._inference(batch, intent_repr, task_feat)

        return self._compute_losses(batch, intent_repr, task_feat, device)

    def _compute_losses(self, batch, intent_repr, task_feat, device):
        losses = {}

        # ── Single unified QA loss over ALL question-answer pairs ─────────────
        # Collect all (question, answer) pairs per sample regardless of type
        # This mirrors the reference pipeline: all QA is treated as caption-style generation
        all_questions = []
        all_answers   = []

        for b in range(intent_repr.shape[0]):
            # Always include the main caption question
            all_questions.append(batch["caption_question"][b])
            all_answers.append(batch["caption"][b])

            # Include all other QA pairs if they exist (non-empty questions)
            for q_key, a_key in [
                ("yes_no_question",  "yes_no_answer_text"),
                ("size_question",    "size_answer_text"),
                ("env_question",     "env_answer_text"),
                ("motion_question",  "motion_answer_text"),
            ]:
                if batch[q_key][b].strip():
                    all_questions.append(batch[q_key][b])
                    all_answers.append(batch[a_key][b])

        # Repeat intent_repr to match expanded QA count
        # Build index mapping: which intent_repr index for each QA pair
        repr_indices = []
        b_idx = 0
        for b in range(intent_repr.shape[0]):
            repr_indices.append(b_idx)  # caption
            for q_key in ["yes_no_question","size_question","env_question","motion_question"]:
                if batch[q_key][b].strip():
                    repr_indices.append(b_idx)
            b_idx += 1

        expanded_repr = intent_repr[repr_indices]  # [N_total, intent_repr_dim]

        qa_out = self.llm_head(
            intent_repr=expanded_repr,
            questions=all_questions,
            answers=all_answers,
            generate=False,
        )
        losses["loss_qa"]    = qa_out["loss"]
        losses["total_loss"] = qa_out["loss"]

        return losses

    @torch.no_grad()
    def _inference(
        self,
        batch: Dict,
        intent_repr: torch.Tensor,
        task_feat: torch.Tensor,
    ) -> Dict:
        """Generate all predictions for evaluation."""

        # Classification predictions
        yn_logits  = self.yes_no_head(task_feat)
        sz_logits  = self.size_head(task_feat)
        env_logits = self.env_head(task_feat)

        yn_preds  = yn_logits.argmax(dim=-1).cpu().tolist()
        sz_preds  = sz_logits.argmax(dim=-1).cpu().tolist()
        env_preds = env_logits.argmax(dim=-1).cpu().tolist()

        # Caption generation
        cap_out = self.llm_head(
            intent_repr,
            questions=batch["caption_question"],
            generate=True,
        )

        # Motion description generation
        mot_out = self.llm_head(
            intent_repr,
            questions=batch["motion_question"],
            generate=True,
        )

        return {
            "video_id":       batch["video_id"],
            "caption_pred":   cap_out["generated_texts"],
            "caption_gt":     batch["caption"],
            "motion_pred":    mot_out["generated_texts"],
            "motion_gt":      batch["motion_answer_text"],
            "yes_no_pred":    yn_preds,
            "yes_no_gt":      batch["yes_no_label"].tolist(),
            "size_pred":      sz_preds,
            "size_gt":        batch["size_label"].tolist(),
            "env_pred":       env_preds,
            "env_gt":         batch["env_label"].tolist(),
        }