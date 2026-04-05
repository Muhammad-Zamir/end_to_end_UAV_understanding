"""
CAPITA Dataset & DataLoader
Handles MultiUAV, Anti-UAV, and NPS datasets with:
- Adaptive keyframe sampling (16 frames per video)
- YOLO bounding box parsing
- QA type auto-classification
- Video-level caption alignment
"""

import os
import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random
import cv2

from config import CAPITAConfig


# ─────────────────────────────────────────────────────────────────────────────
# QA TYPE AUTO-CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_question_type(question: str) -> str:
    """
    Rule-based question type classifier.
    Works for datasets that don't have question_type labels.
    """
    q = question.lower().strip()

    # Yes/No: starts with auxiliary verbs
    if re.match(r'^(are|is|do|does|can|has|have|was|were|will|would|should)\b', q):
        return "yes_no"

    # Size / Count questions
    if any(w in q for w in ["size", "large", "small", "medium", "how big",
                              "how many", "count", "number of", "many uav",
                              "how much"]):
        return "uav_size"

    # Environment / Scene questions
    if any(w in q for w in ["environment", "background", "location", "where",
                              "scene", "weather", "terrain", "setting", "area",
                              "region", "sky", "urban", "forest"]):
        return "environment"

    # Motion / Behavior / Description → intent signal
    if any(w in q for w in ["describe", "how", "speed", "motion", "moving",
                              "formation", "altitude", "pattern", "maneuver",
                              "behavior", "behaviour", "flying", "what type",
                              "what kind", "trajectory", "direction", "path"]):
        return "motion_description"

    # Default: treat unknown as descriptive
    return "motion_description"


def classify_answer_yes_no(answer: str) -> int:
    """Convert Yes/No answer string to class index."""
    a = answer.lower().strip()
    if a.startswith("yes"):
        return 1
    elif a.startswith("no"):
        return 0
    return 0  # default No


SIZE_VOCAB = {
    "tiny": 0, "small": 1, "medium": 2, "large": 3,
    "multiple": 2, "unknown": 5,
    # count-based mapping
    "one": 1, "single": 1, "two": 1, "three": 1,
    "few": 1, "several": 1, "many": 2, "swarm": 4,
}

ENV_VOCAB = {
    "open sky": 0, "sky": 0, "urban": 1, "city": 1, "building": 1,
    "forest": 2, "tree": 2, "coastal": 3, "water": 3, "sea": 3,
    "indoor": 4, "night": 5, "dark": 5, "fog": 6, "unknown": 7,
}


def classify_answer_size(answer: str) -> int:
    a = answer.lower()
    for key, idx in SIZE_VOCAB.items():
        if key in a:
            return idx
    return 5  # unknown


def classify_answer_environment(answer: str) -> int:
    a = answer.lower()
    for key, idx in ENV_VOCAB.items():
        if key in a:
            return idx
    return 7  # unknown


# ─────────────────────────────────────────────────────────────────────────────
# YOLO BOUNDING BOX PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_yolo_label(label_path: str, max_drones: int = 35) -> np.ndarray:
    """
    Parse YOLO format label file.
    Returns: boxes [max_drones, 5] → (class, cx, cy, w, h) normalized [0,1]
             padded with zeros for missing drones
    """
    boxes = np.zeros((max_drones, 5), dtype=np.float32)
    mask  = np.zeros(max_drones, dtype=np.float32)  # 1=valid, 0=padding

    if not os.path.exists(label_path):
        return boxes, mask

    try:
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for i, line in enumerate(lines[:max_drones]):
            parts = line.split()
            if len(parts) >= 5:
                boxes[i] = [float(p) for p in parts[:5]]
                mask[i]  = 1.0
    except Exception:
        pass

    return boxes, mask


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_video_index(frames_dir: str, labels_dir: str) -> Dict[str, List[dict]]:
    """
    Scan frames directory and group files by video ID.

    Frame naming: MultiUAV-018_000001.jpg
    Label naming: MultiUAV-018_000001.txt

    Returns: {video_id: [{"frame_path":..., "label_path":..., "frame_idx":...}, ...]}
    """
    video_index = defaultdict(list)

    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Extract video_id and frame_idx
        # Pattern: <video_id>_<frame_idx>.<ext>
        match = re.match(r'^(.+?)_(\d+)\.(jpg|jpeg|png)$', fname, re.IGNORECASE)
        if not match:
            continue

        video_id  = match.group(1)   # e.g. "MultiUAV-018"
        frame_idx = int(match.group(2))
        stem      = match.group(1) + "_" + match.group(2)

        frame_path = os.path.join(frames_dir, fname)
        label_path = os.path.join(labels_dir, stem + ".txt")

        video_index[video_id].append({
            "frame_path": frame_path,
            "label_path": label_path,
            "frame_idx":  frame_idx,
            "fname":      fname,
        })

    # Sort each video's frames by frame index
    for vid in video_index:
        video_index[vid].sort(key=lambda x: x["frame_idx"])

    return dict(video_index)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE KEYFRAME SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_sample_frames(
    frame_list: List[dict],
    num_frames: int = 16,
    max_drones: int = 35,
    threshold: float = 0.15,
    is_training: bool = True,
) -> List[dict]:
    """
    Adaptive keyframe sampling:
    1. Parse all bounding boxes quickly
    2. Compute frame-to-frame bbox change score
    3. Sample frames biased toward high-change moments
    4. Always include first and last frame

    Falls back to uniform sampling if video is too short.
    """
    total = len(frame_list)

    if total <= num_frames:
        # Pad by repeating last frame
        padded = frame_list.copy()
        while len(padded) < num_frames:
            padded.append(frame_list[-1])
        return padded

    # ── Uniform fallback for very small datasets ─────────────────────────
    if total < num_frames * 2:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return [frame_list[i] for i in indices]

    # ── Compute change scores ─────────────────────────────────────────────
    # Sample a subset for scoring (every 3rd frame) to avoid full scan
    score_indices = list(range(0, total, max(1, total // (num_frames * 3))))
    change_scores = np.zeros(total, dtype=np.float32)

    prev_centers = None
    prev_count   = 0

    for idx in score_indices:
        boxes, mask = parse_yolo_label(frame_list[idx]["label_path"], max_drones)
        valid_boxes = boxes[mask > 0]
        count = len(valid_boxes)

        if prev_centers is not None and count > 0 and prev_count > 0:
            # Mean center displacement (normalized)
            curr_centers = valid_boxes[:, 1:3]  # cx, cy
            min_c = min(count, prev_count)
            displacement = np.mean(np.abs(
                curr_centers[:min_c] - prev_centers[:min_c]
            ))
            # Count change score
            count_change = abs(count - prev_count) / max(max_drones, 1)
            change_scores[idx] = displacement + count_change

        prev_centers = valid_boxes[:, 1:3] if count > 0 else prev_centers
        prev_count   = count

    # Interpolate scores to all frames
    if np.sum(change_scores) > 0:
        # Smooth scores
        from scipy.ndimage import gaussian_filter1d
        try:
            change_scores = gaussian_filter1d(change_scores, sigma=2)
        except Exception:
            pass

    # ── Build sampling probability ─────────────────────────────────────────
    # Base: uniform probability
    probs = np.ones(total, dtype=np.float32)
    # Boost high-change frames
    probs += change_scores * 5.0
    # Always include first and last
    probs[0]  += 100.0
    probs[-1] += 100.0

    probs = probs / probs.sum()

    # ── Sample without replacement ─────────────────────────────────────────
    if is_training:
        # Stochastic: sample with probability
        chosen = np.random.choice(total, size=num_frames, replace=False, p=probs)
        chosen = sorted(chosen.tolist())
    else:
        # Deterministic: take top-scoring + evenly spaced
        # Divide video into num_frames segments, pick best from each
        segment_size = total // num_frames
        chosen = []
        for seg in range(num_frames):
            start = seg * segment_size
            end   = min(start + segment_size, total)
            seg_scores = change_scores[start:end]
            best_local = start + np.argmax(seg_scores)
            chosen.append(int(best_local))
        chosen = sorted(set(chosen))
        # If duplicates removed, fill with uniform
        while len(chosen) < num_frames:
            extra = np.linspace(0, total - 1, num_frames - len(chosen), dtype=int)
            chosen = sorted(set(chosen + extra.tolist()))
        chosen = chosen[:num_frames]

    return [frame_list[i] for i in chosen]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CAPITADataset(Dataset):
    """
    Unified dataset for CAPITA pipeline.
    Supports MultiUAV, Anti-UAV, NPS datasets.
    """

    # Fixed question asked for caption generation
    CAPTION_QUESTION = "Describe the UAV behaviour in the video."

    def __init__(
        self,
        cfg: CAPITAConfig,
        split: str = "train",           # "train" | "test"
        tokenizer=None,
    ):
        super().__init__()
        self.cfg       = cfg
        self.split     = split
        self.tokenizer = tokenizer
        self.is_train  = (split == "train")
        self.img_size  = cfg.get_image_size()
        self.paths     = cfg.get_dataset_paths()
        self.max_drones = cfg.data.max_drones_per_frame

        # ── Select correct paths ──────────────────────────────────────────
        if split == "train":
            frames_dir = self.paths["train_frames"]
            labels_dir = self.paths["train_labels"]
            json_path  = self.paths["train_json"]
        else:
            frames_dir = self.paths["test_frames"]
            labels_dir = self.paths["test_labels"]
            json_path  = self.paths["test_json"]

        # ── Build video index ─────────────────────────────────────────────
        print(f"[Dataset] Building video index from {frames_dir}")
        self.video_index = build_video_index(frames_dir, labels_dir)
        self.video_ids   = sorted(self.video_index.keys())
        print(f"[Dataset] Found {len(self.video_ids)} videos in {split} split")

        # ── Load captions / QA ────────────────────────────────────────────
        self.annotations = self._load_annotations(json_path)

        # ── Filter to videos that have both frames and annotations ─────────
        annotated = set(self.annotations.keys())
        indexed   = set(self.video_ids)
        self.video_ids = sorted(annotated & indexed)
        print(f"[Dataset] {len(self.video_ids)} videos have both frames + annotations")

        # ── Image transforms ──────────────────────────────────────────────
        if self.is_train:
            self.frame_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])
        else:
            self.frame_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])

        self.roi_transform = transforms.Compose([
            transforms.Resize((cfg.data.roi_patch_size, cfg.data.roi_patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std =[0.229, 0.224, 0.225]),
        ])

    def _load_annotations(self, json_path: str) -> Dict:
        """Load and normalize annotation JSON."""
        if not os.path.exists(json_path):
            print(f"[Dataset] WARNING: JSON not found at {json_path}")
            return {}

        with open(json_path, 'r') as f:
            raw = json.load(f)

        annotations = {}
        for vid_id, data in raw.items():
            caption = data.get("caption", "")
            qa_pairs = []

            for qa in data.get("qa_pairs", []):
                q    = qa.get("question", "")
                a    = qa.get("answer", "")
                # Use provided type or auto-classify
                qtype = qa.get("question_type", None)
                if qtype is None or qtype == "":
                    qtype = classify_question_type(q)

                qa_pairs.append({
                    "question":      q,
                    "answer":        a,
                    "question_type": qtype,
                })

            annotations[vid_id] = {
                "caption":  caption,
                "qa_pairs": qa_pairs,
            }

        return annotations

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict:
        video_id = self.video_ids[idx]
        frame_list = self.video_index[video_id]
        ann        = self.annotations[video_id]

        # ── Sample 16 keyframes ───────────────────────────────────────────
        sampled = adaptive_sample_frames(
            frame_list,
            num_frames=self.cfg.data.num_frames,
            max_drones=self.max_drones,
            threshold=self.cfg.data.adaptive_threshold,
            is_training=self.is_train,
        )

        # ── Load frames, ROI patches, and boxes ──────────────────────────
        frames    = []   # [T, 3, H, W]
        roi_patches = [] # [T, M, 3, roi_size, roi_size]
        boxes_all = []   # [T, M, 5]
        masks_all = []   # [T, M]

        for frame_info in sampled:
            # Load full frame
            try:
                img = Image.open(frame_info["frame_path"]).convert("RGB")
                W, H = img.size
            except Exception:
                img = Image.new("RGB", (self.img_size, self.img_size))
                W, H = self.img_size, self.img_size

            frame_tensor = self.frame_transform(img)
            frames.append(frame_tensor)

            # Parse YOLO boxes
            boxes, mask = parse_yolo_label(frame_info["label_path"], self.max_drones)
            boxes_all.append(boxes)
            masks_all.append(mask)

            # Crop ROI patches for each drone
            rois = self._extract_roi_patches(img, boxes, mask, W, H)
            roi_patches.append(rois)

        # Stack tensors
        frames_tensor    = torch.stack(frames, dim=0)            # [T,3,H,W]
        boxes_tensor     = torch.from_numpy(np.stack(boxes_all)) # [T,M,5]
        masks_tensor     = torch.from_numpy(np.stack(masks_all)) # [T,M]
        roi_tensor       = torch.stack(roi_patches, dim=0)       # [T,M,3,rs,rs]

        # ── Prepare QA targets ────────────────────────────────────────────
        caption = ann["caption"]
        qa_targets = self._prepare_qa_targets(ann["qa_pairs"])

        # ── Build sample dict ─────────────────────────────────────────────
        sample = {
            "video_id":       video_id,
            "frames":         frames_tensor,          # [T,3,H,W]
            "roi_patches":    roi_tensor,              # [T,M,3,rs,rs]
            "boxes":          boxes_tensor,            # [T,M,5]
            "drone_mask":     masks_tensor,            # [T,M]
            "caption":        caption,
            "caption_question": self.CAPTION_QUESTION,

            # Per QA-type targets
            "yes_no_question":      qa_targets["yes_no"]["question"],
            "yes_no_label":         qa_targets["yes_no"]["label"],       # int
            "yes_no_answer_text":   qa_targets["yes_no"]["answer"],

            "size_question":        qa_targets["uav_size"]["question"],
            "size_label":           qa_targets["uav_size"]["label"],     # int
            "size_answer_text":     qa_targets["uav_size"]["answer"],

            "env_question":         qa_targets["environment"]["question"],
            "env_label":            qa_targets["environment"]["label"],  # int
            "env_answer_text":      qa_targets["environment"]["answer"],

            "motion_question":      qa_targets["motion_description"]["question"],
            "motion_answer_text":   qa_targets["motion_description"]["answer"],
        }

        return sample

    def _extract_roi_patches(
        self,
        img: Image.Image,
        boxes: np.ndarray,
        mask: np.ndarray,
        W: int,
        H: int,
    ) -> torch.Tensor:
        """
        Crop ROI patches from image for each drone.
        boxes: [M,5] (class,cx,cy,w,h) normalized
        Returns: [M, 3, roi_size, roi_size]
        """
        rs = self.cfg.data.roi_patch_size
        patches = []

        for i in range(self.max_drones):
            if mask[i] < 0.5:
                # Empty/padding drone → zero patch
                patches.append(torch.zeros(3, rs, rs))
                continue

            _, cx, cy, bw, bh = boxes[i]

            # Convert to pixel coordinates
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)

            # Add padding around small drones (5% of image size)
            pad = int(min(W, H) * 0.05)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(W, x2 + pad)
            y2 = min(H, y2 + pad)

            if x2 <= x1 or y2 <= y1:
                patches.append(torch.zeros(3, rs, rs))
                continue

            crop   = img.crop((x1, y1, x2, y2))
            tensor = self.roi_transform(crop)
            patches.append(tensor)

        return torch.stack(patches, dim=0)  # [M,3,rs,rs]

    def _prepare_qa_targets(self, qa_pairs: List[dict]) -> Dict:
        """
        From all QA pairs for a video, select one per type.
        Returns dict with question, answer, and numeric label per type.
        """
        # Group by type
        by_type = defaultdict(list)
        for qa in qa_pairs:
            by_type[qa["question_type"]].append(qa)

        def pick_one(qtype: str) -> dict:
            items = by_type.get(qtype, [])
            if not items:
                return {"question": "", "answer": "", "label": 0}
            # For motion, prefer longer/more descriptive answers
            if qtype == "motion_description":
                items = sorted(items, key=lambda x: len(x["answer"]), reverse=True)
            item = random.choice(items) if self.is_train else items[0]
            return item

        yn  = pick_one("yes_no")
        sz  = pick_one("uav_size")
        env = pick_one("environment")
        mo  = pick_one("motion_description")

        return {
            "yes_no": {
                "question": yn["question"],
                "answer":   yn["answer"],
                "label":    classify_answer_yes_no(yn["answer"]),
            },
            "uav_size": {
                "question": sz["question"],
                "answer":   sz["answer"],
                "label":    classify_answer_size(sz["answer"]),
            },
            "environment": {
                "question": env["question"],
                "answer":   env["answer"],
                "label":    classify_answer_environment(env["answer"]),
            },
            "motion_description": {
                "question": mo["question"],
                "answer":   mo["answer"],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def capita_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate: stack tensors, keep text as lists."""
    keys_tensor = ["frames", "roi_patches", "boxes", "drone_mask"]
    keys_int    = ["yes_no_label", "size_label", "env_label"]
    keys_text   = [
        "video_id", "caption", "caption_question",
        "yes_no_question", "yes_no_answer_text",
        "size_question", "size_answer_text",
        "env_question", "env_answer_text",
        "motion_question", "motion_answer_text",
    ]

    out = {}

    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    for k in keys_int:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)

    for k in keys_text:
        out[k] = [b[k] for b in batch]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    cfg: CAPITAConfig,
    tokenizer=None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""

    train_dataset = CAPITADataset(cfg, split="train", tokenizer=tokenizer)
    val_dataset   = CAPITADataset(cfg, split="test",  tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=capita_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=capita_collate_fn,
        drop_last=False,
    )

    print(f"[DataLoader] Train: {len(train_dataset)} videos → {len(train_loader)} batches")
    print(f"[DataLoader] Val:   {len(val_dataset)} videos → {len(val_loader)} batches")

    return train_loader, val_loader