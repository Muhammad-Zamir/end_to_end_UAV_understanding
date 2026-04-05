"""
Video-LLaVA-7B Fine-tuning for MM-AntiUAV UAV Behaviour Understanding
======================================================================
Matches the working code structure exactly:
- HuggingFace Trainer (no custom loop)
- No LoRA (full fine-tune with frozen vision tower)
- PIL images (not numpy)
- low_cpu_mem_usage=True
- No gradient checkpointing complications

Usage:
    # Train
    python video_llava_train.py --mode train --dataset MultiUAV

    # Evaluate
    python video_llava_train.py --mode eval \
        --checkpoint ./runs/video_llava/final \
        --dataset MultiUAV
"""

import os
import gc
import re
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    VideoLlavaForConditionalGeneration,
    VideoLlavaProcessor,
    Trainer,
    TrainingArguments,
)

# Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer as rouge_scorer_lib
import nltk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

class VideoLLaVAConfig:
    # Model
    model_name   = "LanguageBind/Video-LLaVA-7B-hf"

    # Data paths — update these to match your server
    data_root    = "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/MM-AntiUAV"
    train_qa     = "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/MM-AntiUAV/VQA/TD_UAV_new_QA_train.json"
    test_qa      = "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/MM-AntiUAV/VQA/TD_UAV_new_QA_test.json"

    # Frames directory — same structure as CAPITA dataset
    # Expected: <frames_root>/<video_id>/<frame>.jpg
    frames_root  = "/media/zamir/267a161c-11fb-45e4-86b4-b11cba0972ac/MM-AntiUAV-Yolo/images"

    # Output
    output_dir   = "runs/models/video_llava_finetuned"

    # Training — matches CAPITA for fair comparison
    num_epochs   = 3        # Video-LLaVA converges faster than CAPITA
    batch_size   = 1        # per-device batch size (effective batch = batch_size * grad_accum)
    grad_accum   = 16       # effective batch = 16, same as CAPITA
    lr           = 2e-5
    warmup_steps = 200
    save_steps   = 500
    log_steps    = 10

    # LoRA
    lora_r          = 16
    lora_alpha      = 32
    lora_dropout    = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Quantization
    use_4bit        = True

    # Data
    num_frames      = 8
    max_new_tokens  = 128
    dataset_name    = "MultiUAV"
    device          = "cuda" if torch.cuda.is_available() else "cpu"


cfg = VideoLLaVAConfig()


# ═════════════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════════════

class UAVVQADataset(Dataset):
    """
    MM-AntiUAV dataset for Video-LLaVA.
    Mirrors working code structure exactly.
    Each sample = one (video_frames, question, answer) triple.
    """

    def __init__(
        self,
        qa_path: str,
        frames_root: str,
        processor,
        num_frames: int = 8,
        split: str = "train",
        dataset_name: str = "MultiUAV",
    ):
        self.frames_root  = Path(frames_root)
        self.processor    = processor
        self.num_frames   = num_frames
        self.split        = split
        self.dataset_name = dataset_name

        with open(qa_path, 'r') as f:
            self.qa_data = json.load(f)

        self.samples = self._build_samples()
        logger.info(f"[{split}] {len(self.samples)} samples from "
                    f"{len(self.qa_data)} videos")

    def _build_samples(self) -> List[Dict]:
        samples = []
        for video_id, video_data in self.qa_data.items():
            # Filter by dataset
            if self.dataset_name not in video_id:
                continue

            # Main caption question — always included
            caption = video_data.get("caption", "")
            if caption:
                samples.append({
                    "video_id": video_id,
                    "question": "Describe the UAV behaviour in the video.",
                    "answer":   caption,
                    "task_type": "caption",
                })

            # All QA pairs
            for qa in video_data.get("qa_pairs", []):
                q = qa.get("question", "").strip()
                a = qa.get("answer",   "").strip()
                if q and a:
                    samples.append({
                        "video_id":  video_id,
                        "question":  q,
                        "answer":    a,
                        "task_type": qa.get("question_type", "general"),
                    })

        if self.split == "train":
            random.shuffle(samples)
        return samples

    def _load_frames(self, video_id: str) -> Optional[List[Image.Image]]:
        """
        Load T uniformly sampled frames as PIL images.
        Searches train/ and val/ subdirectories.
        """
        for subdir in ["train", "val", "test", ""]:
            if subdir:
                video_dir = self.frames_root / subdir / video_id
            else:
                video_dir = self.frames_root / video_id

            if video_dir.exists():
                break

            # Try glob pattern (frames named video_id_XXXXXX.jpg)
            parent = self.frames_root / subdir if subdir else self.frames_root
            matches = sorted(parent.glob(f"{video_id}_*.jpg"))
            if not matches:
                matches = sorted(parent.glob(f"{video_id}_*.png"))
            if matches:
                # Load directly from flat file list
                indices = np.linspace(0, len(matches)-1,
                                      self.num_frames, dtype=int)
                frames = []
                for idx in indices:
                    try:
                        img = Image.open(str(matches[idx])).convert("RGB")
                    except:
                        img = Image.new("RGB", (224, 224), (0, 0, 0))
                    frames.append(img)
                return frames
        else:
            return None

        # Load from directory
        frame_files = sorted(
            list(video_dir.glob("*.jpg")) +
            list(video_dir.glob("*.png"))
        )
        if not frame_files:
            return None

        indices = np.linspace(0, len(frame_files)-1,
                              self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            try:
                img = Image.open(str(frame_files[idx])).convert("RGB")
            except:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            frames.append(img)
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        video_id = sample["video_id"]
        question = sample["question"]
        answer   = sample["answer"]

        frames = self._load_frames(video_id)
        if frames is None:
            # Dummy frames — same as working code
            frames = [Image.new("RGB", (224, 224), (0, 0, 0))] * self.num_frames

        # Prompt format — identical to working code
        prompt = (
            f"USER: <video>\n"
            f"You are a UAV video analyst. "
            f"Question: {question}\n"
            f"ASSISTANT: {answer}"
        )

        # Process — no truncation, no padding (same as working code)
        inputs = self.processor(
            text=prompt,
            videos=frames,
            return_tensors="pt",
        )

        # Labels — mask padding with -100 (same as working code)
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "video_id":       video_id,
            "task_type":      sample["task_type"],
            "question":       question,
            "reference":      answer,
            "input_ids":      inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values_videos": inputs.get(
                "pixel_values_videos",
                inputs.get("pixel_values")
            ).squeeze(0),
            "labels":         labels.squeeze(0),
        }


# ═════════════════════════════════════════════════════════════════════════════
# COLLATOR  — identical to working code
# ═════════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[Dict]) -> Dict:
    """Filter None, pad sequences, stack tensors."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Pad input_ids and attention_mask to same length
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids_padded      = []
    attention_mask_padded = []
    labels_padded         = []

    pad_id = 0  # processor uses 0 as pad

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_padded.append(
            torch.cat([b["input_ids"],
                       torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attention_mask_padded.append(
            torch.cat([b["attention_mask"],
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_padded.append(
            torch.cat([b["labels"],
                       torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids":           torch.stack(input_ids_padded),
        "attention_mask":      torch.stack(attention_mask_padded),
        "pixel_values_videos": torch.stack([b["pixel_values_videos"] for b in batch]),
        "labels":              torch.stack(labels_padded),
    }


# ═════════════════════════════════════════════════════════════════════════════
# METRICS  — identical to CAPITA val.py
# ═════════════════════════════════════════════════════════════════════════════

class TextMetrics:

    def __init__(self):
        self.smoothing = SmoothingFunction().method4
        self.rouge     = rouge_scorer_lib.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

    def compute_bleu(self, reference: str, candidate: str) -> dict:
        ref_t  = reference.lower().split()
        cand_t = candidate.lower().split()
        if not cand_t:
            return {f'BLEU-{n}': 0.0 for n in range(1, 5)}
        max_n = min(4, len(cand_t))
        w_sets = [
            (1,0,0,0),(0.5,0.5,0,0),(0.33,0.33,0.33,0),(0.25,0.25,0.25,0.25)
        ]
        scores = {}
        for n, w in enumerate(w_sets, 1):
            if max_n >= n:
                try:
                    scores[f'BLEU-{n}'] = sentence_bleu(
                        [ref_t], cand_t, weights=w,
                        smoothing_function=self.smoothing)
                except:
                    scores[f'BLEU-{n}'] = 0.0
            else:
                scores[f'BLEU-{n}'] = 0.0
        return scores

    def compute_rouge(self, reference: str, candidate: str) -> dict:
        if not candidate:
            return {'ROUGE-1': 0, 'ROUGE-2': 0, 'ROUGE-L': 0}
        s = self.rouge.score(reference, candidate)
        return {'ROUGE-1': s['rouge1'].fmeasure,
                'ROUGE-2': s['rouge2'].fmeasure,
                'ROUGE-L': s['rougeL'].fmeasure}

    def compute_meteor(self, reference: str, candidate: str) -> float:
        if not candidate.strip():
            return 0.0
        try:
            return nltk_meteor(
                [reference.lower().split()], candidate.lower().split())
        except:
            return 0.0

    def compute_spice_simple(self, reference: str, candidate: str) -> float:
        from nltk.stem import PorterStemmer
        stemmer   = PorterStemmer()
        stopwords = {
            'a','an','the','is','are','was','were','be','been','have','has',
            'had','do','does','did','to','of','in','for','on','with','at',
            'by','from','and','or','but','this','that','it','its',
            'they','them','their'
        }
        def stem(text):
            return set(stemmer.stem(w) for w in text.lower().split()
                       if w.strip('.,!?') not in stopwords and len(w) > 1)
        rw = stem(reference)
        cw = stem(candidate)
        if not rw or not cw:
            return 0.0
        inter = len(rw & cw)
        p = inter / len(cw)
        r = inter / len(rw)
        return 2*p*r/(p+r) if p+r > 0 else 0.0

    def compute_acc(self, reference: str, candidate: str) -> float:
        from nltk.stem import PorterStemmer
        stemmer  = PorterStemmer()
        ref_l    = reference.lower().strip().rstrip('.,!?')
        cand_l   = candidate.lower().strip()
        if ref_l in ['yes', 'no']:
            return 1.0 if ref_l in cand_l else 0.0
        stopwords = {
            'a','an','the','is','are','was','were','be','have','has','had',
            'do','does','did','to','of','in','for','and','or','but',
            'it','this','that','they','them'
        }
        def key_words(text):
            return set(stemmer.stem(w)
                       for w in re.findall(r'\b\w+\b', text.lower())
                       if w not in stopwords and len(w) > 1)
        rs = key_words(reference)
        cs = key_words(candidate)
        if not rs:
            return 1.0
        if not cs:
            return 0.0
        recall = len(rs & cs) / len(rs)
        return min(1.0, recall * 1.3) if recall > 0.3 else recall

    def compute_all(self, reference: str, candidate: str) -> dict:
        ref_s = reference.lower().strip().rstrip('.,!?')
        if ref_s in ['yes', 'no'] or len(ref_s.split()) == 1:
            eff_cand = candidate.lower().strip().split()[0].rstrip('.,!?') \
                       if candidate.strip() else ''
            eff_ref  = ref_s
        else:
            eff_cand = candidate
            eff_ref  = reference
        m = {}
        m.update(self.compute_bleu(eff_ref, eff_cand))
        m.update(self.compute_rouge(eff_ref, eff_cand))
        m['METEOR'] = self.compute_meteor(eff_ref, eff_cand)
        m['SPICE']  = self.compute_spice_simple(eff_ref, eff_cand)
        m['ACC']    = self.compute_acc(reference, candidate)
        return m


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train(args):
    logger.info("=" * 60)
    logger.info("Video-LLaVA-7B Fine-tuning — MM-AntiUAV")
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info("=" * 60)

    # ── Load processor and model — identical to working code ──────────────
    logger.info("Loading processor and model...")
    processor = VideoLlavaProcessor.from_pretrained(cfg.model_name)
    



    # 4-bit quantization — same approach as working LLaMA-3 CoT pipeline
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        cfg.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Prepare for LoRA — identical to working CoT pipeline
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # ── Datasets ───────────────────────────────────────────────────────────
    logger.info("Building datasets...")
    train_dataset = UAVVQADataset(
        qa_path=cfg.train_qa,
        frames_root=cfg.frames_root,
        processor=processor,
        num_frames=cfg.num_frames,
        split="train",
        dataset_name=cfg.dataset_name,
    )
    val_dataset = UAVVQADataset(
        qa_path=cfg.test_qa,
        frames_root=cfg.frames_root,
        processor=processor,
        num_frames=cfg.num_frames,
        split="test",
        dataset_name=cfg.dataset_name,
    )

    # ── Training arguments — matches working code ──────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=cfg.log_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,    # avoids extra GPU memory allocation
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    logger.info("Starting training...")
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    final_path = f"{cfg.output_dir}/final"
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Model saved to: {final_path}")


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(args):
    logger.info("=" * 60)
    logger.info(f"Video-LLaVA Evaluation | Dataset: {cfg.dataset_name}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("=" * 60)

    # ── Load model ─────────────────────────────────────────────────────────
    processor = VideoLlavaProcessor.from_pretrained(args.checkpoint)

    # Load base model first with 4-bit quantization
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = VideoLlavaForConditionalGeneration.from_pretrained(
        cfg.model_name,                  # load from original HuggingFace model
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter weights on top of base model
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint,               # load adapter from checkpoint dir
        is_trainable=False,
    )
    model.eval()
    logger.info(f"Loaded base model + LoRA adapter from {args.checkpoint}")

    # ── Dataset ────────────────────────────────────────────────────────────
    test_dataset = UAVVQADataset(
        qa_path=cfg.test_qa,
        frames_root=cfg.frames_root,
        processor=processor,
        num_frames=cfg.num_frames,
        split="test",
        dataset_name=cfg.dataset_name,
    )

    metrics_calc  = TextMetrics()
    all_preds     = []
    task_metrics  = defaultdict(lambda: defaultdict(list))
    metric_names  = [
        'BLEU-1','BLEU-2','BLEU-3','BLEU-4',
        'ROUGE-1','ROUGE-2','ROUGE-L',
        'METEOR','SPICE','ACC',
    ]

    from tqdm import tqdm
    for sample in tqdm(test_dataset, desc="Evaluating"):
        if sample is None:
            continue

        video_id  = sample["video_id"]
        question  = sample["question"]
        reference = sample["reference"]
        task_type = sample["task_type"]

        # Reload frames for generation (no answer in prompt)
        frames = test_dataset._load_frames(video_id)
        if frames is None:
            frames = [Image.new("RGB", (224, 224), (0,0,0))] * cfg.num_frames

        prompt = (
            f"USER: <video>\n"
            f"You are a UAV video analyst. "
            f"Question: {question}\n"
            f"ASSISTANT:"
        )

        inputs = processor(
            text=prompt,
            videos=frames,
            return_tensors="pt",
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        # Decode new tokens only
        input_len  = inputs["input_ids"].shape[1]
        new_tokens = out[0][input_len:]
        pred_text  = processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        # Truncate at follow-up patterns
        for stop in ["\nQuestion:", "\nUSER:", "\nASSISTANT:", "\nQ:"]:
            if stop in pred_text:
                pred_text = pred_text[:pred_text.index(stop)]
        pred_text = pred_text.strip()

        # Compute metrics
        m = metrics_calc.compute_all(reference, pred_text)
        for k, v in m.items():
            task_metrics[task_type][k].append(v)
            task_metrics["all"][k].append(v)

        all_preds.append({
            "video_id":         video_id,
            "task_type":        task_type,
            "question":         question,
            "generated_answer": pred_text,
            "reference":        reference,
            **m,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    results_summary = {}
    for metric in metric_names:
        vals = task_metrics["all"][metric]
        results_summary[metric] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std":  float(np.std(vals))  if vals else 0.0,
        }

    # ── Save ───────────────────────────────────────────────────────────────
    output_dir = Path(args.eval_output) / cfg.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(all_preds, f, indent=2)

    table_results = {
        "Method":  "Video-LLaVA-7B (fine-tuned)",
        "Dataset": cfg.dataset_name,
        **{m: round(results_summary[m]['mean'] * 100, 2) for m in metric_names},
    }
    with open(output_dir / "table_results.json", 'w') as f:
        json.dump(table_results, f, indent=2)

    # ── Print ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Video-LLaVA-7B  |  Dataset: {cfg.dataset_name}")
    print("="*60)
    for m in metric_names:
        print(f"  {m:<12} {results_summary[m]['mean']*100:>8.2f}")
    print("="*60)
    print(f"\nPredictions → {output_dir / 'predictions.json'}")
    print(f"Table       → {output_dir / 'table_results.json'}")

    return table_results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       required=True,
                        choices=["train", "eval", "train_eval"])
    parser.add_argument("--dataset",    default="MultiUAV",
                        choices=["MultiUAV", "Anti-UAV", "NPS"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to saved model dir for evaluation")
    parser.add_argument("--eval_output",default="./eval_results/video_llava")
    args = parser.parse_args()

    cfg.dataset_name = args.dataset

    if args.mode in ("train", "train_eval"):
        train(args)
        if args.mode == "train_eval":
            args.checkpoint = f"{cfg.output_dir}/final"

    if args.mode in ("eval", "train_eval"):
        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval")
        evaluate(args)


if __name__ == "__main__":
    main()