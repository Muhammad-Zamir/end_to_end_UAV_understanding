"""
CAPITA Evaluation Script
Computes: BLEU-1/2/3/4, ROUGE-1/2/L, METEOR, SPICE, ACC
For all 4 QA heads: caption, motion_description, yes_no, uav_size, environment
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List

import torch
from torch.cuda.amp import autocast

# ── Metrics imports (from your existing script) ───────────────────────────────
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer
import nltk
import re

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

SPICE_AVAILABLE = False
try:
    from pycocoevalcap.spice.spice import Spice
    SPICE_AVAILABLE = True
    print("✓ SPICE available")
except ImportError:
    print("⚠ SPICE not available — using fallback SPICE approximation")

# ── CAPITA imports ─────────────────────────────────────────────────────────────
from config import CAPITAConfig
from dataset import CAPITADataset, capita_collate_fn
from model import CAPITAModel

from torch.utils.data import DataLoader


# ═════════════════════════════════════════════════════════════════════════════
# PASTE YOUR TextMetrics CLASS HERE (unchanged from your script)
# ═════════════════════════════════════════════════════════════════════════════

class TextMetrics:
    """Text generation metrics calculator — from your existing evaluation script."""

    def __init__(self):
        self.smoothing = SmoothingFunction().method4
        self.rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.spice_scorer = None
        if SPICE_AVAILABLE:
            try:
                self.spice_scorer = Spice()
                print("✓ SPICE scorer initialized")
            except Exception as e:
                print(f"⚠ SPICE initialization failed: {e}")

    def compute_bleu(self, reference: str, candidate: str) -> dict:
        ref_tokens  = reference.lower().split()
        cand_tokens = candidate.lower().split()
        if len(cand_tokens) == 0:
            return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0}
        references = [ref_tokens]
        max_n = min(4, len(cand_tokens))
        scores = {}
        weight_sets = [
            (1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)
        ]
        for n, w in enumerate(weight_sets, 1):
            if max_n >= n:
                try:
                    scores[f'BLEU-{n}'] = sentence_bleu(
                        references, cand_tokens,
                        weights=w, smoothing_function=self.smoothing
                    )
                except:
                    scores[f'BLEU-{n}'] = 0.0
            else:
                scores[f'BLEU-{n}'] = 0.0
        return scores

    def compute_rouge(self, reference: str, candidate: str) -> dict:
        if len(candidate) == 0:
            return {'ROUGE-1': 0, 'ROUGE-2': 0, 'ROUGE-L': 0}
        s = self.rouge.score(reference, candidate)
        return {
            'ROUGE-1': s['rouge1'].fmeasure,
            'ROUGE-2': s['rouge2'].fmeasure,
            'ROUGE-L': s['rougeL'].fmeasure,
        }

    def compute_meteor(self, reference: str, candidate: str) -> float:
        if len(candidate.strip()) == 0:
            return 0.0
        try:
            return nltk_meteor([reference.lower().split()], candidate.lower().split())
        except:
            return 0.0

    def compute_spice_simple(self, reference: str, candidate: str) -> float:
        from nltk.stem import PorterStemmer
        stemmer   = PorterStemmer()
        stopwords = {
            'a','an','the','is','are','was','were','be','been','have','has',
            'had','do','does','did','to','of','in','for','on','with','at',
            'by','from','and','or','but','this','that','it','its','they','them','their'
        }
        def stem_words(text):
            return [stemmer.stem(w) for w in text.lower().split()
                    if w.strip('.,!?') not in stopwords and len(w) > 1]
        ref_words  = set(stem_words(reference))
        cand_words = set(stem_words(candidate))
        if not ref_words or not cand_words:
            return 0.0
        inter = len(ref_words & cand_words)
        p = inter / len(cand_words)
        r = inter / len(ref_words)
        return 2*p*r/(p+r) if p+r > 0 else 0.0

    def compute_spice(self, reference: str, candidate: str) -> float:
        if self.spice_scorer is None:
            return self.compute_spice_simple(reference, candidate)
        try:
            gts = {"0": [reference]}
            res = {"0": [candidate if candidate.strip() else "empty"]}
            avg_score, _ = self.spice_scorer.compute_score(gts, res)
            if isinstance(avg_score, dict):
                avg_score = avg_score.get('All', {}).get('f', 0.0)
            return float(avg_score)
        except:
            return self.compute_spice_simple(reference, candidate)

    def compute_acc(self, reference: str, candidate: str, question: str = "") -> float:
        from nltk.stem import PorterStemmer
        stemmer   = PorterStemmer()
        ref_lower  = reference.lower().strip()
        cand_lower = candidate.lower().strip()
        if ref_lower in ['yes', 'no', 'yes.', 'no.']:
            return 1.0 if (
                ('yes' in cand_lower and 'yes' in ref_lower) or
                ('no'  in cand_lower and 'no'  in ref_lower)
            ) else 0.0
        ref_nums  = re.findall(r'\d+', reference)
        cand_nums = re.findall(r'\d+', candidate)
        if ref_nums and all(n in cand_nums for n in ref_nums):
            return 0.9
        stopwords = {
            'a','an','the','is','are','was','were','be','have','has','had',
            'do','does','did','to','of','in','for','and','or','but','it','this','that','they','them'
        }
        def key_words(text):
            return set(stemmer.stem(w) for w in re.findall(r'\b\w+\b', text.lower())
                       if w not in stopwords and len(w) > 1)
        ref_set  = key_words(reference)
        cand_set = key_words(candidate)
        if not ref_set:
            return 1.0
        if not cand_set:
            return 0.0
        recall = len(ref_set & cand_set) / len(ref_set)
        return min(1.0, recall * 1.3) if recall > 0.3 else recall

    # def compute_all(self, reference: str, candidate: str, question: str = "") -> dict:
    #     m = {}
    #     m.update(self.compute_bleu(reference, candidate))
    #     m.update(self.compute_rouge(reference, candidate))
    #     m['METEOR'] = self.compute_meteor(reference, candidate)
    #     m['SPICE']  = self.compute_spice(reference, candidate)
    #     m['ACC']    = self.compute_acc(reference, candidate, question)
    #     return m

    def compute_all(self, reference: str, candidate: str, question: str = "") -> dict:
        m = {}

        ref_stripped = reference.lower().strip().rstrip('.,!?')

        # ── Short reference: Yes/No or single meaningful word ─────────────
        if ref_stripped in ['yes', 'no'] or len(ref_stripped.split()) == 1:
            cand_first = candidate.lower().strip().split()[0].rstrip('.,!?') \
                        if candidate.strip() else ''
            # Normalise reference too — strip punctuation for fair comparison
            ref_normalised     = ref_stripped          # already stripped above
            effective_candidate = cand_first
            effective_reference = ref_normalised       # use normalised, not original
        else:
            effective_candidate = candidate
            effective_reference = reference            # use original for long answers

        m.update(self.compute_bleu(effective_reference, effective_candidate))
        m.update(self.compute_rouge(effective_reference, effective_candidate))
        m['METEOR'] = self.compute_meteor(effective_reference, effective_candidate)
        m['SPICE']  = self.compute_spice(effective_reference, effective_candidate)
        m['ACC']    = self.compute_acc(reference, candidate, question)
        return m


# ═════════════════════════════════════════════════════════════════════════════
# LABEL INDEX → TEXT MAPS (mirrors dataset.py vocab)
# ═════════════════════════════════════════════════════════════════════════════

YES_NO_MAP   = {0: "No", 1: "Yes"}

SIZE_MAP     = {
    0: "tiny",    1: "small",   2: "medium",
    3: "large",   4: "multiple", 5: "unknown"
}

ENV_MAP      = {
    0: "open sky", 1: "urban",   2: "forest",
    3: "coastal",  4: "indoor",  5: "night",
    6: "fog",      7: "unknown"
}


def idx_to_text(idx: int, map_dict: dict) -> str:
    return map_dict.get(int(idx), "unknown")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model: CAPITAModel,
    loader: DataLoader,
    cfg: CAPITAConfig,
    metrics_calc: TextMetrics,
    device: torch.device,
) -> (List[Dict], Dict):
    """
    Run full evaluation over the test set.
    Returns:
        all_predictions : list of per-sample prediction dicts
        results_summary : dict of {metric_name: {mean, std}}
    """
    model.eval()

    all_predictions = []

    # Per-task metric accumulators
    # Tasks: caption, motion, yes_no, size, environment
    task_metrics = defaultdict(lambda: defaultdict(list))

    metric_names = [
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
        'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
        'METEOR', 'SPICE', 'ACC'
    ]

    for batch in tqdm(loader, desc="Evaluating"):

        # ── Move batch to GPU ─────────────────────────────────────────────
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # ── Run model in generation mode ──────────────────────────────────
        with autocast(dtype=torch.float16, enabled=cfg.training.use_amp):
            preds = model(batch, generate=True)
            drone_feats = model.dual_stream_encoder(
            batch["frames"], batch["roi_patches"], batch["boxes"], batch["drone_mask"])
            swarm_feat  = model.swarm_gnn(drone_feats, batch["boxes"], batch["drone_mask"])
            intent_repr = model.ctrm(swarm_feat)

        B = len(preds["video_id"])

        for b in range(B):
            vid_id = preds["video_id"][b]

            # ── Evaluate caption (main question) ──────────────────────────
            cap_pred = preds["caption_pred"][b]
            cap_gt   = preds["caption_gt"][b]
            cap_m    = metrics_calc.compute_all(cap_gt, cap_pred,
                           question=batch["caption_question"][b])
            for k, v in cap_m.items():
                task_metrics["all_qa"][k].append(v)
                task_metrics["caption"][k].append(v) 
            all_predictions.append({
                "video_id": vid_id, "task_type": "caption",
                "question": batch["caption_question"][b],
                "generated_answer": cap_pred, "reference": cap_gt, **cap_m,
            })

            # ── Evaluate all other QA pairs as generation tasks ────────────
            for q_key, a_key, task_name in [
                ("yes_no_question",  "yes_no_answer_text",  "yes_no"),
                ("size_question",    "size_answer_text",    "uav_size"),
                ("env_question",     "env_answer_text",     "environment"),
                ("motion_question",  "motion_answer_text",  "motion_description"),
            ]:
                q = batch[q_key][b]
                a = batch[a_key][b]
                if not q.strip():
                    continue
                # Generate answer using caption head (single generative model)
                qa_out = model.llm_head(
                    intent_repr[b:b+1],
                    questions=[q],
                    generate=True,
                )
                pred_text = qa_out["generated_texts"][0]
                m = metrics_calc.compute_all(a, pred_text, question=q)
                for k, v in m.items():
                    task_metrics["all_qa"][k].append(v)
                    task_metrics[task_name][k].append(v) 
                all_predictions.append({
                    "video_id": vid_id, "task_type": task_name,
                    "question": q, "generated_answer": pred_text,
                    "reference": a, **m,
                })

    # ── Compute summary statistics ─────────────────────────────────────────
    # Overall (all tasks combined)
    all_metric_values = defaultdict(list)
    for task_name, task_m in task_metrics.items():
        for metric_name, vals in task_m.items():
            all_metric_values[metric_name].extend(vals)

    results_summary = {}
    for metric in metric_names:
        vals = all_metric_values[metric]
        results_summary[metric] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std":  float(np.std(vals))  if vals else 0.0,
        }

    # Per-task summary
    per_task_summary = {}
    for task_name, task_m in task_metrics.items():
        per_task_summary[task_name] = {}
        for metric in metric_names:
            vals = task_m[metric]
            per_task_summary[task_name][metric] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std":  float(np.std(vals))  if vals else 0.0,
            }

    return all_predictions, results_summary, per_task_summary


# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS  (your existing save block, adapted for CAPITA)
# ═════════════════════════════════════════════════════════════════════════════

def save_results(
    all_predictions: List[Dict],
    results_summary: Dict,
    per_task_summary: Dict,
    output_dir: Path,
    dataset_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        'BLEU-1','BLEU-2','BLEU-3','BLEU-4',
        'ROUGE-1','ROUGE-2','ROUGE-L',
        'METEOR','SPICE','ACC'
    ]

    # ── Save all predictions ───────────────────────────────────────────────
    output_path = output_dir / 'predictions.json'
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\n✓ Predictions saved to: {output_path}")

    # ── Save metrics summary ───────────────────────────────────────────────
    metrics_summary = {
        'model':   'CAPITA — Anti-UAV Intent Understanding',
        'dataset': dataset_name,
        'samples': len(all_predictions),
        'overall_metrics': {
            metric: {
                'mean': float(results_summary[metric]['mean']),
                'std':  float(results_summary[metric]['std']),
            }
            for metric in metric_names
        },
        'per_task_metrics': per_task_summary,
    }

    metrics_path = output_dir / 'metrics_summary.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✓ Metrics summary saved to: {metrics_path}")

    task_acc = {}
    for task_key, task_label in [
        ("all_qa",             "ACC_Overall"),
        ("caption",            "ACC_Caption"),
        ("yes_no",             "ACC_YesNo"),
        ("motion_description", "ACC_Motion"),
        ("uav_size",           "ACC_Size"),
        ("environment",        "ACC_Environment"),
    ]:
        task_acc[task_label] = round(
            per_task_summary.get(task_key, {}).get("ACC", {}).get("mean", 0) * 100, 2
        )

    # ── Save paper-ready table results ────────────────────────────────────
    table_results = {
        'Method':  'CAPITA',
        'Dataset': dataset_name,
        # Overall
        'BLEU-1':  round(results_summary['BLEU-1']['mean']  * 100, 2),
        'BLEU-2':  round(results_summary['BLEU-2']['mean']  * 100, 2),
        'BLEU-3':  round(results_summary['BLEU-3']['mean']  * 100, 2),
        'BLEU-4':  round(results_summary['BLEU-4']['mean']  * 100, 2),
        'ROUGE-1': round(results_summary['ROUGE-1']['mean'] * 100, 2),
        'ROUGE-2': round(results_summary['ROUGE-2']['mean'] * 100, 2),
        'ROUGE-L': round(results_summary['ROUGE-L']['mean'] * 100, 2),
        'METEOR':  round(results_summary['METEOR']['mean']  * 100, 2),
        'SPICE':   round(results_summary['SPICE']['mean']   * 100, 2),
        'ACC':     round(results_summary['ACC']['mean']     * 100, 2),
        # Per-task accuracy
        **task_acc,
    }

    table_path = output_dir / 'table_results.json'
    with open(table_path, 'w') as f:
        json.dump(table_results, f, indent=2)
    print(f"✓ Table results saved to: {table_path}")

    # ── Print sample predictions ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    # Show 1 sample per task type
    shown_tasks = set()
    for pred in all_predictions:
        task = pred['task_type']
        if task in shown_tasks:
            continue
        shown_tasks.add(task)
        print(f"\n{'='*60}")
        print(f"Video: {pred['video_id']}  |  Task: {task}")
        print(f"{'='*60}")
        print(f"Question:  {pred['question']}")
        print(f"Generated: {pred['generated_answer']}")
        print(f"Reference: {pred['reference']}")
        print(f"Metrics → BLEU-4: {pred.get('BLEU-4', 0):.4f} | "
              f"METEOR: {pred.get('METEOR', 0):.4f} | "
              f"SPICE: {pred.get('SPICE', 0):.4f} | "
              f"ACC: {pred.get('ACC', 0):.4f}")
        if len(shown_tasks) == 5:
            break

    # ── Print final summary table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE — CAPITA")
    print("=" * 70)
    print(f"\n{'Metric':<14} {'Overall':>10}  {'Caption':>10}  {'Motion':>10}")
    print("-" * 50)
    for m in ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','SPICE','ROUGE-L']:
        overall = results_summary[m]['mean'] * 100
        cap     = per_task_summary.get('caption',{}).get(m,{}).get('mean',0) * 100
        mot     = per_task_summary.get('motion', {}).get(m,{}).get('mean',0) * 100
        print(f"  {m:<12} {overall:>10.2f}  {cap:>10.2f}  {mot:>10.2f}")
    print("-" * 50)
    print(f"\n{'Task':<20} {'ACC':>10}")
    print("-" * 35)
    for task, label in [
        ('yes_no',      'Yes/No'),
        ('size',        'UAV Size'),
        ('environment', 'Environment'),
    ]:
        acc = per_task_summary.get(task, {}).get('ACC', {}).get('mean', 0) * 100
        print(f"  {label:<18} {acc:>10.2f}")

    print(f"\n{'Task':<25} {'ACC':>10}  {'BLEU-4':>10}  {'METEOR':>10}")
    print("-" * 60)
    for task_key, task_label in [
        ("caption",            "Caption"),
        ("yes_no",             "Yes/No"),
        ("motion_description", "Motion"),
        ("uav_size",           "UAV Size"),
        ("environment",        "Environment"),
    ]:
        t     = per_task_summary.get(task_key, {})
        acc   = t.get("ACC",    {}).get("mean", 0) * 100
        bleu4 = t.get("BLEU-4", {}).get("mean", 0) * 100
        meteor= t.get("METEOR", {}).get("mean", 0) * 100
        print(f"  {task_label:<23} {acc:>10.2f}  {bleu4:>10.2f}  {meteor:>10.2f}")
    print("-" * 60)

    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CAPITA Evaluation")
    parser.add_argument("--dataset",    type=str, default="MultiUAV",
                        choices=["MultiUAV", "Anti-UAV", "NPS"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth or checkpoint_epoch_XXXX.pth")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (1 recommended)")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["test", "train"])
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = CAPITAConfig()
    cfg.data.dataset_name  = args.dataset
    cfg.training.batch_size = args.batch_size
    device = torch.device(cfg.training.device)

    output_dir = Path(args.output_dir) / args.dataset

    print("=" * 70)
    print(f"CAPITA Evaluation | Dataset: {args.dataset} | Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    print("\nBuilding dataset...")
    eval_dataset = CAPITADataset(cfg, split=args.split)
    eval_loader  = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=capita_collate_fn,
        drop_last=False,
    )
    print(f"Evaluation samples: {len(eval_dataset)}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading CAPITA model...")
    model = CAPITAModel(cfg).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle both raw state_dict and full checkpoint formats
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"✓ Checkpoint loaded: {args.checkpoint}")

    # ── Metrics calculator ───────────────────────────────────────────────────
    metrics_calc = TextMetrics()

    # ── Run evaluation ───────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    all_predictions, results_summary, per_task_summary = evaluate(
        model, eval_loader, cfg, metrics_calc, device
    )

    # ── Save & display results ───────────────────────────────────────────────
    save_results(
        all_predictions, results_summary, per_task_summary,
        output_dir, args.dataset
    )


if __name__ == "__main__":
    main()