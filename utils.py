import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import os
from pathlib import Path



def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_yolo_boxes(txt_path):
    """
    Load YOLO format boxes from txt file
    Returns: tensor of shape (N, 4) in [cx, cy, w, h] format (normalized)
    """
    if not os.path.exists(txt_path):
        return torch.zeros(0, 4)
    
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, cx, cy, w, h = map(float, parts[:5])
                boxes.append([cx, cy, w, h])
    
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)

def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU
    boxes1, boxes2: (N, 4) in xyxy format
    Returns: (N,) GIoU values
    """
    # IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)
    
    # Enclosing box
    lt_enclosing = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, 0] * wh_enclosing[:, 1]
    
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)
    return giou

def hungarian_matcher(pred_boxes, gt_boxes):
    """
    Hungarian matching
    pred_boxes: (Q, 4) in cxcywh format
    gt_boxes: (N, 4) in cxcywh format
    Returns: (matched_pred_idx, matched_gt_idx)
    """
    if gt_boxes.shape[0] == 0:
        return [], []
    
    if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
        return [], []
    
    # Cost matrix (L1 distance)
    cost = torch.cdist(pred_boxes, gt_boxes, p=1)
    
    if torch.isnan(cost).any() or torch.isinf(cost).any():
        return [], []
    
    # Hungarian algorithm
    pred_idx, gt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
    
    return pred_idx, gt_idx



def save_checkpoint(model, optimizer, epoch, loss, filepath, save_llm=True):
    """Save checkpoint with LoRA support"""
    from pathlib import Path
    
    is_lora = hasattr(model.llm, 'save_pretrained')
    
    if is_lora:
        save_dir = Path(filepath).parent / Path(filepath).stem
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        model.llm.save_pretrained(save_dir / "lora_adapters")
        
        # Save vision components - AUTO-DETECT MODEL TYPE
        state_dict = {
            'epoch': epoch,
            'loss': loss,
            'backbone': model.backbone.state_dict(),
            'temporal_linker': model.temporal_linker.state_dict(),
            'box_head': model.box_head.state_dict(),
            'scene_mlp': model.scene_mlp.state_dict(),
            'visual_proj': model.visual_proj.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # Add TCTR or DVTR
        if hasattr(model, 'tctr'):
            state_dict['tctr'] = model.tctr.state_dict()
            print("  Saving TCTR model...")
        elif hasattr(model, 'dvtr'):
            state_dict['dvtr'] = model.dvtr.state_dict()
            print("  Saving DVTR model...")
        
        torch.save(state_dict, save_dir / "vision_components.pt")
        
        size_mb = sum(
            f.stat().st_size for f in save_dir.rglob('*') if f.is_file()
        ) / (1024 * 1024)
        print(f"  Checkpoint saved: {save_dir} ({size_mb:.1f} MB)")
    else:
        # Non-LoRA
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  Checkpoint saved: {filepath} ({size_mb:.1f} MB)")


        

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath} (Epoch {epoch}, Loss {loss:.4f})")
    return epoch, loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.01):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print("  Early stopping triggered!")
        
        return self.should_stop