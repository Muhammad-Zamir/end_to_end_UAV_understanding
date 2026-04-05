"""
CAPITA Publication-Quality Visualizations (Revised)
=====================================================
Figure 1: Annotated FFT Frequency Analysis  (2 rows x 4 cols)
Figure 2: Single-Video Temporal Attention   (frames + bar + heatmap)

Usage:
    python visualize.py \
        --checkpoint ./checkpoints/capita_multiuav/best_model.pth \
        --dataset MultiUAV \
        --output_dir ./paper_figures \
        --n_samples 4 \
        --attn_video 0
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CAPITAConfig
from dataset import CAPITADataset, capita_collate_fn
from model import CAPITAModel


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_rgb(t):
    x = t.cpu().float().numpy()
    x = np.clip(x.transpose(1, 2, 0), 0, 1)
    return (x * 255).astype(np.uint8)


def compute_fft(roi_tensor, model):
    inp     = roi_tensor.unsqueeze(0).float()
    fft_out = model.dual_stream_encoder.appearance_enc.compute_fft_features(inp)
    fft_np  = fft_out.squeeze(0).cpu().float().numpy()
    return fft_np.mean(axis=0)


def find_dominant_direction(fft_map):
    h, w   = fft_map.shape
    cx, cy = w // 2, h // 2
    masked = fft_map.copy()
    masked[cy-3:cy+4, cx-3:cx+4] = 0
    flat_idx = np.argmax(masked)
    py, px   = np.unravel_index(flat_idx, masked.shape)
    angle    = np.degrees(np.arctan2(py - cy, px - cx)) % 180
    return angle, px, py


def draw_annotated_fft(ax, fft_map):
    h, w   = fft_map.shape
    cx, cy = w // 2, h // 2
    vmin   = max(fft_map.min(), 0.01)
    im     = ax.imshow(fft_map, cmap='inferno',
                       norm=LogNorm(vmin=vmin, vmax=fft_map.max()),
                       interpolation='nearest')

    angle, px, py = find_dominant_direction(fft_map)

    # DC component
    dc = Circle((cx, cy), radius=3, fill=False,
                edgecolor='#00FF88', linewidth=1.4, linestyle='--')
    ax.add_patch(dc)
    ax.annotate('DC', xy=(cx, cy - 4.5), fontsize=8, color='#00FF88',
                ha='center', va='bottom',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # Motion orientation arrow
    r  = min(h, w) * 0.28
    ex = cx + r * np.cos(np.radians(angle))
    ey = cy + r * np.sin(np.radians(angle))
    ax.annotate('', xy=(ex, ey), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.4))
    lx = cx + (r + 5) * np.cos(np.radians(angle))
    ly = cy + (r + 5) * np.sin(np.radians(angle))
    ax.annotate('motion\ndir', xy=(lx, ly), fontsize=8, color='#FFD700',
                ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # High-frequency blur ring
    hfr = min(h, w) * 0.38
    hf  = Circle((cx, cy), radius=hfr, fill=False,
                 edgecolor='#FF6B6B', linewidth=1.0, linestyle=':', alpha=0.85)
    ax.add_patch(hf)
    ax.annotate('blur\nenergy', xy=(cx + hfr * 0.7, cy - hfr * 0.7),
                fontsize=8, color='#FF6B6B', ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    # Rotor harmonic peak
    ax.plot(px, py, 'o', color='#00BFFF', markersize=4,
            markeredgecolor='white', markeredgewidth=0.6)
    ax.annotate('rotor\nharmonic', xy=(px, py),
                xytext=(min(px + 5, w - 1), max(py - 5, 0)),
                fontsize=8, color='#00BFFF', ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#00BFFF', lw=0.7),
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

    ax.set_xticks([])
    ax.set_yticks([])
    return im


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — FFT  (2 rows x 4 cols)
# ─────────────────────────────────────────────────────────────────────────────

def generate_fft_figure(model, loader, device, output_dir, n_samples=4):
    print("Collecting ROI patches for FFT figure...")
    samples = []

    for batch in loader:
        if len(samples) >= n_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        roi  = batch["roi_patches"]    # [B,T,M,3,rs,rs]
        mask = batch["drone_mask"]
        B, T, M, C, rs, _ = roi.shape
        t_mid = T // 2

        for b in range(B):
            if len(samples) >= n_samples:
                break
            valid = (mask[b, t_mid] > 0.5).nonzero(as_tuple=True)[0]
            if not len(valid):
                continue
            patch = roi[b, t_mid, valid[0].item()]
            with torch.no_grad():
                fft_map = compute_fft(patch, model)
            samples.append({
                "video_id": batch["video_id"][b],
                "roi_rgb":  tensor_to_rgb(patch),
                "fft_map":  fft_map,
            })

    if not samples:
        print("No samples found for FFT figure"); return

    n   = len(samples)
    fig = plt.figure(figsize=(2.8 * n, 5.6))
    fig.patch.set_facecolor('white')
    gs  = plt.GridSpec(2, n, hspace=0.08, wspace=0.05,
                       top=0.80, bottom=0.10, left=0.07, right=0.91)
    last_im = None

    for col, s in enumerate(samples):
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(s["roi_rgb"], interpolation='bilinear')
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_edgecolor('#bbbbbb'); sp.set_linewidth(0.6)
        ax0.set_title(f"UAV-{s['video_id'].split('-')[-1]}",
                      fontsize=16, pad=3, color='#333333')
        if col == 0:
            ax0.set_ylabel("Pixel domain", fontsize=16,
                           labelpad=5, color='#222222')

        ax1 = fig.add_subplot(gs[1, col])
        last_im = draw_annotated_fft(ax1, s["fft_map"])
        for sp in ax1.spines.values():
            sp.set_edgecolor('#bbbbbb'); sp.set_linewidth(0.6)
        if col == 0:
            ax1.set_ylabel("Frequency domain", fontsize=16,
                           labelpad=5, color='#222222')

    if last_im is not None:
        cbar_ax = fig.add_axes([0.925, 0.13, 0.015, 0.35])
        cbar    = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label('Log magnitude', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='#00FF88',
                       linestyle='--', linewidth=1.2, label='DC component'),
        mpatches.Patch(facecolor='none', edgecolor='#FFD700',
                       linewidth=1.2, label='Motion orientation'),
        mpatches.Patch(facecolor='none', edgecolor='#FF6B6B',
                       linestyle=':', linewidth=1.2, label='Blur energy ring'),
        mpatches.Patch(color='#00BFFF', linewidth=1.2,
                       label='Rotor harmonic peak'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=14, framealpha=0.9, bbox_to_anchor=(0.5, 0.01),
               handlelength=1.5, columnspacing=1.0)

    fig.suptitle(
        "Pixel vs. Frequency Domain UAV Signatures",
        fontsize=18, fontweight='bold', y=0.95, color='#111111')

    for fmt in ('pdf', 'png'):
        fig.savefig(output_dir / f"fig1_fft_visualization.{fmt}",
                    dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Figure 1 saved to {output_dir}/fig1_fft_visualization.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — TEMPORAL ATTENTION  (single video)
# ─────────────────────────────────────────────────────────────────────────────

def register_hook(model):
    store = {}
    def fn(module, inp, out):
        store['w'] = out.detach().cpu()
    h = model.swarm_gnn.temporal_attn_pool.register_forward_hook(fn)
    return store, h


def get_thumbs(batch, b, device, T, size=64):
    frames  = batch["frames"]           # [B, T, 3, H, W]
    T_data  = frames.shape[1]
    indices = np.linspace(0, T_data - 1, T, dtype=int)
    thumbs  = []
    for t in indices:
        f = frames[b, t].cpu().float()
        f = F.interpolate(f.unsqueeze(0), size=(size, size),
                          mode='bilinear', align_corners=False).squeeze(0)
        thumbs.append(tensor_to_rgb(f))
    return thumbs


def generate_attention_figure(model, loader, device, output_dir,
                               target_idx=0):
    print("Extracting temporal attention...")
    store, handle = register_hook(model)
    chosen = None
    count  = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            _ = model(bg, generate=False)
            if 'w' not in store:
                store.clear(); continue

            B = store['w'].shape[0]
            for b in range(B):
                if count == target_idx:
                    chosen = {
                        "batch":    batch,
                        "b":        b,
                        "attn":     store['w'][b, :, 0].numpy(),
                        "video_id": batch["video_id"][b],
                        "caption":  batch.get("caption", [""])[b],
                    }
                count += 1
            store.clear()
            if chosen: break

    handle.remove()
    if not chosen:
        print("Target video not found"); return

    attn     = chosen["attn"]
    attn     = attn / attn.sum()           # ensure sums to 1
    T        = len(attn)
    peak     = int(np.argmax(attn))
    uniform  = 1.0 / T
    caption  = chosen["caption"]
    cap_disp = (caption[:85] + "...") if len(caption) > 85 else caption

    thumbs = get_thumbs(chosen["batch"], chosen["b"], device, T)

    fig = plt.figure(figsize=(10.5, 5.8))
    fig.patch.set_facecolor('white')
    gs  = plt.GridSpec(3, 1, height_ratios=[2.2, 2.0, 0.65],
                       hspace=0.05, top=0.90, bottom=0.15,
                       left=0.06, right=0.97)

    # ── Row 0: frame thumbnails ────────────────────────────────────────────
    gs_f = gs[0].subgridspec(1, T, wspace=0.04)
    for t in range(T):
        ax = fig.add_subplot(gs_f[0, t])
        ax.imshow(thumbs[t], interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        color = '#EE3333' if t == peak else '#cccccc'
        lw    = 2.5       if t == peak else 0.5
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(lw)
        ax.set_xlabel(f"f{t+1}", fontsize=10, labelpad=2,
                      color='#EE3333' if t == peak else '#888888')
        if t == peak:
            ax.set_title("peak", fontsize=10, color='#EE3333',
                         pad=2, fontweight='bold')
        if t == 0:
            ax.set_ylabel("Frames", fontsize=12, labelpad=5)

    # ── Row 1: attention bar chart ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    bar_colors = ['#EE3333' if i == peak else '#7BA7D0' for i in range(T)]
    ax_b.bar(np.arange(T), attn, color=bar_colors, width=0.65,
             edgecolor='white', linewidth=0.5)

    ax_b.annotate(
        f"peak ({attn[peak]:.3f})",
        xy=(peak, attn[peak]),
        xytext=(peak + (1 if peak < T - 2 else -1.5),
                attn[peak] + 0.006),
        fontsize=10, color='#CC0000', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#CC0000', lw=0.9))

    ax_b.axhline(uniform, color='#999999', linestyle='--',
                 linewidth=0.9, alpha=0.8)
    ax_b.annotate(f'uniform (1/T={uniform:.3f})',
                  xy=(T - 0.5, uniform + 0.001), fontsize=10,
                  color='#777777', va='bottom', ha='right')

    ax_b.set_xticks(np.arange(T))
    ax_b.set_xticklabels([f"f{i+1}" for i in range(T)], fontsize=10)
    ax_b.set_ylabel("Attention\nweight", fontsize=10, labelpad=4)
    ax_b.set_xlim(-0.5, T - 0.5)
    ax_b.set_ylim(0, attn.max() * 1.40)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.tick_params(axis='y', labelsize=10)
    ax_b.grid(axis='y', linestyle=':', alpha=0.4, linewidth=0.6)
    ax_b.axvspan(peak - 0.45, peak + 0.45, alpha=0.10, color='#EE3333')

    # ── Row 2: 1×T heatmap ────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2])
    im   = ax_h.imshow(attn.reshape(1, -1), aspect='auto',
                       cmap='YlOrRd', vmin=0, vmax=attn.max(),
                       interpolation='nearest')
    ax_h.set_xticks(np.arange(T))
    ax_h.set_xticklabels([f"f{i+1}" for i in range(T)], fontsize=10)
    ax_h.set_yticks([])
    ax_h.set_ylabel("Attn", fontsize=10, labelpad=5)
    ax_h.plot(peak, 0, 's', color='white', markersize=5,
              markeredgecolor='#CC0000', markeredgewidth=1.2)
    fig.colorbar(im, ax=ax_h, orientation='vertical',
                 fraction=0.012, pad=0.01, shrink=0.85).ax.tick_params(labelsize=7)

    # Vertical guide lines
    for ax in [ax_b, ax_h]:
        ax.axvline(peak, color='#EE3333', linestyle='-',
                   alpha=0.15, linewidth=9)

    fig.suptitle(
        f"Spatio-Temporal Swarm Graph Network Learned Attention Weights",
        fontsize=14, fontweight='bold', y=0.87, color='#111111')
    fig.text(0.5, 0.06, f'Caption: "{cap_disp}"',
             ha='center', fontsize=10, color='#555555', fontstyle='italic')
    fig.text(0.5, 0.02,
             "Red = peak attention frame  |  Dashed = uniform baseline  |  "
             "Model attends to the most behaviourally informative moment",
             ha='center', fontsize=10, color='#666666')

    for fmt in ('pdf', 'png'):
        fig.savefig(output_dir / f"fig2_temporal_attention.{fmt}",
                    dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Figure 2 saved to {output_dir}/fig2_temporal_attention.pdf/.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--dataset",     default="MultiUAV",
                        choices=["MultiUAV", "Anti-UAV", "NPS"])
    parser.add_argument("--output_dir",  default="./paper_figures")
    parser.add_argument("--n_samples",   type=int, default=4)
    parser.add_argument("--attn_video",  type=int, default=0,
                        help="Test video index to use for attention figure")
    parser.add_argument("--vis",         default="both",
                        choices=["fft", "attention", "both"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CAPITAConfig()
    cfg.data.dataset_name = args.dataset
    device = torch.device(cfg.training.device)

    print(f"Loading CAPITA from {args.checkpoint}...")
    model = CAPITAModel(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model.eval()
    print("✓ Model loaded")

    loader = DataLoader(
        CAPITADataset(cfg, split="test"),
        batch_size=1, shuffle=False,
        num_workers=2, collate_fn=capita_collate_fn,
    )

    if args.vis in ("fft", "both"):
        generate_fft_figure(model, loader, device,
                            output_dir, n_samples=args.n_samples)

    if args.vis in ("attention", "both"):
        generate_attention_figure(model, loader, device,
                                  output_dir, target_idx=args.attn_video)

    print(f"\nDone. Figures in: {output_dir}")


if __name__ == "__main__":
    main()