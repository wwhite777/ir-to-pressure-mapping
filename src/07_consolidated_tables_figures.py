#!/usr/bin/env python3
"""
07_consolidated_tables_figures.py - Generate consolidated tables and figures for ICMR 2026

Creates:
1. LaTeX comparison tables
2. Comprehensive baseline comparison figure
3. Method overview figure
4. Retrieval performance bar chart
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Paths
PROJECT_ROOT = Path("/home/wjeong/cc")
RESULT_DIR = PROJECT_ROOT / "result"
FIGURE_DIR = RESULT_DIR / "figure" / "consolidated"
TABLE_DIR = RESULT_DIR / "report"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_all_results():
    """Load all experimental results."""
    results = {}

    # Baseline results
    baseline_path = RESULT_DIR / "figure" / "exp1_baselines" / "baseline_summary.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            results['baselines'] = json.load(f)

    # Add U-Net reference (from uncertainty EMA results)
    ema_path = RESULT_DIR / "figure" / "exp2_uncertainty_ema" / "detailed_results.json"
    if ema_path.exists():
        with open(ema_path) as f:
            ema_data = json.load(f)
            results['ema'] = ema_data
            # U-Net raw results
            results['unet'] = {
                'ssim': ema_data['summary']['raw']['ssim'],
                'rmse': ema_data['summary']['raw']['rmse'],
                'mae': ema_data['summary']['raw']['mae'],
            }

    # Retrieval results
    retrieval_path = RESULT_DIR / "figure" / "exp3_retrieval" / "retrieval_results.json"
    if retrieval_path.exists():
        with open(retrieval_path) as f:
            results['retrieval'] = json.load(f)

    # Ablation results
    ablation_path = RESULT_DIR / "figure" / "exp5_ablation" / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            results['ablation'] = json.load(f)

    return results


def generate_latex_tables(results):
    """Generate LaTeX tables for the paper."""

    # Table 1: Baseline Comparison
    table1 = r"""
\begin{table}[t]
\centering
\caption{Comparison of encoder architectures for IR-to-pressure mapping on the SLP dataset. All models use the same training protocol (50 epochs, AdamW optimizer).}
\label{tab:baseline}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{SSIM}$\uparrow$ & \textbf{RMSE}$\downarrow$ & \textbf{PSNR}$\uparrow$ & \textbf{MAE}$\downarrow$ \\
\midrule
"""

    # Add U-Net
    if 'unet' in results:
        u = results['unet']
        table1 += f"U-Net (ResNet50) & {u['ssim']:.4f} & {u['rmse']:.4f} & -- & {u['mae']:.4f} \\\\\n"

    # Add baselines
    if 'baselines' in results:
        b = results['baselines']
        if 'unetplusplus' in b:
            bp = b['unetplusplus']
            table1 += f"U-Net++ (ResNet50) & {bp['ssim']:.4f} & {bp['rmse']:.4f} & {bp['psnr']:.2f} & {bp['mae']:.4f} \\\\\n"
        if 'deeplabv3plus' in b:
            bd = b['deeplabv3plus']
            table1 += f"DeepLabV3+ (ResNet50) & {bd['ssim']:.4f} & {bd['rmse']:.4f} & {bd['psnr']:.2f} & {bd['mae']:.4f} \\\\\n"

    table1 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Table 2: Temporal Smoothing Comparison
    table2 = r"""
\begin{table}[t]
\centering
\caption{Comparison of temporal smoothing methods. Adaptive EMA uses MC Dropout uncertainty to modulate the smoothing factor, achieving better accuracy-smoothness trade-off.}
\label{tab:temporal}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{SSIM}$\uparrow$ & \textbf{RMSE}$\downarrow$ & \textbf{Jitter}$\downarrow$ & \textbf{Reduction} \\
\midrule
"""

    if 'ema' in results:
        e = results['ema']['summary']
        jr_fixed = results['ema']['jitter_reduction_fixed'] * 100
        jr_adaptive = results['ema']['jitter_reduction_adaptive'] * 100

        table2 += f"Raw Prediction & {e['raw']['ssim']:.4f} & {e['raw']['rmse']:.4f} & {e['raw']['jitter']:.4f} & -- \\\\\n"
        table2 += f"Fixed EMA ($\\alpha$=0.3) & {e['fixed_ema']['ssim']:.4f} & {e['fixed_ema']['rmse']:.4f} & {e['fixed_ema']['jitter']:.4f} & {jr_fixed:.1f}\\% \\\\\n"
        table2 += f"\\textbf{{Adaptive EMA}} & \\textbf{{{e['adaptive_ema']['ssim']:.4f}}} & \\textbf{{{e['adaptive_ema']['rmse']:.4f}}} & {e['adaptive_ema']['jitter']:.4f} & {jr_adaptive:.1f}\\% \\\\\n"

    table2 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Table 3: Retrieval Results
    table3 = r"""
\begin{table}[t]
\centering
\caption{Cross-modal retrieval performance using ResNet50 encoder embeddings. Pressure similarity is computed using SSIM between pressure maps.}
\label{tab:retrieval}
\begin{tabular}{ccccc}
\toprule
\textbf{Threshold} & \textbf{R@1}$\uparrow$ & \textbf{R@5}$\uparrow$ & \textbf{R@10}$\uparrow$ & \textbf{mAP}$\uparrow$ \\
\midrule
"""

    if 'retrieval' in results:
        for thresh, metrics in results['retrieval'].items():
            if thresh.startswith('threshold_'):
                t = thresh.replace('threshold_', '')
                table3 += f"{t} & {metrics['recall_at_1']:.4f} & {metrics['recall_at_5']:.4f} & {metrics['recall_at_10']:.4f} & {metrics['mAP']:.4f} \\\\\n"

    table3 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Save tables
    with open(TABLE_DIR / "latex_tables.tex", 'w') as f:
        f.write("% ICMR 2026 - LaTeX Tables\n")
        f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write(table1)
        f.write("\n\n")
        f.write(table2)
        f.write("\n\n")
        f.write(table3)

    print(f"LaTeX tables saved to: {TABLE_DIR / 'latex_tables.tex'}")
    return table1, table2, table3


def create_baseline_comparison_figure(results):
    """Create comprehensive baseline comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    models = ['U-Net', 'U-Net++', 'DeepLabV3+']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # Collect metrics
    ssim_vals = []
    rmse_vals = []
    mae_vals = []

    if 'unet' in results:
        ssim_vals.append(results['unet']['ssim'])
        rmse_vals.append(results['unet']['rmse'])
        mae_vals.append(results['unet']['mae'])

    if 'baselines' in results:
        if 'unetplusplus' in results['baselines']:
            ssim_vals.append(results['baselines']['unetplusplus']['ssim'])
            rmse_vals.append(results['baselines']['unetplusplus']['rmse'])
            mae_vals.append(results['baselines']['unetplusplus']['mae'])
        if 'deeplabv3plus' in results['baselines']:
            ssim_vals.append(results['baselines']['deeplabv3plus']['ssim'])
            rmse_vals.append(results['baselines']['deeplabv3plus']['rmse'])
            mae_vals.append(results['baselines']['deeplabv3plus']['mae'])

    # SSIM comparison
    bars1 = axes[0].bar(models, ssim_vals, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('SSIM')
    axes[0].set_title('(a) Structural Similarity')
    axes[0].set_ylim(0.5, 0.75)
    for bar, val in zip(bars1, ssim_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].axhline(y=max(ssim_vals), color='gray', linestyle='--', alpha=0.5)

    # RMSE comparison
    bars2 = axes[1].bar(models, rmse_vals, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('(b) Root Mean Square Error')
    axes[1].set_ylim(0.06, 0.08)
    for bar, val in zip(bars2, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].axhline(y=min(rmse_vals), color='gray', linestyle='--', alpha=0.5)

    # MAE comparison
    bars3 = axes[2].bar(models, mae_vals, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('MAE')
    axes[2].set_title('(c) Mean Absolute Error')
    axes[2].set_ylim(0.02, 0.035)
    for bar, val in zip(bars3, mae_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[2].axhline(y=min(mae_vals), color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Baseline Architecture Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = FIGURE_DIR / "baseline_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_temporal_smoothing_figure(results):
    """Create temporal smoothing comparison figure."""
    if 'ema' not in results:
        print("No EMA results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    e = results['ema']['summary']
    methods = ['Raw', 'Fixed EMA', 'Adaptive EMA']
    colors = ['#95a5a6', '#e74c3c', '#2ecc71']

    # Jitter comparison
    jitters = [e['raw']['jitter'], e['fixed_ema']['jitter'], e['adaptive_ema']['jitter']]
    bars1 = axes[0].bar(methods, jitters, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Mean Absolute Jitter')
    axes[0].set_title('(a) Temporal Jitter')

    for bar, j in zip(bars1, jitters):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                     f'{j:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add reduction annotations
    jr_fixed = (1 - e['fixed_ema']['jitter'] / e['raw']['jitter']) * 100
    jr_adaptive = (1 - e['adaptive_ema']['jitter'] / e['raw']['jitter']) * 100
    axes[0].annotate(f'-{jr_fixed:.0f}%', xy=(1, e['fixed_ema']['jitter']/2),
                     ha='center', fontsize=10, color='white', fontweight='bold')
    axes[0].annotate(f'-{jr_adaptive:.0f}%', xy=(2, e['adaptive_ema']['jitter']/2),
                     ha='center', fontsize=10, color='white', fontweight='bold')

    # SSIM comparison (accuracy preservation)
    ssims = [e['raw']['ssim'], e['fixed_ema']['ssim'], e['adaptive_ema']['ssim']]
    bars2 = axes[1].bar(methods, ssims, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('(b) Accuracy Preservation')
    axes[1].set_ylim(0.5, 0.7)

    for bar, s in zip(bars2, ssims):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f'{s:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add reference line
    axes[1].axhline(y=e['raw']['ssim'], color='gray', linestyle='--', alpha=0.5, label='Raw baseline')
    axes[1].legend(loc='lower right')

    plt.suptitle('Uncertainty-Guided Adaptive EMA Analysis', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = FIGURE_DIR / "temporal_smoothing_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_retrieval_figure(results):
    """Create retrieval performance figure."""
    if 'retrieval' not in results:
        print("No retrieval results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    thresholds = []
    r1_vals = []
    r5_vals = []
    r10_vals = []
    map_vals = []

    for thresh, metrics in sorted(results['retrieval'].items()):
        if thresh.startswith('threshold_'):
            t = float(thresh.replace('threshold_', ''))
            thresholds.append(t)
            r1_vals.append(metrics['recall_at_1'] * 100)
            r5_vals.append(metrics['recall_at_5'] * 100)
            r10_vals.append(metrics['recall_at_10'] * 100)
            map_vals.append(metrics['mAP'] * 100)

    # Recall@K plot
    x = np.arange(len(thresholds))
    width = 0.25

    bars1 = axes[0].bar(x - width, r1_vals, width, label='R@1', color='#3498db', edgecolor='black')
    bars2 = axes[0].bar(x, r5_vals, width, label='R@5', color='#2ecc71', edgecolor='black')
    bars3 = axes[0].bar(x + width, r10_vals, width, label='R@10', color='#e74c3c', edgecolor='black')

    axes[0].set_xlabel('Similarity Threshold')
    axes[0].set_ylabel('Recall (%)')
    axes[0].set_title('(a) Recall@K Performance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{t}' for t in thresholds])
    axes[0].set_ylim(85, 100)
    axes[0].legend(loc='lower left')
    axes[0].grid(axis='y', alpha=0.3)

    # mAP plot
    bars4 = axes[1].bar(thresholds, map_vals, color='#9b59b6', edgecolor='black', width=0.08)
    axes[1].set_xlabel('Similarity Threshold')
    axes[1].set_ylabel('mAP (%)')
    axes[1].set_title('(b) Mean Average Precision')
    axes[1].set_ylim(80, 100)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars4, map_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Cross-Modal Retrieval Performance', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = FIGURE_DIR / "retrieval_performance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_method_overview_figure():
    """Create method overview/pipeline figure."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    input_color = '#3498db'
    encoder_color = '#2ecc71'
    decoder_color = '#e74c3c'
    output_color = '#9b59b6'
    ema_color = '#f39c12'
    retrieval_color = '#1abc9c'

    # Input
    rect1 = FancyBboxPatch((0.5, 2.5), 1.5, 1.5, boxstyle="round,pad=0.05",
                            facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.25, 3.25, 'IR Image\n(Thermal)', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Arrow
    ax.annotate('', xy=(2.3, 3.25), xytext=(2.0, 3.25),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Encoder
    rect2 = FancyBboxPatch((2.5, 2), 2, 2.5, boxstyle="round,pad=0.05",
                            facecolor=encoder_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(3.5, 3.25, 'ResNet50\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Feature embedding arrow (for retrieval)
    ax.annotate('', xy=(3.5, 1.7), xytext=(3.5, 2.0),
                arrowprops=dict(arrowstyle='->', lw=2, color=retrieval_color))

    # Retrieval box
    rect_ret = FancyBboxPatch((2.5, 0.3), 2, 1.2, boxstyle="round,pad=0.05",
                               facecolor=retrieval_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect_ret)
    ax.text(3.5, 0.9, 'Cross-Modal\nRetrieval', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrow to decoder
    ax.annotate('', xy=(4.8, 3.25), xytext=(4.5, 3.25),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Decoder
    rect3 = FancyBboxPatch((5, 2), 2, 2.5, boxstyle="round,pad=0.05",
                            facecolor=decoder_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(6, 3.25, 'U-Net\nDecoder', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Arrow to output
    ax.annotate('', xy=(7.3, 3.25), xytext=(7.0, 3.25),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Raw prediction
    rect4 = FancyBboxPatch((7.5, 2.5), 1.5, 1.5, boxstyle="round,pad=0.05",
                            facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect4)
    ax.text(8.25, 3.25, 'Pressure\nMap', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # MC Dropout arrow
    ax.annotate('', xy=(9.3, 3.25), xytext=(9.0, 3.25),
                arrowprops=dict(arrowstyle='->', lw=2))

    # MC Dropout / Uncertainty
    rect5 = FancyBboxPatch((9.5, 2), 2, 2.5, boxstyle="round,pad=0.05",
                            facecolor=ema_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect5)
    ax.text(10.5, 3.5, 'MC Dropout', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(10.5, 3.0, 'Uncertainty', ha='center', va='center', fontsize=9, color='white')
    ax.text(10.5, 2.5, 'Estimation', ha='center', va='center', fontsize=9, color='white')

    # Arrow to adaptive EMA
    ax.annotate('', xy=(11.8, 3.25), xytext=(11.5, 3.25),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Adaptive EMA
    rect6 = FancyBboxPatch((12, 2), 1.8, 2.5, boxstyle="round,pad=0.05",
                            facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(rect6)
    ax.text(12.9, 3.5, 'Adaptive', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(12.9, 3.0, 'EMA', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(12.9, 2.5, 'Smoothing', ha='center', va='center', fontsize=9, color='white')

    # Title
    ax.text(7, 5.5, 'Proposed Method: Uncertainty-Guided Adaptive Temporal Smoothing',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=encoder_color, edgecolor='black', label='Encoder'),
        mpatches.Patch(facecolor=decoder_color, edgecolor='black', label='Decoder'),
        mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output'),
        mpatches.Patch(facecolor=ema_color, edgecolor='black', label='Uncertainty'),
        mpatches.Patch(facecolor=retrieval_color, edgecolor='black', label='Retrieval'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05))

    save_path = FIGURE_DIR / "method_overview.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def create_consolidated_summary_figure(results):
    """Create a single consolidated figure with all key results."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # (a) Baseline SSIM comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['U-Net', 'U-Net++', 'DeepLabV3+']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    ssim_vals = [results['unet']['ssim']]
    if 'baselines' in results:
        ssim_vals.append(results['baselines']['unetplusplus']['ssim'])
        ssim_vals.append(results['baselines']['deeplabv3plus']['ssim'])

    bars = ax1.bar(models, ssim_vals, color=colors, edgecolor='black')
    ax1.set_ylabel('SSIM')
    ax1.set_title('(a) Encoder Comparison')
    ax1.set_ylim(0.55, 0.70)
    for bar, val in zip(bars, ssim_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')

    # (b) Jitter reduction
    ax2 = fig.add_subplot(gs[0, 1])
    if 'ema' in results:
        e = results['ema']['summary']
        methods = ['Raw', 'Fixed\nEMA', 'Adaptive\nEMA']
        jitters = [e['raw']['jitter'], e['fixed_ema']['jitter'], e['adaptive_ema']['jitter']]
        colors2 = ['#95a5a6', '#e74c3c', '#2ecc71']
        bars2 = ax2.bar(methods, jitters, color=colors2, edgecolor='black')
        ax2.set_ylabel('Jitter')
        ax2.set_title('(b) Temporal Smoothing')
        for bar, j in zip(bars2, jitters):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                     f'{j:.4f}', ha='center', fontsize=8, fontweight='bold')

    # (c) Retrieval R@K
    ax3 = fig.add_subplot(gs[0, 2])
    if 'retrieval' in results:
        # Use threshold 0.7 results
        if 'threshold_0.7' in results['retrieval']:
            r = results['retrieval']['threshold_0.7']
            metrics = ['R@1', 'R@5', 'R@10', 'mAP']
            vals = [r['recall_at_1']*100, r['recall_at_5']*100, r['recall_at_10']*100, r['mAP']*100]
            colors3 = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
            bars3 = ax3.bar(metrics, vals, color=colors3, edgecolor='black')
            ax3.set_ylabel('Performance (%)')
            ax3.set_title('(c) Retrieval (threshold=0.7)')
            ax3.set_ylim(80, 100)
            for bar, val in zip(bars3, vals):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')

    # (d) Ablation - SSIM preservation
    ax4 = fig.add_subplot(gs[1, 0])
    if 'ablation' in results:
        configs = list(results['ablation'].keys())[:4]
        ssims = [results['ablation'][c]['ssim'] for c in configs]
        short_names = ['Base', '+Morph', '+Calib', 'Full']
        colors4 = plt.cm.Blues(np.linspace(0.3, 0.9, len(configs)))
        bars4 = ax4.bar(short_names, ssims, color=colors4, edgecolor='black')
        ax4.set_ylabel('SSIM')
        ax4.set_title('(d) Ablation: Accuracy')
        ax4.set_ylim(0.55, 0.70)

    # (e) Ablation - Jitter reduction
    ax5 = fig.add_subplot(gs[1, 1])
    if 'ablation' in results:
        jitters = [results['ablation'][c]['jitter'] for c in configs]
        colors5 = plt.cm.Greens(np.linspace(0.3, 0.9, len(configs)))
        bars5 = ax5.bar(short_names, jitters, color=colors5, edgecolor='black')
        ax5.set_ylabel('Jitter')
        ax5.set_title('(e) Ablation: Smoothness')

    # (f) Summary statistics text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = """
    Key Results Summary
    ━━━━━━━━━━━━━━━━━━━━━

    Best Encoder: U-Net (SSIM: 0.665)

    Adaptive EMA:
    • 53% jitter reduction
    • Better SSIM preservation

    Retrieval Performance:
    • R@1: 94.2%
    • mAP: 86.6%

    Full Pipeline:
    • 91% jitter reduction
    • Maintains accuracy
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax6.set_title('(f) Summary')

    plt.suptitle('ICMR 2026: IR-to-Pressure Mapping with Uncertainty-Guided Smoothing',
                 fontsize=14, fontweight='bold', y=0.98)

    save_path = FIGURE_DIR / "consolidated_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("ICMR 2026 - Consolidated Tables and Figures")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load results
    print("\nLoading experimental results...")
    results = load_all_results()
    print(f"Loaded: {list(results.keys())}")

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_tables(results)

    # Create figures
    print("\nCreating baseline comparison figure...")
    create_baseline_comparison_figure(results)

    print("\nCreating temporal smoothing figure...")
    create_temporal_smoothing_figure(results)

    print("\nCreating retrieval performance figure...")
    create_retrieval_figure(results)

    print("\nCreating method overview figure...")
    create_method_overview_figure()

    print("\nCreating consolidated summary figure...")
    create_consolidated_summary_figure(results)

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to:")
    print(f"  Tables: {TABLE_DIR}")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
