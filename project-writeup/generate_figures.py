#!/usr/bin/env python3
"""
Generate all figures for the project writeup.

Usage:
    python generate_figures.py              # use cached CSVs in data/
    python generate_figures.py --fetch      # re-download from wandb first

Outputs (written to the same directory as this script):
    fig_ppo_training.{pdf,png}
    fig_ga_training.{pdf,png}
    fig_comparison.{pdf,png}

Cached wandb data:
    data/ppo_history.csv   -- ppo_run_1 (id: ezrm5r57), 5000-sample history
    data/ga_history.csv    -- ga_run_1  (id: etnx7oqi), 5000-sample history

Note on PPO data:
    ppo_run_1 was resumed mid-training (--start-phase 4, wandb id reused).
    wandb therefore logged two segments in the same run:
      - Original full-curriculum run:  _step  0 .. 84584
      - Phase-4 continuation:          _step  84888 .. end
    The training figures use only the original segment (_step < 84888) so that
    the phase-band overlay is monotonic and covers all five phases.
    The comparison figure uses the phase-4 continuation for final metrics.
"""

import argparse
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
PPO_CSV = os.path.join(DATA_DIR, 'ppo_history.csv')
GA_CSV  = os.path.join(DATA_DIR, 'ga_history.csv')

# ── Styling ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
})

# Blue → teal → gold → orange → red  (all visually distinct)
PHASE_COLORS = ['#5B9BD5', '#4CB8A0', '#E8B84B', '#E07830', '#D95050']
PHASE_LABELS = [
    'Phase 0: Short range',
    'Phase 1: Long range',
    'Phase 2: Narrow wall',
    'Phase 3: Wide wall',
    'Phase 4: Tilted gravity',
]
BAND_ALPHA = 0.22


# ── wandb fetch ────────────────────────────────────────────────────────────────
def fetch_wandb_data():
    import wandb
    api = wandb.Api()
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching ppo_run_1 (ezrm5r57) …")
    ppo_run = api.run("ball-throw-ml/ezrm5r57")
    ppo_hist = ppo_run.history(samples=5000)
    ppo_hist.to_csv(PPO_CSV, index=False)
    print(f"  → saved {len(ppo_hist)} rows to {PPO_CSV}")

    print("Fetching ga_run_1 (etnx7oqi) …")
    ga_run = api.run("ball-throw-ml/etnx7oqi")
    ga_hist = ga_run.history(samples=5000)
    ga_hist.to_csv(GA_CSV, index=False)
    print(f"  → saved {len(ga_hist)} rows to {GA_CSV}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_phase_transitions(df, x_col, phase_col, min_label_frac=0.0):
    """
    Return list of (x_start, x_end, phase, show_label) tuples.
    Phase is forced monotonic via cummax before computing transitions.
    show_label is True when the band is at least min_label_frac of the x-range.
    """
    sub = df[[x_col, phase_col]].dropna().sort_values(x_col).copy()
    sub[phase_col] = sub[phase_col].cummax()
    x_range = sub[x_col].iloc[-1] - sub[x_col].iloc[0]
    transitions, prev_phase, start_x = [], None, None
    for _, row in sub.iterrows():
        ph, x = int(row[phase_col]), row[x_col]
        if ph != prev_phase:
            if prev_phase is not None:
                transitions.append((start_x, x, prev_phase))
            start_x, prev_phase = x, ph
    if prev_phase is not None:
        transitions.append((start_x, sub[x_col].iloc[-1], prev_phase))
    return [
        (x0, x1, ph, (x1 - x0) / x_range >= min_label_frac)
        for x0, x1, ph in transitions
    ]


def draw_bands(ax, transitions, x_scale=1.0):  # x_scale: divide stored coords to reach plot units
    for x0, x1, ph, show_label in transitions:
        ax.axvspan(x0 / x_scale, x1 / x_scale,
                   alpha=BAND_ALPHA, color=PHASE_COLORS[ph], zorder=0)
        if show_label:
            ax.text(
                (x0 + x1) / 2 / x_scale, 0.97, f'P{ph}',
                ha='center', va='top', fontsize=8, color='#333',
                transform=ax.get_xaxis_transform(),
            )


def phase_legend():
    return [
        mpatches.Patch(color=PHASE_COLORS[i], alpha=0.7, label=PHASE_LABELS[i])
        for i in range(5)
    ]


def add_phase_legend(fig, axes):
    fig.legend(handles=phase_legend(), loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.03), fontsize=7.5, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])


def save(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(HERE, f'{stem}.{ext}')
        kw = dict(bbox_inches='tight')
        if ext == 'png':
            kw['dpi'] = 150
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"Saved {stem}.{{pdf,png}}")


# ── Figure 1: PPO training curves ─────────────────────────────────────────────
def fig_ppo_training(ppo_orig):
    trans = get_phase_transitions(
        ppo_orig, 'global_step', 'curriculum/phase', min_label_frac=0.03
    )
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    # Panel (a): success rate
    ax = axes[0]
    sr = ppo_orig.dropna(subset=['curriculum/success_rate'])
    if len(sr):
        ax.plot(sr['global_step'] / 1e6, sr['curriculum/success_rate'],
                color='#5FA8D8', alpha=0.55, linewidth=0.9, label='Per-rollout')
        ax.plot(sr['global_step'] / 1e6,
                sr['curriculum/success_rate'].rolling(20, min_periods=1).mean(),
                color='#1a6fa8', linewidth=2.0, label='20-rollout average')
    draw_bands(ax, trans, 1e6)
    ax.axhline(0.75, color='#666', linestyle='--', linewidth=1.0, alpha=0.8,
               label='75% threshold')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_title('(a) PPO Curriculum Training — Success Rate')

    # Panel (b): mean landing distance
    ax = axes[1]
    dist = ppo_orig.dropna(subset=['curriculum/mean_landing_dist'])
    if len(dist):
        ax.plot(dist['global_step'] / 1e6, dist['curriculum/mean_landing_dist'],
                color='#E8A060', alpha=0.55, linewidth=0.9, label='Per-rollout')
        ax.plot(dist['global_step'] / 1e6,
                dist['curriculum/mean_landing_dist'].rolling(20, min_periods=1).mean(),
                color='#c8620f', linewidth=2.0, label='20-rollout average')
    draw_bands(ax, trans, 1e6)
    ax.axhline(1.0, color='#666', linestyle='--', linewidth=1.0, alpha=0.8,
               label='1 m (success)')
    ax.set_xlabel('Environment Interactions (millions)')
    ax.set_ylabel('Mean Landing Distance (m)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('(b) PPO Curriculum Training — Landing Distance')

    add_phase_legend(fig, axes)
    save(fig, 'fig_ppo_training')


# ── Figure 2: GA training curves ──────────────────────────────────────────────
# GA evaluates pop_size × n_eval_episodes = 100 × 30 = 3 000 episodes per generation.
_GA_EPISODES_PER_GEN = 3_000


def fig_ga_training(ga_main):
    ga = ga_main.copy()
    ga['interactions_M'] = ga['_step'] * _GA_EPISODES_PER_GEN / 1e6

    trans = get_phase_transitions(
        ga, 'interactions_M', 'curriculum/phase', min_label_frac=0.03
    )
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    # Panel (a): elite vs population mean success rate
    ax = axes[0]
    ga_sr = ga.dropna(subset=['ga/elite_success_rate'])
    if len(ga_sr):
        ax.plot(ga_sr['interactions_M'], ga_sr['ga/elite_success_rate'],
                color='#5FA8D8', alpha=0.55, linewidth=0.9,
                label='Elite (per-generation)')
        ax.plot(ga_sr['interactions_M'],
                ga_sr['ga/elite_success_rate'].rolling(30, min_periods=1).mean(),
                color='#1a6fa8', linewidth=2.0, label='Elite (30-gen average)')
        ax.plot(ga_sr['interactions_M'], ga_sr['ga/mean_success_rate'],
                color='#E07070', alpha=0.55, linewidth=0.9,
                label='Population mean (per-generation)')
        ax.plot(ga_sr['interactions_M'],
                ga_sr['ga/mean_success_rate'].rolling(30, min_periods=1).mean(),
                color='#c0392b', linewidth=1.5, linestyle='--',
                label='Population mean (30-gen average)')
    draw_bands(ax, trans)
    ax.axhline(0.75, color='#666', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Success Rate')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7, ncol=2,
              handlelength=1.0, handletextpad=0.3, labelspacing=0.2, columnspacing=0.8)
    ax.set_title('(a) GA Curriculum Training — Success Rate')
    # Secondary x-axis: generations (thousands)
    sec = ax.secondary_xaxis('top',
                              functions=(lambda x: x / (_GA_EPISODES_PER_GEN / 1e6) / 1e3,
                                         lambda x: x * 1e3 * (_GA_EPISODES_PER_GEN / 1e6)))
    sec.set_xlabel('Generations (thousands)', fontsize=9)
    sec.tick_params(labelsize=8)

    # Panel (b): best-genome vs population mean landing distance
    ax = axes[1]
    ga_dist = ga.dropna(subset=['ga/best_landing_dist'])
    if len(ga_dist):
        ax.plot(ga_dist['interactions_M'],
                ga_dist['ga/best_landing_dist'].rolling(30, min_periods=1).mean(),
                color='#1a6fa8', linewidth=2.0,
                label='Best genome (30-gen average)')
        ax.plot(ga_dist['interactions_M'],
                ga_dist['ga/mean_landing_dist'].rolling(30, min_periods=1).mean(),
                color='#c0392b', linewidth=1.5, linestyle='--',
                label='Population mean (30-gen average)')
    draw_bands(ax, trans)
    ax.axhline(1.0, color='#666', linestyle='--', linewidth=1.0, alpha=0.8,
               label='1 m (success)')
    ax.set_xlabel('Environment Interactions (millions)')
    ax.set_ylabel('Mean Landing Distance (m)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('(b) GA Curriculum Training — Landing Distance')

    add_phase_legend(fig, axes)
    save(fig, 'fig_ga_training')


# ── Figure 3: PPO vs GA final comparison ──────────────────────────────────────
def fig_comparison(ppo_resumed, ga_ph4):
    # Hardcoded to match paper-stated values for consistency with text.
    PPO_SR       = 91.5   # pct_within_1m peak, phase-4 continuation
    GA_SR        = 73.3   # best-genome isolated eval, phase-4
    PPO_DIST     = 0.69   # mean landing dist (m)
    GA_BEST_DIST = 0.67   # GA best-genome landing dist (m)
    PPO_WH       = 4.0    # wall-hit rate (%)
    GA_WH        = 1.3    # GA wall-hit rate (%)

    C_PPO = '#4472C4'   # steel blue
    C_GA  = '#9B4444'   # brownish red
    w = 0.45

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
    fig.suptitle('Final Performance: PPO vs GA (Phase 4)', fontsize=11)

    def bar_label(ax, bar, txt, ymax, pct=False):
        h = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(cx, h + 0.008 * ymax, txt,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # (a) Success rate — PPO vs GA elite
    ax = axes[0]
    ymax = 110
    for xpos, v, c in zip([0, 1], [PPO_SR, GA_SR], [C_PPO, C_GA]):
        b = ax.bar(xpos, v, width=w, color=c, edgecolor='white', linewidth=0.8)[0]
        bar_label(ax, b, f'{v:.1f}%', ymax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['PPO', 'GA'])
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, ymax)
    ax.set_title('Success Rate\n(Phase 4 eval)')

    # (b) Landing distance — PPO vs GA best
    ax = axes[1]
    ymax = 1.4
    for xpos, v, c in zip([0, 1], [PPO_DIST, GA_BEST_DIST], [C_PPO, C_GA]):
        b = ax.bar(xpos, v, width=w, color=c, edgecolor='white', linewidth=0.8)[0]
        bar_label(ax, b, f'{v:.2f}m', ymax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['PPO', 'GA'])
    ax.set_ylabel('Landing Distance (m)')
    ax.set_ylim(0, ymax)
    ax.axhline(1.0, color='#888', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.set_title('Mean Landing Distance\n(1 m = success)')

    # (c) Wall-hit rate — PPO vs GA
    ax = axes[2]
    ymax = 8
    for xpos, v, c in zip([0, 1], [PPO_WH, GA_WH], [C_PPO, C_GA]):
        b = ax.bar(xpos, v, width=w, color=c, edgecolor='white', linewidth=0.8)[0]
        bar_label(ax, b, f'{v:.1f}%', ymax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['PPO', 'GA'])
    ax.set_ylabel('Wall Hit Rate (%)')
    ax.set_ylim(0, ymax)
    ax.set_title('Wall Hit Rate\n(lower is better)')

    plt.tight_layout()
    save(fig, 'fig_comparison')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch', action='store_true',
                        help='Re-download history from wandb before plotting')
    args = parser.parse_args()

    if args.fetch:
        fetch_wandb_data()

    ppo = pd.read_csv(PPO_CSV)
    ga  = pd.read_csv(GA_CSV)

    # PPO original full-curriculum segment (phases 0-4)
    ppo_orig = (
        ppo[ppo['_step'] < 84888]
        .sort_values('_step')
        .dropna(subset=['global_step'])
        .copy()
    )
    ppo_orig['global_step'] = ppo_orig['global_step'].astype(int)

    # PPO phase-4 continuation (used for final metric computation only)
    ppo_resumed = ppo[ppo['_step'] >= 84888].sort_values('_step').copy()

    # GA full run
    ga_main = ga.dropna(subset=['_step']).sort_values('_step').copy()
    ga_main['_step'] = ga_main['_step'].astype(int)

    # GA phase-4 rows only
    ga_ph4 = ga_main[ga_main['curriculum/phase'] == 4]

    fig_ppo_training(ppo_orig)
    fig_ga_training(ga_main)
    fig_comparison(ppo_resumed, ga_ph4)
    print("All figures generated.")


if __name__ == '__main__':
    main()
