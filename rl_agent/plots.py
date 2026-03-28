"""
plots.py
========
Training visualisation for the DQN traffic signal controller.

Functions:
  plot_training_results()   - 4-panel chart: reward, wait, loss, epsilon
  plot_weather_performance()- bar chart: avg reward per weather condition
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth(x, window: int = 20):
    if len(x) < window:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode='valid')


# ── Weather colours (consistent across all plots) ────────────────────────────

WEATHER_STYLE = {
    'clear': {'color': '#4CAF50', 'label': 'Clear'},
    'rain':  {'color': '#2196F3', 'label': 'Rain'},
    'fog':   {'color': '#9E9E9E', 'label': 'Fog'},
}


# ── Main training plot ────────────────────────────────────────────────────────

def plot_training_results(rewards, waits, losses, epsilons,
                          save_dir: str = 'results'):
    """
    4-panel training metrics chart.

    Panels:
      1. Episode reward (raw + 20-ep smoothed)
      2. Total waiting time per episode
      3. Training loss (Huber)
      4. Epsilon decay curve
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        'DQN Traffic Signal Control — Training Metrics',
        fontsize=14, fontweight='bold', y=1.01
    )

    configs = [
        (axes[0, 0], rewards,  'Episode Reward',             'Total Reward',    '#1565C0'),
        (axes[0, 1], waits,    'Total Waiting Time/Episode', 'Seconds',         '#C62828'),
        (axes[1, 0], losses,   'Training Loss (Huber)',       'Loss',            '#2E7D32'),
        (axes[1, 1], epsilons, 'Exploration Rate (Epsilon)',  'Epsilon',         '#6A1B9A'),
    ]

    for ax, data, title, ylabel, color in configs:
        x_raw = np.arange(1, len(data) + 1)
        ax.plot(x_raw, data, alpha=0.2, color=color)

        s = smooth(data)
        x_smooth = np.linspace(1, len(data), len(s))
        ax.plot(x_smooth, s, color=color, linewidth=2, label='20-ep avg')

        # Last-50 mean reference line
        last50 = np.mean(data[-50:]) if len(data) >= 50 else np.mean(data)
        ax.axhline(last50, color='black', linestyle='--', linewidth=1,
                   label=f'Last-50 mean: {last50:.2f}')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved -> {path}")
    plt.show()


# ── Weather performance plot ──────────────────────────────────────────────────

def plot_weather_performance(weather_rewards: dict, save_dir: str = 'results'):
    """
    Bar chart comparing DQN performance across weather conditions.

    weather_rewards: {'clear': [r1,r2,...], 'rain': [...], 'fog': [...]}
    Shows mean ± std for each weather condition.
    """
    weathers = [w for w in ['clear', 'rain', 'fog'] if weather_rewards.get(w)]
    if not weathers:
        return

    means  = [np.mean(weather_rewards[w]) for w in weathers]
    stds   = [np.std(weather_rewards[w])  for w in weathers]
    colors = [WEATHER_STYLE[w]['color']   for w in weathers]
    labels = [WEATHER_STYLE[w]['label']   for w in weathers]
    counts = [len(weather_rewards[w])     for w in weathers]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6,
                  color=colors, edgecolor='white', linewidth=1.5,
                  error_kw={'elinewidth': 2, 'ecolor': '#333333'})

    # Value labels on bars
    for bar, m, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f'{m:.2f}\n(n={n})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('DQN Performance by Weather Condition',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Weather Condition', fontsize=10)
    ax.set_ylabel('Average Episode Reward', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add factor annotation
    factors = {'clear': 1.00, 'rain': 0.70, 'fog': 0.55}
    for i, w in enumerate(weathers):
        ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,
                f'factor={factors[w]}',
                ha='center', va='top', fontsize=8, color='#555555')

    plt.tight_layout()
    path = os.path.join(save_dir, 'weather_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved -> {path}")
    plt.show()


# ── Green time distribution (optional diagnostic) ────────────────────────────

def plot_green_time_distribution(green_times_chosen: list,
                                 weather_labels: list,
                                 save_dir: str = 'results'):
    """
    Histogram of green time choices, coloured by weather.
    Useful to verify: agent gives longer green in rain/fog.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle('Green Time Distribution per Weather Condition',
                 fontsize=12, fontweight='bold')

    for ax, weather in zip(axes, ['clear', 'rain', 'fog']):
        times = [t for t, w in zip(green_times_chosen, weather_labels)
                 if w == weather]
        if not times:
            ax.set_title(f'{weather.capitalize()} (no data)')
            continue

        color = WEATHER_STYLE[weather]['color']
        ax.hist(times, bins=10, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(times), color='black', linestyle='--',
                   linewidth=1.5, label=f'mean={np.mean(times):.1f}s')
        ax.set_title(f'{weather.capitalize()}  (n={len(times)})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Green Duration (s)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, 'green_time_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved -> {path}")
    plt.show()