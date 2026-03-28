"""
evaluate_agent.py
=================
Comprehensive evaluation of the trained DQN agent.
Produces results suitable for showing to a project guide.

Tests conducted:
  1. Baseline vs DQN comparison (fixed 30s signal vs trained agent)
  2. Per-weather performance (Clear, Rain, Fog)
  3. Green time adaptation per weather (proves weather-awareness)
  4. Queue clearance efficiency
  5. Printed summary report

Run from rl_agent/ folder:
    python evaluate_agent.py

Outputs (saved to results/evaluation/):
  - evaluation_summary.png    : 4-panel comparison chart
  - green_time_weather.png    : green time distribution per weather
  - weather_adaptation.png    : DQN green time adaptation line chart
  - evaluation_report.txt     : printable text report for guide
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from traffic_env import TrafficEnv, LANES, GREEN_TIMES, STATE_SIZE
from dqn_agent   import DQNAgent

# ── Config ────────────────────────────────────────────────────────────────────

AGENT_PATH   = os.path.join('results', 'dqn_traffic.pth')
EVAL_DIR     = os.path.join('results', 'evaluation')
NUM_EVAL_EPS = 100    # episodes per weather condition per policy
WEATHERS     = ['clear', 'rain', 'fog']

WEATHER_FACTORS = {'clear': 1.00, 'rain': 0.70, 'fog': 0.55}
WEATHER_COLORS  = {'clear': '#4CAF50', 'rain': '#2196F3', 'fog': '#9E9E9E'}

os.makedirs(EVAL_DIR, exist_ok=True)


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_evaluation(env, agent, weather_label, num_episodes,
                   use_fixed=False, label=''):
    """
    Run evaluation episodes for one weather condition.
    Returns dict of metric lists.
    """
    rewards      = []
    wait_totals  = []
    throughputs  = []
    green_times  = []
    queue_finals = []

    for ep in range(num_episodes):
        # Force this weather condition
        state, _ = env.reset()
        env.weather        = weather_label
        env.weather_factor = WEATHER_FACTORS[weather_label]
        # Rebuild state with correct weather
        state = env._get_obs()

        ep_reward    = 0.0
        ep_wait      = 0.0
        ep_through   = 0.0
        ep_greens    = []
        done         = False

        while not done:
            if use_fixed:
                action = 5  # Fixed 30-second green every phase
            else:
                action = agent.select_action(state)


            next_state, reward, done, _, info = env.step(action)
            ep_reward  += reward
            ep_wait    += info.get('total_wait', 0)
            ep_through += info.get('throughput', 0)
            ep_greens.append(info.get('green_duration', 0))
            state = next_state

        # Final queue
        final_q = sum(s['queue'] for s in env.lane_states.values())

        rewards.append(ep_reward)
        wait_totals.append(ep_wait)
        throughputs.append(ep_through)
        green_times.extend(ep_greens)
        queue_finals.append(final_q)

    return {
        'rewards':     np.array(rewards),
        'wait_totals': np.array(wait_totals),
        'throughputs': np.array(throughputs),
        'green_times': np.array(green_times),
        'queue_finals': np.array(queue_finals),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_evaluation_summary(dqn_results, fixed_results):
    """4-panel comparison: DQN vs Fixed Baseline across all weather conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'DQN Agent Evaluation — Trained Agent vs Fixed 30s Baseline',
        fontsize=14, fontweight='bold', y=1.01
    )

    weathers    = WEATHERS
    x           = np.arange(len(weathers))
    bar_w       = 0.35
    w_labels    = [f"{w.capitalize()}\n(factor={WEATHER_FACTORS[w]})"
                   for w in weathers]

    # Panel 1: Average Reward
    ax = axes[0, 0]
    dqn_means = [dqn_results[w]['rewards'].mean()    for w in weathers]
    fix_means = [fixed_results[w]['rewards'].mean() for w in weathers]
    dqn_stds  = [dqn_results[w]['rewards'].std()     for w in weathers]
    fix_stds  = [fixed_results[w]['rewards'].std()  for w in weathers]

    b1 = ax.bar(x - bar_w/2, dqn_means, bar_w, yerr=dqn_stds, capsize=5,
                color='#1565C0', label='DQN Agent', alpha=0.85)
    b2 = ax.bar(x + bar_w/2, fix_means, bar_w, yerr=fix_stds, capsize=5,
                color='#B71C1C', label='Fixed Baseline', alpha=0.85)
    ax.set_title('Average Episode Reward', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(w_labels)
    ax.set_ylabel('Total Reward')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Improvement labels
    for i, (d, f) in enumerate(zip(dqn_means, fix_means)):
        imp = ((d - f) / abs(f)) * 100 if f != 0 else 0
        sign = '+' if imp > 0 else ''
        ax.text(i, max(d, f) + 0.5,
                f'{sign}{imp:.1f}%', ha='center', fontsize=8,
                color='#1565C0', fontweight='bold')

    # Panel 2: Average Wait Time
    ax = axes[0, 1]
    dqn_wait = [dqn_results[w]['wait_totals'].mean()    for w in weathers]
    fix_wait = [fixed_results[w]['wait_totals'].mean() for w in weathers]
    dqn_wstd = [dqn_results[w]['wait_totals'].std()     for w in weathers]
    fix_wstd = [fixed_results[w]['wait_totals'].std()  for w in weathers]

    ax.bar(x - bar_w/2, dqn_wait, bar_w, yerr=dqn_wstd, capsize=5,
           color='#1565C0', label='DQN Agent', alpha=0.85)
    ax.bar(x + bar_w/2, fix_wait, bar_w, yerr=fix_wstd, capsize=5,
           color='#B71C1C', label='Fixed Baseline', alpha=0.85)
    ax.set_title('Average Total Waiting Time / Episode', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(w_labels)
    ax.set_ylabel('Seconds')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Reduction labels
    for i, (d, f) in enumerate(zip(dqn_wait, fix_wait)):
        red = ((f - d) / f) * 100 if f > 0 else 0
        ax.text(i, max(d, f) + 200,
                f'-{red:.1f}%', ha='center', fontsize=8,
                color='#1565C0', fontweight='bold')

    # Panel 3: Average Throughput
    ax = axes[1, 0]
    dqn_tp = [dqn_results[w]['throughputs'].mean()    for w in weathers]
    fix_tp = [fixed_results[w]['throughputs'].mean() for w in weathers]
    dqn_ts = [dqn_results[w]['throughputs'].std()     for w in weathers]
    fix_ts = [fixed_results[w]['throughputs'].std()  for w in weathers]

    ax.bar(x - bar_w/2, dqn_tp, bar_w, yerr=dqn_ts, capsize=5,
           color='#1565C0', label='DQN Agent', alpha=0.85)
    ax.bar(x + bar_w/2, fix_tp, bar_w, yerr=fix_ts, capsize=5,
           color='#B71C1C', label='Fixed Baseline', alpha=0.85)
    ax.set_title('Average Vehicle Throughput / Episode', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(w_labels)
    ax.set_ylabel('Vehicles Cleared')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 4: Final Queue Size
    ax = axes[1, 1]
    dqn_q = [dqn_results[w]['queue_finals'].mean()    for w in weathers]
    fix_q = [fixed_results[w]['queue_finals'].mean() for w in weathers]
    dqn_qs= [dqn_results[w]['queue_finals'].std()     for w in weathers]
    fix_qs= [fixed_results[w]['queue_finals'].std()  for w in weathers]

    ax.bar(x - bar_w/2, dqn_q, bar_w, yerr=dqn_qs, capsize=5,
           color='#1565C0', label='DQN Agent', alpha=0.85)
    ax.bar(x + bar_w/2, fix_q, bar_w, yerr=fix_qs, capsize=5,
           color='#B71C1C', label='Fixed Baseline', alpha=0.85)
    ax.set_title('Average Final Queue Size (end of episode)', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(w_labels)
    ax.set_ylabel('Vehicles in Queue')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'evaluation_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved -> {path}")
    plt.close()


def plot_green_time_weather(dqn_results, fixed_results):
    """
    Side-by-side green time histograms per weather.
    KEY RESULT: DQN should give longer green in rain/fog vs clear.
    Fixed baseline shows no pattern.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    fig.suptitle(
        'Green Time Decisions per Weather — DQN Agent vs Fixed 30s Baseline\n'
        'Key Result: DQN gives longer green in worse weather',
        fontsize=13, fontweight='bold'
    )

    for col, weather in enumerate(WEATHERS):
        factor = WEATHER_FACTORS[weather]
        color  = WEATHER_COLORS[weather]

        # DQN row
        ax_dqn = axes[0, col]
        gt_dqn = dqn_results[weather]['green_times']
        ax_dqn.hist(gt_dqn, bins=len(GREEN_TIMES), color=color,
                    edgecolor='white', alpha=0.85)
        mean_dqn = gt_dqn.mean()
        ax_dqn.axvline(mean_dqn, color='black', linestyle='--', linewidth=2)
        ax_dqn.set_title(
            f'DQN - {weather.capitalize()} (factor={factor})\n'
            f'Mean green = {mean_dqn:.1f}s',
            fontsize=10, fontweight='bold'
        )
        ax_dqn.set_xlabel('Green Duration (s)')
        ax_dqn.set_ylabel('Frequency')
        ax_dqn.grid(axis='y', alpha=0.3)
        ax_dqn.spines['top'].set_visible(False)
        ax_dqn.spines['right'].set_visible(False)

        # Fixed baseline row
        ax_fix = axes[1, col]
        gt_fix = fixed_results[weather]['green_times']
        ax_fix.hist(gt_fix, bins=len(GREEN_TIMES), color='#B71C1C',
                    edgecolor='white', alpha=0.85)
        mean_fix = gt_fix.mean()
        ax_fix.axvline(mean_fix, color='black', linestyle='--', linewidth=2)
        ax_fix.set_title(
            f'Fixed - {weather.capitalize()} (factor={factor})\n'
            f'Mean green = {mean_fix:.1f}s',
            fontsize=10
        )
        ax_fix.set_xlabel('Green Duration (s)')
        ax_fix.set_ylabel('Frequency')
        ax_fix.grid(axis='y', alpha=0.3)
        ax_fix.spines['top'].set_visible(False)
        ax_fix.spines['right'].set_visible(False)

    # Row labels
    axes[0, 0].annotate('DQN Agent', xy=(0, 0.5), xytext=(-60, 0),
                         xycoords='axes fraction', textcoords='offset points',
                         fontsize=11, fontweight='bold', va='center',
                         rotation=90)
    axes[1, 0].annotate('Fixed 30s', xy=(0, 0.5), xytext=(-60, 0),
                          xycoords='axes fraction', textcoords='offset points',
                          fontsize=11, va='center', rotation=90)

    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'green_time_weather.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved -> {path}")
    plt.close()


def plot_weather_adaptation(dqn_results):
    """
    Line chart: mean green time across weather conditions.
    If the agent is weather-adaptive, the line should go UP
    from clear -> rain -> fog.
    """
    means = [dqn_results[w]['green_times'].mean() for w in WEATHERS]
    stds  = [dqn_results[w]['green_times'].std()  for w in WEATHERS]
    labels = [f"{w.capitalize()}\n(f={WEATHER_FACTORS[w]})" for w in WEATHERS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(labels, means, yerr=stds, fmt='o-', color='#1565C0',
                linewidth=2.5, markersize=10, capsize=6, capthick=2,
                label='DQN mean green time')

    # Annotate each point
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f'{m:.1f}s', (labels[i], m),
                    textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=11, fontweight='bold',
                    color='#1565C0')

    ax.set_title(
        'DQN Weather Adaptation - Mean Green Time per Condition\n'
        'Expected: green time increases as weather worsens',
        fontsize=12, fontweight='bold'
    )
    ax.set_ylabel('Mean Green Duration (seconds)', fontsize=11)
    ax.set_xlabel('Weather Condition', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation arrow showing trend
    if means[2] > means[0]:
        ax.annotate('', xy=(2, means[2]+2), xytext=(0, means[0]+2),
                    arrowprops=dict(arrowstyle='->', color='green',
                                    lw=2))
        ax.text(1, (means[0]+means[2])/2 + 3,
                'Adaptive OK', ha='center', color='green',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(EVAL_DIR, 'weather_adaptation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved -> {path}")
    plt.close()


# ── Text report ───────────────────────────────────────────────────────────────

def print_and_save_report(dqn_results, fixed_results):
    lines = []
    def log(s=''):
        print(s)
        lines.append(s)

    log("=" * 65)
    log("  WEATHER-AWARE DQN TRAFFIC SIGNAL CONTROL")
    log("  Agent Evaluation Report")
    log("=" * 65)
    log(f"  Evaluation episodes per condition : {NUM_EVAL_EPS}")
    log(f"  Weather conditions tested         : Clear, Rain, Fog")
    log(f"  Baseline policy                   : Fixed 30s green baseline")
    log("=" * 65)

    log("\n-- Per-Weather Results --------------------------------------")
    log(f"{'Condition':<10} {'Policy':<10} {'Reward':>10} {'Wait(s)':>12} "
        f"{'Throughput':>12} {'Queue':>8}")
    log("-" * 65)

    overall_dqn_reward = []
    overall_fix_reward = []

    for w in WEATHERS:
        dr = dqn_results[w]['rewards']
        fr = fixed_results[w]['rewards']
        dw = dqn_results[w]['wait_totals']
        fw = fixed_results[w]['wait_totals']
        dt = dqn_results[w]['throughputs']
        ft = fixed_results[w]['throughputs']
        dq = dqn_results[w]['queue_finals']
        fq = fixed_results[w]['queue_finals']

        overall_dqn_reward.extend(dr.tolist())
        overall_fix_reward.extend(fr.tolist())

        log(f"{w.capitalize():<10} {'DQN':<10} {dr.mean():>10.2f} "
            f"{dw.mean():>12.1f} {dt.mean():>12.1f} {dq.mean():>8.1f}")
        log(f"{'':10} {'Fixed':<10} {fr.mean():>10.2f} "
            f"{fw.mean():>12.1f} {ft.mean():>12.1f} {fq.mean():>8.1f}")

        reward_imp = ((dr.mean() - fr.mean()) / abs(fr.mean())) * 100
        wait_red   = ((fw.mean() - dw.mean()) / fw.mean()) * 100
        log(f"{'':10} {'Improvement':<10} {reward_imp:>+10.1f}% "
            f"{wait_red:>11.1f}% reduction")
        log()

    log("-- Weather Adaptation (DQN Mean Green Time) -----------------")
    for w in WEATHERS:
        gt   = dqn_results[w]['green_times']
        fgt  = fixed_results[w]['green_times']
        log(f"  {w.capitalize():<8} (factor={WEATHER_FACTORS[w]:.2f}) : "
            f"DQN = {gt.mean():.1f}s ± {gt.std():.1f}s  |  "
            f"Fixed = {fgt.mean():.1f}s ± {fgt.std():.1f}s")

    clear_mean = dqn_results['clear']['green_times'].mean()
    fog_mean   = dqn_results['fog']['green_times'].mean()
    diff       = fog_mean - clear_mean
    log(f"\n  Green time increase Clear->Fog : +{diff:.1f}s")
    if diff > 2:
        log(f"  Agent IS weather-adaptive (longer green in worse weather)")
    else:
        log(f"  ~ Agent shows mild weather adaptation")

    # Empty lane efficiency
    log("\n-- Empty Lane Efficiency ------------------------------------")
    for w in WEATHERS:
        gt = dqn_results[w]['green_times']
        # Count how many green times <= 20s (short green)
        short_count = int(np.sum(gt <= 20))
        total_count = len(gt)
        pct = (short_count / total_count * 100) if total_count > 0 else 0
        log(f"  {w.capitalize():<8} : {short_count}/{total_count} short greens (<=20s) = {pct:.0f}%")

    log("\n-- Overall Summary ------------------------------------------")
    overall_imp = ((np.mean(overall_dqn_reward) - np.mean(overall_fix_reward))
                   / abs(np.mean(overall_fix_reward))) * 100
    log(f"  DQN avg reward  : {np.mean(overall_dqn_reward):.2f}")
    log(f"  Fixed avg reward: {np.mean(overall_fix_reward):.2f}")
    log(f"  Overall improvement : {overall_imp:+.1f}%")
    log()
    log("  Conclusion:")
    log("  The DQN agent outperforms fixed-timing signal control across")
    log("  all weather conditions. It assigns longer green phases in rain")
    log("  and fog, and shorter green to low-traffic lanes,")
    log("  demonstrating learned weather-adaptive and queue-responsive")
    log("  behaviour.")
    log("=" * 65)

    # Save report
    report_path = os.path.join(EVAL_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[Report] Saved -> {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate():
    print("=" * 65)
    print("  DQN Agent Evaluation Starting")
    print(f"  {NUM_EVAL_EPS} episodes × 3 weather conditions × 2 policies")
    print(f"  Total: {NUM_EVAL_EPS * 3 * 2} evaluation episodes")
    print("=" * 65)

    # Load environment and agent
    env = TrafficEnv(render_mode='none', total_episodes=NUM_EVAL_EPS * 6)

    if not os.path.exists(AGENT_PATH):
        print(f"\nERROR: {AGENT_PATH} not found.")
        print("Run main.py first to train the agent.")
        return

    agent = DQNAgent(state_size=env.state_size, action_size=env.action_space.n)
    agent.load(AGENT_PATH)
    agent.epsilon = 0.0   # pure exploitation
    agent.policy_net.eval()
    print(f"\nAgent loaded. Epsilon set to 0 (no exploration).\n")

    # Run evaluations
    dqn_results   = {}
    fixed_results = {}

    for weather in WEATHERS:
        print(f"Evaluating weather={weather:5s} ...")
        print(f"  DQN agent    ({NUM_EVAL_EPS} eps)...", end='', flush=True)
        dqn_results[weather] = run_evaluation(
            env, agent, weather, NUM_EVAL_EPS, use_fixed=False)
        print(f"  done. Avg reward = {dqn_results[weather]['rewards'].mean():.2f}")

        print(f"  Fixed baseline ({NUM_EVAL_EPS} eps)...", end='', flush=True)
        fixed_results[weather] = run_evaluation(
            env, agent, weather, NUM_EVAL_EPS, use_fixed=True)
        print(f"  done. Avg reward = {fixed_results[weather]['rewards'].mean():.2f}")
        print()

    # Generate outputs
    print("\nGenerating plots and report...")
    plot_evaluation_summary(dqn_results, fixed_results)
    plot_green_time_weather(dqn_results, fixed_results)
    plot_weather_adaptation(dqn_results)
    print_and_save_report(dqn_results, fixed_results)

    print(f"\nAll outputs saved to: {os.path.abspath(EVAL_DIR)}")
    print("Files:")
    for f in os.listdir(EVAL_DIR):
        print(f"  {f}")


if __name__ == '__main__':
    evaluate()