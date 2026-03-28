"""
main.py  —  v4
==============
Run from rl_agent/ folder:
    python main.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from traffic_env import TrafficEnv
from dqn_agent   import DQNAgent
from plots       import plot_training_results, plot_weather_performance

NUM_EPISODES  = 1500
MAX_STEPS     = 200   # must match traffic_env MAX_STEPS
SAVE_INTERVAL = 100
RESULTS_DIR   = 'results'
BEST_MODEL    = os.path.join(RESULTS_DIR, 'dqn_traffic.pth')

os.makedirs(RESULTS_DIR, exist_ok=True)


def train():
    print("=" * 58)
    print("  Weather-Aware DQN Traffic Signal Control  v4")
    print("  Soft target update + LR scheduler")
    print("=" * 58)

    env = TrafficEnv(render_mode='none', total_episodes=NUM_EPISODES)

    # Pass total_steps so scheduler knows the full training horizon
    agent = DQNAgent(
        state_size  = env.state_size,
        action_size = env.action_space.n,
        total_steps = NUM_EPISODES * MAX_STEPS,
    )

    ep_rewards  = []
    ep_waits    = []
    ep_losses   = []
    ep_epsilons = []
    weather_rewards = {'clear': [], 'rain': [], 'fog': []}
    best_reward = -np.inf

    print(f"\nTraining {NUM_EPISODES} episodes...\n")

    for ep in range(1, NUM_EPISODES + 1):
        state, info = env.reset()
        weather     = info['weather']

        ep_reward = 0.0
        ep_wait   = 0.0
        losses    = []
        done      = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            ep_reward += reward
            ep_wait   += info.get('total_wait', 0)
            state      = next_state

        agent.decay_epsilon()

        ep_rewards.append(ep_reward)
        ep_waits.append(ep_wait)
        ep_losses.append(np.mean(losses) if losses else 0.0)
        ep_epsilons.append(agent.epsilon)
        weather_rewards[weather].append(ep_reward)

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(BEST_MODEL)

        if ep % SAVE_INTERVAL == 0:
            agent.save(os.path.join(RESULTS_DIR, f'dqn_ep{ep}.pth'))

        if ep % 10 == 0 or ep == 1:
            avg10 = np.mean(ep_rewards[-10:])
            print(f"[Ep {ep:>4}]  {weather:5s}  "
                  f"reward={ep_reward:6.2f}  avg10={avg10:6.2f}  "
                  f"wait={ep_wait/MAX_STEPS:5.1f}s/step  "
                  f"eps={agent.epsilon:.3f}  "
                  f"lr={agent.get_lr():.6f}  "
                  f"loss={ep_losses[-1]:.4f}")

    print(f"\nTraining complete. Best reward: {best_reward:.3f}")
    plot_training_results(ep_rewards, ep_waits, ep_losses, ep_epsilons,
                          save_dir=RESULTS_DIR)
    plot_weather_performance(weather_rewards, save_dir=RESULTS_DIR)
    return agent


if __name__ == '__main__':
    train()