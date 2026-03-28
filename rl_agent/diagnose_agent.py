"""
diagnose_agent.py
=================
Quick diagnostic: confirms the model gives weather-adapted green times
when queues AND wait times are set to realistic values (as the fixed demo does).
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from dqn_agent import DQNAgent
from traffic_env import STATE_SIZE, GREEN_TIMES, MAX_VEHICLES, MAX_WAIT, LANES

AGENT_PATH = os.path.join('results', 'dqn_traffic.pth')

agent = DQNAgent(state_size=STATE_SIZE, action_size=len(GREEN_TIMES))
agent.load(AGENT_PATH)
agent.epsilon = 0.0
agent.policy_net.eval()

print(f"\nGREEN_TIMES = {GREEN_TIMES}\n")

DEMO_WAIT_PER_VEHICLE = 6.0

def build_state(queues, weather_factor, phase_idx):
    """Build state vector matching traffic_env._get_obs()"""
    feats = []
    for lane in LANES:
        q = queues[lane]
        w = min(q * DEMO_WAIT_PER_VEHICLE, 150.0)
        feats.append(float(np.clip(q / MAX_VEHICLES, 0.0, 1.0)))   # density
        feats.append(float(np.clip(q / MAX_VEHICLES, 0.0, 1.0)))   # queue
        feats.append(float(np.clip(w / MAX_WAIT,     0.0, 1.0)))   # wait
    feats.append(float(weather_factor))
    one_hot = [0.0] * 4
    one_hot[phase_idx] = 1.0
    feats.extend(one_hot)
    return np.array(feats, dtype=np.float32)

print("=" * 65)
print("  REAL DEMO SCENARIOS  (queues + wait times, as live_demo.py sets)")
print("=" * 65)

DEMO_INIT_QUEUES = {
    'clear': {'NORTH': 20, 'SOUTH': 18, 'WEST': 25, 'EAST': 22},
    'rain':  {'NORTH': 28, 'SOUTH': 25, 'WEST': 30, 'EAST': 27},
    'fog':   {'NORTH': 35, 'SOUTH': 30, 'WEST': 38, 'EAST': 33},
}
WF = {'clear': 1.00, 'rain': 0.70, 'fog': 0.55}

print(f"\n  {'Weather':<8} {'Factor':>6}  {'Chosen Green':>14}  {'Q-spread':>9}  {'Q-vals (top 3)'}")
print(f"  {'-'*63}")

for weather in ['clear', 'rain', 'fog']:
    queues = DEMO_INIT_QUEUES[weather]
    wf = WF[weather]

    # Test each phase (which lane is active), collect chosen greens
    chosen_greens = []
    for phase in range(4):
        state = build_state(queues, wf, phase)
        with torch.no_grad():
            q_vals = agent.policy_net(torch.FloatTensor(state).unsqueeze(0)).squeeze().numpy()
        action = int(q_vals.argmax())
        chosen_greens.append(int(GREEN_TIMES[action]))
        if phase == 0:
            spread = q_vals.max() - q_vals.min()
            top3 = sorted(zip(q_vals, GREEN_TIMES), reverse=True)[:3]
            top3_str = ', '.join(f'{int(gt)}s({v:.3f})' for v, gt in top3)
            chosen_phase0 = int(GREEN_TIMES[action])
            spread0 = spread
            top3_str0 = top3_str

    print(f"  {weather.capitalize():<8} {wf:>6.2f}  {chosen_phase0:>12}s  "
          f"{spread0:>9.4f}  {top3_str0}")

print()
print("=" * 65)
print("  PHASE-BY-PHASE BREAKDOWN  (active lane changes each step)")
print("=" * 65)

for weather in ['clear', 'rain', 'fog']:
    queues = DEMO_INIT_QUEUES[weather]
    wf = WF[weather]
    print(f"\n  {weather.upper()} (factor={wf})  "
          f"Queues: N={queues['NORTH']} S={queues['SOUTH']} "
          f"W={queues['WEST']} E={queues['EAST']}")
    print(f"  {'Phase':>6}  {'Active':>7}  {'Green (DQN)':>12}  {'Green (Fixed)':>14}")
    for phase, lane in enumerate(LANES):
        state = build_state(queues, wf, phase)
        with torch.no_grad():
            q_vals = agent.policy_net(torch.FloatTensor(state).unsqueeze(0)).squeeze().numpy()
        action = int(q_vals.argmax())
        chosen = int(GREEN_TIMES[action])
        print(f"  {phase+1:>6}  {lane:>7}  {chosen:>10}s  {'30s':>14}")

print()
print("Expected: Green time should INCREASE clear → rain → fog")
print("Expected: Busier lanes should get LONGER green than lighter lanes")
