"""
traffic_env.py  —  v5
=====================
Key fix: Weather-scaled queue penalty in reward.

Root cause of weak fog adaptation (confirmed by evaluation):
  - DQN green time: Clear=20.2s, Rain=21.4s, Fog=22.1s  (only +1.9s)
  - Agent should give Clear=20s, Rain=27s, Fog=35s

  The old reward penalised queue equally regardless of weather.
  In fog the queue grows faster (lower throughput) but the penalty
  was the same — so the agent saw no extra incentive to give longer
  green in fog.

Fix: divide queue penalty by weather_factor:
  Clear: penalty / 1.00 = normal
  Rain:  penalty / 0.70 = 1.43× harder
  Fog:   penalty / 0.55 = 1.82× harder

  Now the agent MUST give longer green in fog to avoid the 1.82×
  penalty — it discovers this through reward signals alone.
"""

import numpy as np
import json
import os
import sys
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from weather.weather_classifier import sample_training_weather, get_factor

LANES      = ['NORTH', 'SOUTH', 'WEST', 'EAST']
NUM_LANES  = 4
STATE_SIZE = NUM_LANES * 3 + 1 + NUM_LANES   # 17

GREEN_TIMES = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 75], dtype=np.float32)
NUM_ACTIONS = len(GREEN_TIMES)

MAX_VEHICLES    = 50
MAX_WAIT        = 300
MAX_STEPS       = 200
BASE_THROUGHPUT = 0.8
BASE_ARRIVAL    = 0.25

COUNTS_JSON = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'yolo_detection', 'counts_output.json'
))


class TrafficEnv(gym.Env):
    metadata = {'render_modes': ['human', 'none']}

    def __init__(self, render_mode='none', total_episodes=1000):
        super().__init__()
        self.render_mode     = render_mode
        self.total_episodes  = total_episodes
        self.current_episode = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._real_data = self._load_real_data()
        if self._real_data:
            print(f"[TrafficEnv] Loaded {len(self._real_data)} real frames.")
        else:
            print("[TrafficEnv] No real data — using synthetic init.")

        self.weather          = 'clear'
        self.weather_factor   = 1.0
        self.lane_states      = None
        self.phase_idx        = 0
        self.step_count       = 0
        self._prev_total_wait = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_episode += 1
        self.step_count       = 0
        self.phase_idx        = 0

        self.weather        = sample_training_weather(
            self.current_episode, self.total_episodes)
        self.weather_factor = get_factor(self.weather)
        self.lane_states    = self._init_lane_states()
        self._prev_total_wait = 0.0

        return self._get_obs(), {
            'weather': self.weather, 'weather_factor': self.weather_factor}

    def step(self, action: int):
        green_duration = float(GREEN_TIMES[action])
        active_lane    = LANES[self.phase_idx]
        total_through  = 0.0

        # Save active lane queue BEFORE processing (for empty lane penalty)
        active_q_before = self.lane_states[active_lane]['queue']

        for lane in LANES:
            s = self.lane_states[lane]
            arrivals   = np.random.poisson(BASE_ARRIVAL * green_duration)
            s['queue'] = min(s['queue'] + arrivals, MAX_VEHICLES)

            if lane == active_lane:
                effective_tp = BASE_THROUGHPUT * self.weather_factor
                departures   = min(s['queue'], int(effective_tp * green_duration))
                s['queue']   = max(0, s['queue'] - departures)
                s['wait']    = 0.0 if s['queue'] == 0 else \
                    max(0.0, s['wait'] * (s['queue'] / MAX_VEHICLES))
                total_through += departures
            else:
                if s['queue'] > 0:
                    density = s['queue'] / MAX_VEHICLES
                    s['wait'] = min(s['wait'] + (green_duration * 0.5) * density, MAX_WAIT)

            s['density'] = s['queue'] / MAX_VEHICLES

        current_wait = sum(s['wait'] for s in self.lane_states.values())
        total_queue  = sum(s['queue'] for s in self.lane_states.values())
        max_lane_q   = max(s['queue'] for s in self.lane_states.values())

        delta_wait   = self._prev_total_wait - current_wait
        norm_delta   = delta_wait   / (NUM_LANES * MAX_WAIT)
        norm_queue   = total_queue  / (NUM_LANES * MAX_VEHICLES)

        # --- Throughput EFFICIENCY (vehicles per second of green) ---
        # Rewards clearing vehicles quickly, not just clearing them.
        # 10 vehicles in 15s is better than 10 vehicles in 75s.
        throughput_rate       = total_through / max(green_duration, 1.0)
        max_possible_rate     = BASE_THROUGHPUT * self.weather_factor
        norm_throughput_eff   = min(throughput_rate / max(max_possible_rate, 0.01), 1.0)

        # --- Weather-scaled queue penalty (same principle as before) ---
        weather_scaled_queue_penalty = (0.2 * norm_queue) / self.weather_factor

        # --- Empty lane penalty ---
        # If active lane had ≤2 vehicles and agent gave ≥30s green,
        # penalise the wasted green time. Forces short green on empty lanes.
        empty_lane_penalty = 0.0
        if active_q_before <= 2 and green_duration >= 30:
            excess = (green_duration - 15.0) / GREEN_TIMES[-1]
            empty_lane_penalty = 0.15 * excess

        # --- Queue imbalance penalty ---
        # Penalise having one very congested lane (encourages fairness)
        max_lane_norm = max_lane_q / MAX_VEHICLES

        raw_reward = (
            0.4 * norm_delta
          + 0.3 * norm_throughput_eff
          - weather_scaled_queue_penalty
          - empty_lane_penalty
          - 0.1 * max_lane_norm
        )

        reward = float(np.clip(raw_reward, -1.0, 1.0))

        self._prev_total_wait = current_wait
        self.phase_idx  = (self.phase_idx + 1) % NUM_LANES
        self.step_count += 1

        return self._get_obs(), reward, self.step_count >= MAX_STEPS, False, {
            'weather': self.weather, 'weather_factor': self.weather_factor,
            'active_lane': active_lane, 'green_duration': green_duration,
            'total_wait': current_wait, 'throughput': total_through,
        }

    def _get_obs(self) -> np.ndarray:
        feats = []
        for lane in LANES:
            s = self.lane_states[lane]
            feats.append(np.clip(s['density'],              0.0, 1.0))
            feats.append(np.clip(s['queue'] / MAX_VEHICLES, 0.0, 1.0))
            feats.append(np.clip(s['wait']  / MAX_WAIT,     0.0, 1.0))
        feats.append(float(self.weather_factor))
        one_hot = [0.0] * NUM_LANES
        one_hot[self.phase_idx] = 1.0
        feats.extend(one_hot)
        return np.array(feats, dtype=np.float32)

    def _init_lane_states(self) -> dict:
        if self._real_data:
            rec = self._real_data[np.random.randint(len(self._real_data))]
            return {
                lane: {
                    'queue':   int(rec['lanes'].get(lane, 0)),
                    'density': rec['lanes'].get(lane, 0) / MAX_VEHICLES,
                    'wait':    0.0,
                } for lane in LANES
            }
        return {
            lane: {
                'queue':   int(np.random.uniform(0, MAX_VEHICLES * 0.5)),
                'density': np.random.uniform(0.0, 0.5),
                'wait':    0.0,
            } for lane in LANES
        }

    def _load_real_data(self) -> list:
        if not os.path.exists(COUNTS_JSON):
            return []
        try:
            with open(COUNTS_JSON) as f:
                data = json.load(f)
            return data if data and 'lanes' in data[0] else []
        except Exception:
            return []

    def set_weather(self, label: str):
        self.weather        = label
        self.weather_factor = get_factor(label)

    def set_lane_counts(self, counts: dict):
        for lane in LANES:
            if lane in counts and self.lane_states:
                q = int(counts[lane])
                self.lane_states[lane]['queue']   = q
                self.lane_states[lane]['density'] = q / MAX_VEHICLES

    def render(self):
        if self.render_mode == 'human':
            print(f"\n[Step {self.step_count}] {self.weather} "
                  f"(f={self.weather_factor:.2f}) Active={LANES[self.phase_idx]}")
            for lane in LANES:
                s = self.lane_states[lane]
                print(f"  {lane}: q={s['queue']} d={s['density']:.2f} "
                      f"w={s['wait']:.1f}s")

    @property
    def state_size(self): return STATE_SIZE

    @property
    def green_times(self): return GREEN_TIMES