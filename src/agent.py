"""
QLearningAgent - tabular Q-learning with epsilon-greedy exploration.

Q-table size: 8 stock buckets x 7 weekdays x 3 trend levels = 168 states,
              168 states x 4 actions = 672 numbers.
"""

import numpy as np

from utils import load_json, save_json

N_STOCK, N_DOW, N_TREND = 8, 7, 3


class QLearningAgent:
    def __init__(self, n_actions: int, alpha: float, gamma: float,
                 epsilon: float, seed: int = 0):
        self.n_actions = n_actions
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration probability
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((N_STOCK, N_DOW, N_TREND, n_actions))

    def act(self, state: tuple) -> int:
        """Epsilon-greedy: explore with probability epsilon, otherwise pick the best action."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s_next, done) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]"""
        future = 0.0 if done else np.max(self.q[s_next])
        target = r + self.gamma * future
        self.q[s][a] += self.alpha * (target - self.q[s][a])

    # ---- save / load as readable JSON ---------------------------------- #
    def save(self, path, actions) -> None:
        table = {}
        for s in np.ndindex(N_STOCK, N_DOW, N_TREND):
            key = f"stock{s[0]}_dow{s[1]}_trend{s[2]}"
            table[key] = [round(float(v), 3) for v in self.q[s]]
        save_json({"actions": actions, "q_table": table}, path)

    def load(self, path) -> None:
        table = load_json(path)["q_table"]
        for s in np.ndindex(N_STOCK, N_DOW, N_TREND):
            self.q[s] = table[f"stock{s[0]}_dow{s[1]}_trend{s[2]}"]
