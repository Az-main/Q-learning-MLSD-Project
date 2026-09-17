"""
Stage 2 - train
---------------
Trains the Q-learning agent on random 60-day windows of the TRAIN split.

Outputs:
    models/q_table.json                  -> the learned policy
    metrics/train.json                   -> summary of training
    metrics/plots/training_curve.csv     -> reward per episode (shown by `dvc plots show`)
"""

import numpy as np
import pandas as pd

from agent import QLearningAgent
from env import InventoryEnv
from utils import ROOT, load_params, save_json


def main() -> None:
    params = load_params()
    ep, tp = params["env"], params["train"]

    data = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    env = InventoryEnv(data, ep)
    agent = QLearningAgent(len(ep["actions"]), tp["alpha"], tp["gamma"],
                           tp["epsilon_start"], seed=tp["seed"])
    rng = np.random.default_rng(tp["seed"])

    history = []
    for episode in range(tp["episodes"]):
        # linear epsilon decay: explore a lot early, exploit later
        frac = episode / max(1, tp["episodes"] - 1)
        agent.epsilon = tp["epsilon_start"] + frac * (tp["epsilon_end"] - tp["epsilon_start"])

        start = int(rng.integers(0, env.n_days - ep["episode_length"]))
        state = env.reset(start, ep["episode_length"])
        total, done = 0.0, False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state, total = next_state, total + reward
        history.append({"episode": episode, "total_reward": round(total, 2)})

    # ---- save outputs --------------------------------------------------- #
    agent.save(ROOT / "models" / "q_table.json", ep["actions"])

    curve = pd.DataFrame(history)
    plots = ROOT / "metrics" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    curve.to_csv(plots / "training_curve.csv", index=False)

    summary = {
        "episodes": tp["episodes"],
        "gamma": tp["gamma"],
        "avg_reward_first_50": round(float(curve["total_reward"].head(50).mean()), 2),
        "avg_reward_last_50": round(float(curve["total_reward"].tail(50).mean()), 2),
    }
    save_json(summary, ROOT / "metrics" / "train.json")
    print("train done:", summary)


if __name__ == "__main__":
    main()
