"""
Quick sanity checks. Run after `dvc repro prepare`:
    python tests/smoke_test.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import QLearningAgent          # noqa: E402
from env import InventoryEnv              # noqa: E402
from utils import ROOT, load_params       # noqa: E402


def main() -> None:
    params = load_params()
    ep = params["env"]
    processed = ROOT / "data" / "processed"

    # 1. prepare produced the three splits, in time order, without gaps in the columns
    splits = {name: pd.read_parquet(processed / f"{name}.parquet")
              for name in ("train", "stream", "test")}
    for name, df in splits.items():
        assert len(df) > 0, f"{name} split is empty"
        assert df.isna().sum().sum() == 0, f"{name} has missing values"
        assert set(df["demand_trend"].unique()) <= {0, 1, 2}
    assert splits["train"]["date"].max() < splits["stream"]["date"].min()
    assert splits["stream"]["date"].max() < splits["test"]["date"].min()
    print("OK  splits:", {k: len(v) for k, v in splits.items()})

    # 2. the environment runs a full episode and states stay inside the Q-table
    env = InventoryEnv(splits["train"], ep)
    agent = QLearningAgent(len(ep["actions"]), 0.1, 0.95, 1.0, seed=0)
    state, done, steps = env.reset(0, ep["episode_length"]), False, 0
    while not done:
        assert 0 <= state[0] < 8 and 0 <= state[1] < 7 and 0 <= state[2] < 3
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        assert 0 <= env.stock <= ep["capacity"]
        state, steps = next_state, steps + 1
    assert steps == ep["episode_length"]
    print("OK  environment episode:", steps, "days")

    # 3. the Q-update actually changed the table
    assert agent.q.any(), "Q-table is still all zeros"
    print("OK  Q-table updated")
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
