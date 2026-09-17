"""
InventoryEnv - a tiny simulator built on top of the real demand series.

One step = one day:
    1. The agent sees the state  (stock bucket, day_of_week, demand_trend)
    2. It chooses how much to order  -> pays order_cost for it TODAY
    3. Today's real demand (from the data) is served from the current stock
    4. Leftover stock pays holding_cost, unmet demand pays stockout_penalty
    5. The order arrives overnight -> it is on the shelf TOMORROW

Because the order helps only tomorrow but costs money today, a myopic agent
(gamma = 0) never orders and keeps running out of stock. A far-sighted agent
(gamma = 0.95) learns that paying today prevents stock-outs later.
"""

import numpy as np
import pandas as pd


class InventoryEnv:
    def __init__(self, data: pd.DataFrame, env_params: dict):
        self.demand = data["demand"].to_numpy()
        self.dow = data["day_of_week"].to_numpy()
        self.trend = data["demand_trend"].to_numpy()
        self.p = env_params
        self.actions = env_params["actions"]
        self.n_days = len(data)

    # ------------------------------------------------------------------ #
    def state(self) -> tuple:
        stock_bucket = int(np.digitize(self.stock, self.p["stock_bins"]))   # 0..7
        return (stock_bucket, int(self.dow[self.t]), int(self.trend[self.t]))

    def reset(self, start: int = 0, length: int | None = None) -> tuple:
        """Start an episode at day `start`, lasting `length` days (default: to the end)."""
        self.t = start
        self.end = self.n_days if length is None else min(start + length, self.n_days)
        self.stock = self.p["initial_stock"]
        return self.state()

    def step(self, action_index: int):
        p = self.p
        order = self.actions[action_index]
        demand = self.demand[self.t]

        sold = min(self.stock, demand)
        unmet = demand - sold
        leftover = self.stock - sold

        reward = (p["price"] * sold
                  - p["order_cost"] * order
                  - p["holding_cost"] * leftover
                  - p["stockout_penalty"] * unmet)

        # order arrives overnight, warehouse cannot exceed capacity
        self.stock = min(p["capacity"], leftover + order)
        self.t += 1
        done = self.t >= self.end

        info = {"demand": demand, "sold": sold, "unmet": unmet,
                "leftover": leftover, "order": order}
        next_state = None if done else self.state()
        return next_state, float(reward), done, info
