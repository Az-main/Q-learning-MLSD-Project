"""
Stage 4 (bonus) - online_drift
------------------------------
Replays the STREAM year (2016) one day at a time, like live data arriving.

Online learning (two learners, both updated one sample at a time):
    * the Q-table keeps learning from every new day  (online reinforcement learning)
    * a River linear regression predicts each day's demand, then learns from the true value

Drift detection (River ADWIN):
    * DATA drift    -> ADWIN watches the demand values themselves     (did the input change?)
    * CONCEPT drift -> ADWIN watches the forecaster's absolute error  (did the input->output
                       relationship change so the old model is now wrong?)

Response to drift (only if drift.respond is true):
    retrain the agent on the most recent `retrain_window` days by replaying them
    `retrain_passes` times in the simulator (the classic "drift -> retrain" loop).

Outputs:
    metrics/drift.json                -> drift events + online performance
    metrics/plots/online_rewards.csv  -> day-by-day curve for `dvc plots show`
    results/online_drift.png          -> reward curve with drift markers
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from river import drift, linear_model, metrics, optim, preprocessing

from agent import QLearningAgent
from env import InventoryEnv
from utils import ROOT, load_params, save_json

PROCESSED = ROOT / "data" / "processed"


def features(row) -> dict:
    """Input features for the River forecaster (a plain dict, as River expects)."""
    x = {"rolling_demand": float(row.rolling_demand)}
    for d in range(7):
        x[f"dow_{d}"] = 1.0 if row.day_of_week == d else 0.0
    return x


def retrain_on_recent(agent, recent: pd.DataFrame, ep: dict, dp: dict) -> None:
    """Drift response: replay the last few days in the simulator many times.

    The live stream gives only one sample per day, which is far too slow for the
    Q-table to catch up after a sudden change. Replaying the recent window lets
    the agent adapt to the NEW demand level quickly.
    """
    sim = InventoryEnv(recent.reset_index(drop=True), ep)
    agent.alpha, agent.epsilon = dp["retrain_alpha"], dp["retrain_epsilon"]
    for _ in range(dp["retrain_passes"]):
        state, done = sim.reset(), False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = sim.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state


def main() -> None:
    params = load_params()
    ep, op, dp = params["env"], params["online"], params["drift"]

    train = pd.read_parquet(PROCESSED / "train.parquet")
    stream = pd.read_parquet(PROCESSED / "stream.parquet").reset_index(drop=True)

    # ---- the RL agent starts from the trained Q-table ------------------- #
    agent = QLearningAgent(len(ep["actions"]), op["alpha"], params["train"]["gamma"],
                           op["epsilon"], seed=params["train"]["seed"])
    agent.load(ROOT / "models" / "q_table.json")

    # ---- River forecaster: warm it up on the training data, one row at a time
    forecaster = preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(op["river_lr"]))
    for row in train.itertuples():
        forecaster.learn_one(features(row), row.demand)

    # ---- drift detectors (clock=1 -> check for drift every single day) -- #
    data_detector = drift.ADWIN(delta=dp["adwin_delta"], clock=1)
    concept_detector = drift.ADWIN(delta=dp["adwin_delta"], clock=1)
    mae = metrics.MAE()

    env = InventoryEnv(stream, ep)
    state = env.reset()
    data_events, concept_events, log = [], [], []
    retrain_count = 0

    for i, row in enumerate(stream.itertuples()):
        # 1. agent acts, the day happens, agent learns from it (online Q-learning)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state

        # 2. forecaster: predict first, then learn (prequential evaluation)
        x, y = features(row), row.demand
        y_pred = forecaster.predict_one(x)
        error = abs(y - y_pred)
        mae.update(y, y_pred)
        forecaster.learn_one(x, y)

        # 3. drift detection
        data_detector.update(y)
        concept_detector.update(error)
        date = row.date.strftime("%Y-%m-%d")
        detected = False
        if data_detector.drift_detected:
            data_events.append(date)
            detected = True
        if concept_detector.drift_detected:
            concept_events.append(date)
            detected = True

        # 4. response: retrain the agent on the most recent days
        if detected and dp["respond"] and i + 1 >= dp["retrain_window"]:
            recent = stream.iloc[i + 1 - dp["retrain_window"]: i + 1]
            retrain_on_recent(agent, recent, ep, dp)
            agent.alpha, agent.epsilon = op["alpha"], op["epsilon"]
            retrain_count += 1

        log.append({"day": i, "date": date, "demand": float(y), "reward": round(reward, 2),
                    "forecast_error": round(float(error), 2)})
        if done:
            break

    # ---- outputs -------------------------------------------------------- #
    df = pd.DataFrame(log)
    df["rolling_reward"] = df["reward"].rolling(14, min_periods=1).mean().round(2)
    plots = ROOT / "metrics" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    df[["day", "rolling_reward", "forecast_error"]].to_csv(plots / "online_rewards.csv", index=False)

    after = df[df["date"] >= params["prepare"]["drift_start"]]
    before = df[df["date"] < params["prepare"]["drift_start"]]
    summary = {
        "respond_to_drift": dp["respond"],
        "retrain_count": retrain_count,
        "data_drift_count": len(data_events),
        "concept_drift_count": len(concept_events),
        "data_drift_dates": data_events,
        "concept_drift_dates": concept_events,
        "river_forecaster_mae": round(float(mae.get()), 3),
        "online_total_reward": round(float(df["reward"].sum()), 1),
        "avg_daily_reward_before_drift": round(float(before["reward"].mean()), 2),
        "avg_daily_reward_after_drift": round(float(after["reward"].mean()), 2),
    }
    save_json(summary, ROOT / "metrics" / "drift.json")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    dates = pd.to_datetime(df["date"])
    plt.figure(figsize=(9, 4))
    plt.plot(dates, df["rolling_reward"], label="14-day avg reward")
    plt.axvline(pd.Timestamp(params["prepare"]["drift_start"]), color="black",
                linestyle="--", label="simulated drift starts")
    for k, d in enumerate(data_events):
        plt.axvline(pd.Timestamp(d), color="tab:orange", alpha=0.7,
                    label="data drift detected" if k == 0 else None)
    for k, d in enumerate(concept_events):
        plt.axvline(pd.Timestamp(d), color="tab:red", alpha=0.7, linestyle=":",
                    label="concept drift detected" if k == 0 else None)
    plt.title(f"Online stream (respond_to_drift = {dp['respond']})")
    plt.ylabel("reward per day")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "online_drift.png", dpi=120)
    plt.close()

    print("online_drift done:", summary)


if __name__ == "__main__":
    main()
