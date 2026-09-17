"""
Optional Stage 5 (bonus) - feast_serve
--------------------------------------
Shows the two sides of a feature store with Feast:

  OFFLINE store  (full history, Parquet)  -> get_historical_features()
      builds point-in-time-correct training rows.

  ONLINE store   (latest values, SQLite)  -> get_online_features()
      answers "what are today's features?" in milliseconds,
      which we feed to the trained Q-table to recommend today's order.

Steps: write offline data -> apply definitions -> offline retrieval
       -> materialize (offline -> online) -> online retrieval -> decision

Output: metrics/feast.json
"""

import sys

import pandas as pd
from feast import FeatureStore

from agent import QLearningAgent
from utils import ROOT, load_params, save_json

REPO = ROOT / "feature_repo"
sys.path.insert(0, str(REPO))
from features import demand_features, demand_source, store_item  # noqa: E402

FEATURES = ["demand_features:day_of_week",
            "demand_features:rolling_demand",
            "demand_features:demand_trend"]


def main() -> None:
    params = load_params()
    ep = params["env"]
    processed = ROOT / "data" / "processed"

    # 1. Write the offline store (history of features) --------------------
    hist = pd.concat([pd.read_parquet(processed / "train.parquet"),
                      pd.read_parquet(processed / "test.parquet")])
    hist["store_item_id"] = (hist["store"] * 1000 + hist["item"]).astype("int64")
    hist["event_timestamp"] = pd.to_datetime(hist["date"]).dt.tz_localize("UTC")
    hist["day_of_week"] = hist["day_of_week"].astype("int64")
    hist["demand_trend"] = hist["demand_trend"].astype("int64")
    (REPO / "data").mkdir(exist_ok=True)
    hist[["store_item_id", "event_timestamp", "day_of_week", "rolling_demand",
          "demand_trend"]].to_parquet(REPO / "data" / "demand_features.parquet", index=False)
    entity_id = int(hist["store_item_id"].iloc[0])

    # 2. Register the feature definitions (same as `feast apply`) ---------
    store = FeatureStore(repo_path=str(REPO))
    store.apply([store_item, demand_source, demand_features])

    # 3. OFFLINE retrieval: features "as of" chosen past timestamps --------
    sample = hist.sample(5, random_state=0)
    entity_df = sample[["store_item_id", "event_timestamp"]]
    training_rows = store.get_historical_features(entity_df=entity_df,
                                                  features=FEATURES).to_df()
    merged = training_rows.merge(sample[["event_timestamp", "rolling_demand"]],
                                 on="event_timestamp", suffixes=("_feast", "_file"))
    offline_ok = bool((merged["rolling_demand_feast"] - merged["rolling_demand_file"])
                      .abs().max() < 1e-9)

    # 4. Materialize: copy the latest values into the online store --------
    start = hist["event_timestamp"].min().to_pydatetime()
    end = hist["event_timestamp"].max().to_pydatetime() + pd.Timedelta(days=1)
    store.materialize(start_date=start, end_date=end)

    # 5. ONLINE retrieval: today's features for one store/item ------------
    online = store.get_online_features(features=FEATURES,
                                       entity_rows=[{"store_item_id": entity_id}]).to_dict()
    dow = int(online["day_of_week"][0])
    rolling = float(online["rolling_demand"][0])
    trend = int(online["demand_trend"][0])

    # 6. Use the served features to make a decision with the Q-table ------
    agent = QLearningAgent(len(ep["actions"]), 0, 0, epsilon=0.0)
    agent.load(ROOT / "models" / "q_table.json")
    decisions = {}
    for stock in (0, 20, 60):                      # a few example stock levels
        stock_bucket = sum(stock >= edge for edge in ep["stock_bins"])
        action = agent.act((stock_bucket, dow, trend))
        decisions[f"stock_{stock}"] = ep["actions"][action]

    result = {
        "offline_rows_retrieved": len(training_rows),
        "offline_matches_source": offline_ok,
        "online_features": {"store_item_id": entity_id, "day_of_week": dow,
                            "rolling_demand": round(rolling, 2), "demand_trend": trend},
        "recommended_order": decisions,
    }
    save_json(result, ROOT / "metrics" / "feast.json")
    print("feast_serve done:", result)


if __name__ == "__main__":
    main()
