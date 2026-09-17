"""
Feast feature definitions.

Entity       : store_item   (one store + one item, key = store_item_id, e.g. 1001)
Feature view : demand_features  -> day_of_week, rolling_demand, demand_trend

Note: current stock is NOT a feature here. It changes with the agent's own
orders, so it is live state of the simulator, not historical data.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

store_item = Entity(name="store_item", join_keys=["store_item_id"], value_type=ValueType.INT64)

# Offline store: a Parquet file written by src/feast_serve.py
demand_source = FileSource(
    name="demand_source",
    path="data/demand_features.parquet",
    timestamp_field="event_timestamp",
)

demand_features = FeatureView(
    name="demand_features",
    entities=[store_item],
    ttl=timedelta(days=0),          # 0 = feature values never expire
    schema=[
        Field(name="day_of_week", dtype=Int64),
        Field(name="rolling_demand", dtype=Float64),
        Field(name="demand_trend", dtype=Int64),
    ],
    online=True,
    source=demand_source,
)
