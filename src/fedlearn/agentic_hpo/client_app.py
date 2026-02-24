from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from flwr.app import Context
from flwr.clientapp import ClientApp
from flwr.common import Message, RecordDict, ArrayRecord, MetricRecord
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fedlearn.common.annotation import annotate_categorical_columns
from fedlearn.common.config import HParams
from fedlearn.common.metrics import compute_binary_metrics
from fedlearn.common.model import get_model, set_model_params, get_input_feature_names, get_model_params

app = ClientApp()

# Constants

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DUCKDB_PATH = PROJECT_ROOT / "data" / "duckdb" / "fedlearn.duckdb"
VIEW_NAME = "v_features_icu_stay_clean"

TARGET_COL = "prolonged_stay"
REGION_COL = "hospital_region"

CLIENT_REGION_MAP = {
    "client_midwest": ["Midwest"],
    "client_south": ["South"],
    "client_other": ["West", "Northeast", None],
}

# Partitioning scheme
# partition-id = 0 -> client_midwest
# partition-id = 1 -> client_south
# partition-id = 2 -> client_other
CLIENT_KEYS = ["client_midwest", "client_south", "client_other"]


def _get_client_key(context: Context) -> str:
    """
    Map Flower's partition-id to our logical client bucket name.
    """
    partition_id = int(context.node_config["partition-id"])
    try:
        return CLIENT_KEYS[partition_id]
    except IndexError:
        raise RuntimeError(
            f"partition-id={partition_id} out of range for CLIENT_KEYS={CLIENT_KEYS}"
        )


def _load_client_data(context: Context) -> pd.DataFrame:
    """
    Load only this client's partition from DuckDB.

    The mapping is:
      partition-id -> client bucket -> list of raw regions.

    Example:
      partition-id=0 -> "client_midwest" -> ["Midwest"]
      partition-id=1 -> "client_south"   -> ["South"]
      partition-id=2 -> "client_other"   -> ["West", "Northeast", NULL]
    """
    client_key = _get_client_key(context)
    regions = CLIENT_REGION_MAP[client_key]

    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        include_null = None in regions
        real_regions = [r for r in regions if r is not None]

        where_clauses = []
        params: list = []

        if real_regions:
            placeholders = ", ".join(["?"] * len(real_regions))
            where_clauses.append(f"{REGION_COL} IN ({placeholders})")
            params.extend(real_regions)

        if include_null:
            where_clauses.append(f"{REGION_COL} IS NULL")

        where_sql = " OR ".join(where_clauses) if where_clauses else "TRUE"

        # noinspection SqlNoDataSourceInspection
        query = f"SELECT * FROM {VIEW_NAME} WHERE {where_sql} ORDER BY patientunitstayid"

        df = conn.execute(query, params).df()
    finally:
        conn.close()

    # normalize pandas.NA -> np.nan so sklearn imputers are happy
    df = df.where(df.notna(), np.nan)

    # ensure categorical columns have right categories
    df = annotate_categorical_columns(df)

    return df


def _get_train_eval_data(context: Context) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load this client's local data and split into train/eval sets.
    """
    df = _load_client_data(context)

    if df.empty:
        raise RuntimeError("No rows found for this client's partition")

    y = df[TARGET_COL]

    # Use exactly the columns the preprocessor expects, in the same order
    feat_cols = get_input_feature_names()
    missing = [c for c in feat_cols if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Client partition is missing expected feature columns: {missing}"
        )

    X = df[feat_cols]

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    return X_train, y_train, X_eval, y_eval


def _init_model(message: Message, context: Context, hp: HParams | None = None) -> Pipeline:
    """
    Build model and load incoming model params.
    """
    incoming_arrays = message.content["arrays"]

    if hp is None:
        hp = HParams.from_message(message, context)

    model = get_model(hp)
    set_model_params(model, incoming_arrays.to_numpy_ndarrays())

    return model


@app.train()
def train(message: Message, context: Context) -> Message:
    """
    Perform one round of local training for this client.

    Steps:
      1) Load this client's local train/eval data.
      2) Load global model parameters from the server.
      3) Initialize the local model with those parameters.
      4) Train for 'local-epochs' on the local training data.
      5) Compute metrics on the local training data.
      6) Return updated parameters and metrics to the server.
    """
    # load local train data (i.e. for this region)
    X_train, y_train, _, _ = _get_train_eval_data(context)

    hp = HParams.from_message(message, context)
    print(f"[Client] Hyperparams this round: {hp}")

    model = _init_model(message, context, hp)
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # local training
    X_proc = pre.transform(X_train)
    clf.fit(X_proc, y_train)  # uses max_iter=local_epochs

    # compute metrics on train split
    metrics_dict = compute_binary_metrics(model, X_train, y_train)
    num_examples = float(len(X_train))
    metrics_dict["num-examples"] = num_examples

    reply_content = RecordDict({
        "arrays": ArrayRecord(get_model_params(model)),
        "metrics": MetricRecord(metrics_dict),
    })

    reply_message = Message(
        content=reply_content,
        reply_to=message,
    )

    return reply_message


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """
    Local evaluation using current global parameters.

    This uses the same partitioning logic and eval split as `train`,
    but does not perform any further training.
    """
    # load local eval data
    _, _, X_eval, y_eval = _get_train_eval_data(context)

    model = _init_model(message, context)

    # compute metrics on eval split
    metrics_dict = compute_binary_metrics(model, X_eval, y_eval)
    num_examples = float(len(X_eval))
    metrics_dict["num-examples"] = num_examples

    reply_content = RecordDict({
        "metrics": MetricRecord(metrics_dict),
    })

    return Message(
        content=reply_content,
        reply_to=message,
    )
