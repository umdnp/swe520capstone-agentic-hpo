from __future__ import annotations

import logging

import pandas as pd
from flwr.app import Context
from flwr.clientapp import ClientApp
from flwr.common import ArrayRecord, Message, MetricRecord, RecordDict
from sklearn.pipeline import Pipeline

from fedlearn.common.config import HParams, PHASE_FINAL, PHASE_PARAM, PHASE_HPO_TRIAL
from fedlearn.common.data_split import CLIENT_KEYS, get_client_train_val_test_by_key
from fedlearn.common.metrics import compute_binary_metrics
from fedlearn.common.model import get_model, get_model_params, set_model_params

app = ClientApp()

logger = logging.getLogger(__name__)


def _get_client_key(context: Context) -> str:
    """
    Map Flower's partition-id to our logical client bucket name.
    """
    partition_id = int(context.node_config["partition-id"])
    try:
        return CLIENT_KEYS[partition_id]
    except IndexError:
        raise ValueError(
            f"partition-id={partition_id} out of range for CLIENT_KEYS={CLIENT_KEYS}"
        )


def _get_phase(message: Message, context: Context) -> str:
    cfg = message.content.get("config")
    if cfg is not None and PHASE_PARAM in cfg:
        return str(cfg[PHASE_PARAM])

    return str(context.run_config.get(PHASE_PARAM, PHASE_FINAL))


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

    Depending on phase:
      - hpo-trial: fit on local train split
      - final: fit on local train + val splits
    """
    # load this client's local train/val/test splits
    client_key = _get_client_key(context)
    X_train, y_train, X_val, y_val, _, _ = get_client_train_val_test_by_key(client_key)

    phase = _get_phase(message, context)

    if phase == PHASE_HPO_TRIAL:
        X_fit, y_fit = X_train, y_train
    elif phase == PHASE_FINAL:
        X_fit = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_fit = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    else:
        raise ValueError(f"Unknown data phase: {phase!r}")

    hp = HParams.from_message(message, context)
    logger.info("[Client] Hyperparams this round: %s, phase=%s", hp, phase)

    model = _init_model(message, context, hp)
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # local training
    X_proc = pre.transform(X_fit)
    clf.fit(X_proc, y_fit)  # uses max_iter=local_epochs

    # compute metrics on the local fit split
    metrics_dict = compute_binary_metrics(model, X_fit, y_fit)
    metrics_dict["num-examples"] = float(len(X_fit))

    reply_content = RecordDict({
        "arrays": ArrayRecord(get_model_params(model)),
        "metrics": MetricRecord(metrics_dict),
    })

    return Message(content=reply_content, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """
    Local evaluation using current global parameters.

    Depending on phase:
      - hpo-trial: evaluate on local val split
      - final: evaluate on local test split
    """
    # load this client's local train/val/test splits
    client_key = _get_client_key(context)
    _, _, X_val, y_val, X_test, y_test = get_client_train_val_test_by_key(client_key)

    phase = _get_phase(message, context)

    if phase == PHASE_HPO_TRIAL:
        X_eval, y_eval = X_val, y_val
    elif phase == PHASE_FINAL:
        X_eval, y_eval = X_test, y_test
    else:
        raise ValueError(f"Unknown data phase: {phase!r}")

    model = _init_model(message, context)

    # compute metrics on the phase-appropriate evaluation split
    metrics_dict = compute_binary_metrics(model, X_eval, y_eval)
    metrics_dict["num-examples"] = float(len(X_eval))

    reply_content = RecordDict({
        "metrics": MetricRecord(metrics_dict),
    })

    return Message(content=reply_content, reply_to=message)
