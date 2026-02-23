from __future__ import annotations

from pathlib import Path

import joblib
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedlearn.federated.utils import (
    get_model,
    get_model_params,
    set_initial_params,
    set_model_params,
)

app = ServerApp()

# Constants

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_PATH = CONFIG_DIR / "federated_sgd.pkl"


@app.main()
def main(grid: Grid, context: Context) -> None:
    # make sure "configs" dir exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # read run config from pyproject.toml or cli
    num_rounds: int = context.run_config["num-server-rounds"]
    local_epochs: int = context.run_config["local-epochs"]
    penalty: str = context.run_config["penalty"]
    class_weight_cfg = str(context.run_config.get("class-weight", "none")).lower()
    class_weight = "balanced" if class_weight_cfg == "balanced" else None
    sgd_learning_rate = str(context.run_config.get("sgd-learning-rate", "optimal"))
    sgd_eta0 = float(context.run_config.get("sgd-eta0", 0.0))

    # create and initialize model
    model = get_model(
        penalty=penalty,
        local_epochs=local_epochs,
        class_weight=class_weight,
        sgd_learning_rate=sgd_learning_rate,
        sgd_eta0=sgd_eta0,
    )
    set_initial_params(model)

    arrays = ArrayRecord(get_model_params(model))

    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)

    # initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # run agentic_hpo training
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # get final global params and save model
    final_params = result.arrays.to_numpy_ndarrays()
    set_model_params(model, final_params)

    print(f"Saving final model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
