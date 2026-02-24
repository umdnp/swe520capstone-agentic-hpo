from __future__ import annotations

import logging
from pathlib import Path

import joblib
from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fedlearn.agentic_hpo.runners import BaselineRunner, StaticHPORunner, AgenticHPORunner, ExperimentRunner
from fedlearn.common.model import set_model_params

app = ServerApp()

logger = logging.getLogger(__name__)

# Constants

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"

RUNNERS: dict[str, ExperimentRunner] = {
    "baseline": BaselineRunner(),
    "static_hpo": StaticHPORunner(),
    "agentic_hpo": AgenticHPORunner(),
}


@app.main()
def main(grid: Grid, context: Context) -> None:
    # make sure "configs" dir exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    experiment = str(context.run_config.get("experiment", "baseline")).lower()
    runner = RUNNERS.get(experiment)

    if runner is None:
        raise ValueError(f"Unknown experiment {experiment!r}. Valid: {sorted(RUNNERS)}")

    result, model = runner.run(grid, context)

    # get final global params and save model
    final_params = result.arrays.to_numpy_ndarrays()
    set_model_params(model, final_params)

    save_file = CONFIG_DIR / f"{experiment}.pkl"
    logger.info(f"Saving final model to {save_file}")
    joblib.dump(model, save_file)
