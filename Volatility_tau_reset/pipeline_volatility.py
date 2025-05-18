import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from tau_volatility import TauResetStrategy, build_observations, THE_GRAPH_API_KEY


def build_grid():
    grid = ParameterGrid({
        'INFO_TIME': [8, 24, 24 * 3, 24 * 7, 30 * 24], 
        'INITIAL_BALANCE': [1_000_000],
        'C' : [1000, 2000, 5000, 10000, 12000],
        'ALPHA' : [0, 0.1, 0.2, 0.5, 0.8, 0.9, 1]
    })
    return grid


if __name__ == '__main__':
    ticker: str = 'ETHUSDT'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 1, 1, tzinfo=UTC)
    fidelity = 'hour'
    experiment_name = f'rtau_{fidelity}_{ticker}_{pool_address}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    TauResetStrategy.token0_decimals = 6
    TauResetStrategy.token1_decimals = 18
    TauResetStrategy.tick_spacing = 60

    # Define MLFlow and Experiment configurations
    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri='http://127.0.0.1:8080',
        experiment_name='tau_volatility_exp-1'
    )
    observations = build_observations(ticker, pool_address, THE_GRAPH_API_KEY, start_time, end_time, fidelity=fidelity)
    assert len(observations) > 0
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=TauResetStrategy,
        backtest_observations=observations,
        window_size=12,
        params_grid=build_grid(),
        debug=True,
    )
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()