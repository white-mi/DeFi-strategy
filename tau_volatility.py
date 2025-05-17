from dataclasses import dataclass
from typing import List
import numpy as np

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity)
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity


@dataclass
class TauResetParams(BaseStrategyParams):
    """
    Parameters for the τ-reset strategy:
    - TAU: The width of the price range (bucket) around the current price.
    - INITIAL_BALANCE: The initial balance for liquidity allocation.
    """

    INITIAL_BALANCE: float
    INFO_TIME : int
    ALPHA : float
    C : float



class TauResetStrategy(BaseStrategy):
    """
    The τ-reset strategy manages liquidity in Uniswap v3 by concentrating it
    within a price range around the current market price. If the price exits this range,
    the liquidity is reallocated. If no position is open, it deposits funds first.

    Based on
    https://drops.dagstuhl.de/storage/00lipics/lipics-vol282-aft2023/LIPIcs.AFT.2023.25/LIPIcs.AFT.2023.25.pdf
    """

    # Decimals for token0 and token1 for Uniswap V3 LP Config
    # This is pool-specific and should be set before running the strategy
    # They are not in the BaseStrategyParams because they are not hyperparameters
    # and are specific to the pool being traded consts.
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1
    tau: int = 30
    time: int = 0
    

    def __init__(self, params: TauResetParams, debug: bool = False, *args, **kwargs):
        self._params: TauResetParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        self.distribution = []

    def set_up(self):
        """
        Register the Uniswap V3 LP entity to manage liquidity in the pool.
        """
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self.token0_decimals,
                    token1_decimals=self.token1_decimals
                )
            )
        ))
        assert isinstance(self.get_entity('UNISWAP_V3'), UniswapV3LPEntity)

    def _recalculate_tau(self):
        prices = np.array(self.distribution)
        prices = np.log(prices[1:]) - np.log(prices[:-1])
        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1
        std = np.std(prices, ddof = 1)
        self.tau = self._params.C * (self._params.ALPHA * std + (1 - self._params.ALPHA) * IQR)



    def predict(self) -> List[ActionToTake]:
        """
        Main logic of the strategy. Checks if the price has moved outside
        the predefined range and takes actions if necessary.
        """
        # Retrieve the pool state from the registered entity
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity.global_state
        current_price = global_state.price  # Get the current market price
        self.distribution.append(current_price)
        self.time += 1
        if self.time > self._params.INFO_TIME:
            self._recalculate_tau()
            self.distribution = []
            self.time = 0

        # Check if we need to deposit funds into the LP before proceeding
        if not uniswap_entity.is_position and not self.deposited_initial_funds:
            self._debug("No active position. Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()

        if not uniswap_entity.is_position:
            self._debug("No active position. Run first rebalance")
            return self._rebalance()

        # Calculate the boundaries of the price range (bucket)
        lower_bound, upper_bound = uniswap_entity.internal_state.price_lower, uniswap_entity.internal_state.price_upper

        # If the price moves outside the range, reallocate liquidity
        if current_price < lower_bound or current_price > upper_bound:
            self._debug(f"Rebalance {current_price} moved outside range [{lower_bound}, {upper_bound}].")
            return self._rebalance()
        return []

    def _deposit_to_lp(self) -> List[ActionToTake]:
        """
        Deposit funds into the Uniswap LP if no position is currently open.
        """
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self) -> List[ActionToTake]:
        """
        Reallocate liquidity to a new range centered around the new price.
        """
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')

        # Step 1: Withdraw liquidity from the current range
        if entity.internal_state.positions:
            actions.append(
                ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={}))
            )
            self._debug("Liquidity withdrawn from the current range.")

        # Step 2: Calculate new range boundaries
        tau = self.tau
        reference_price: float = entity.global_state.price
        tick_spacing = self.tick_spacing
        price_lower = reference_price * 1.0001 ** (-tau * tick_spacing)
        price_upper = reference_price * 1.0001 ** (tau * tick_spacing)

        # Step 3: Open a new position centered around the new price
        delegate_get_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
        actions.append(ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(
                action='open_position',
                args={
                    'amount_in_notional': delegate_get_cash,  # Allocate all available cash
                    'price_lower': price_lower,
                    'price_upper': price_upper
                }
            )
        ))
        self._debug(f"New position opened with range [{price_lower}, {price_upper}].")
        return actions
    

import os

from typing import List
from datetime import datetime, UTC

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import (
    UniswapV3EthereumPoolHourDataLoader, EthereumUniswapV3Loader, UniswapV3EthereumPoolMinuteDataLoader
)
from fractal.loaders.binance import BinanceHourPriceLoader, BinanceMinutePriceLoader
from fractal.loaders.structs import PriceHistory, PoolHistory

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState


THE_GRAPH_API_KEY = '149d25473edb2b1e7db32f27051d74bd'


def get_observations(
        pool_data: PoolHistory, price_data: PriceHistory,
        start_time: datetime = None, end_time: datetime = None
    ) -> List[Observation]:
    """
    Get observations from the pool and price data for the TauResetStrategy.

    Returns:
        List[Observation]: The observation list for TauResetStrategy.
    """
    observations_df: pd.DataFrame = pool_data.join(price_data)
    observations_df = observations_df.dropna()
    observations_df = observations_df.loc[start_time:end_time]
    if start_time is None:
        start_time = observations_df.index.min()
    if end_time is None:
        end_time = observations_df.index.max()
    observations_df = observations_df[observations_df.tvl > 0]
    observations_df = observations_df.sort_index()
    return [
        Observation(
            timestamp=timestamp,
            states={
                'UNISWAP_V3': UniswapV3LPGlobalState(price=price, tvl=tvls, volume=volume, fees=fees, liquidity=liquidity),
            }
        ) for timestamp, (tvls, volume, fees, liquidity, price) in observations_df.iterrows()
    ]


def build_observations(
        ticker: str, pool_address: str, api_key: str,
        start_time: datetime = None, end_time: datetime = None, fidelity: str = 'hour',
    ) -> List[Observation]:
    """
    Build observations for the TauResetStrategy from the given start and end time.
    """
    if fidelity == 'hour':
        pool_data: PoolHistory = UniswapV3EthereumPoolHourDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        binance_prices: PriceHistory = BinanceHourPriceLoader(ticker, loader_type=LoaderType.CSV).read(with_run=True)
    elif fidelity == 'minute':
        pool_data: PoolHistory = UniswapV3EthereumPoolMinuteDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        binance_prices: PriceHistory = BinanceMinutePriceLoader(ticker, loader_type=LoaderType.CSV,
                                                                start_time=start_time, end_time=end_time).read(with_run=True)
    else:
        raise ValueError("Fidelity must be either 'hour' or 'minute'.")
    return get_observations(pool_data, binance_prices, start_time, end_time)
