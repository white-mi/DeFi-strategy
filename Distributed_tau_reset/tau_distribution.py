from dataclasses import dataclass
from typing import List
import numpy as np

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from Modified_entity.uniswap_v3_lp_modified import UniswapV3LPConfig, UniswapV3LPEntity




@dataclass
class TauResetParams(BaseStrategyParams):
    """
    Parameters for the τ-reset strategy:
    - TAU: The width of the price range (bucket) around the current price.
    - INITIAL_BALANCE: The initial balance for liquidity allocation.
    """
    TAU: float
    INITIAL_BALANCE: float
    BINS: int
    INFO_TIME: int
    U : int



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
    previous_price: float = 0
    current_price: float = 0
    tick_counter: int = 0
    last_center : float = 0

    def __init__(self, params: TauResetParams, debug: bool = False, *args, **kwargs):
        self._params: TauResetParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        self.distribution = [1] * self._params.BINS
        self.new_distribution  = []

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
        self.previous_price = self.get_entity('UNISWAP_V3').global_state.price
        self.current_price = self.previous_price

    def _update_dist(self):
        
        prices = np.asarray(self.new_distribution, dtype=np.float64)

        if self._params.U == 1:
            prices = np.log(prices[1:]) - np.log(prices[:-1])
        if self._params.U == 0:
            prices = prices[1:] - prices[:-1]

        hist, bin_edges = np.histogram(prices, bins=self._params.BINS)

        self.distribution = list(hist)


    def _check_rebalance(self):
        tau = self._params.TAU
        tick_spacing = self.tick_spacing
        price_lower = self.last_center * 1.0001 ** (-tau * tick_spacing)
        price_upper = self.last_center * 1.0001 ** (tau * tick_spacing)
        if self.current_price > price_upper or self.current_price < price_lower:
            return True
        return False
    
    def predict(self) -> List[ActionToTake]:
        """
        Main logic of the strategy. Checks if the price has moved outside
        the predefined range and takes actions if necessary.
        """
        # Retrieve the pool state from the registered entity
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity.global_state
        self.previous_price = self.current_price
        self.current_price = global_state.price  # Get the current market price
        self.tick_counter += 1
        self.new_distribution.append(self.current_price)
        if self.tick_counter > self._params.INFO_TIME:
            self._update_dist()
            self.new_distribution = []
            self.tick_counter = 0

        # Check if we need to deposit funds into the LP before proceeding
        if not uniswap_entity._internal_state.positions and not self.deposited_initial_funds:
            self._debug("No active position. Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()

        if not uniswap_entity._internal_state.positions:
            self._debug("No active position. Run first rebalance")
            return self._rebalance()

        if self._check_rebalance():
            self._debug(f"Rebalance {self.current_price}.")
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
        tau = self._params.TAU
        reference_price: float = entity.global_state.price
        tick_spacing = self.tick_spacing
        price_lower = reference_price * 1.0001 ** (-tau * tick_spacing)
        price_upper = reference_price * 1.0001 ** (tau * tick_spacing)
        self.last_center = reference_price
        delta = price_upper - price_lower

        
        # Step 3: Open a new position centered around the new price
        for i in range(self._params.BINS):
            partial_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash * \
                (self.distribution[i] / sum(self.distribution))
            partial_lower = price_lower + i * (delta / self._params.BINS)
            partial_upper = price_lower + (i + 1) * (delta / self._params.BINS)
            actions.append(ActionToTake(
                entity_name='UNISWAP_V3',
                action=Action(
                    action='open_position',
                    args={
                        'amount_in_notional': partial_cash,
                        'price_lower': partial_lower,
                        'price_upper': partial_upper
                    }
                )
            ))
        self._debug(f"New position opened with range [{price_lower}, {price_upper}].")
        return actions