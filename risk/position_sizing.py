# ==========================
# risk/position_sizing.py
# ==========================
# Advanced position sizing strategies.

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from utils.logger import get_logger

logger = get_logger("position_sizing")

class PositionSizingType(Enum):
    """Types of position sizing strategies."""
    KELLY = "kelly"              # Kelly Criterion
    FIXED_FRACTION = "fixed"     # Fixed Fraction
    FIXED_RATIO = "ratio"        # Fixed Ratio
    VOLATILITY = "volatility"    # Volatility-based
    OPTIMAL_F = "optimal_f"      # Optimal F
    MARTINGALE = "martingale"    # Martingale
    ANTI_MARTINGALE = "anti"     # Anti-Martingale
    RISK_PARITY = "parity"       # Risk Parity
    CORE_SATELLITE = "core"      # Core-Satellite
    DYNAMIC = "dynamic"          # Dynamic Sizing

@dataclass
class PositionSizingConfig:
    """Configuration for position sizing strategies."""
    sizing_type: PositionSizingType
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    risk_per_trade: float = 0.01  # Risk per trade as fraction of portfolio
    max_drawdown: float = 0.2  # Maximum drawdown allowed
    position_scaling: float = 0.5  # Position scaling factor based on confidence
    min_position_size: float = 0.01  # Minimum position size
    max_leverage: float = 1.0  # Maximum leverage allowed
    correlation_threshold: float = 0.7  # Maximum correlation for position overlap
    sector_exposure: float = 0.3  # Maximum sector exposure
    kelly_fraction: float = 0.5  # Fraction of Kelly Criterion to use
    volatility_window: int = 20  # Window for volatility calculation
    optimal_f_lookback: int = 100  # Lookback period for Optimal F
    martingale_factor: float = 2.0  # Martingale multiplication factor
    anti_martingale_factor: float = 1.5  # Anti-Martingale multiplication factor
    risk_parity_target: float = 0.1  # Target risk contribution
    core_weight: float = 0.7  # Core portfolio weight
    dynamic_sensitivity: float = 0.5  # Sensitivity to market conditions

class PositionSizingManager:
    """Manages position sizing strategies."""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self._setup_position_sizing()
    
    def _setup_position_sizing(self) -> None:
        """Initialize position sizing manager."""
        try:
            logger.info("Initializing position sizing manager")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup position sizing manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate position sizing configuration."""
        try:
            if self.config.max_position_size <= 0 or self.config.max_position_size > 1:
                raise ValueError("Max position size must be between 0 and 1")
            if self.config.risk_per_trade <= 0 or self.config.risk_per_trade > 1:
                raise ValueError("Risk per trade must be between 0 and 1")
            if self.config.max_drawdown <= 0 or self.config.max_drawdown > 1:
                raise ValueError("Max drawdown must be between 0 and 1")
            if self.config.position_scaling <= 0 or self.config.position_scaling > 1:
                raise ValueError("Position scaling must be between 0 and 1")
            if self.config.min_position_size <= 0:
                raise ValueError("Min position size must be positive")
            if self.config.max_leverage <= 0:
                raise ValueError("Max leverage must be positive")
            if self.config.correlation_threshold <= 0 or self.config.correlation_threshold > 1:
                raise ValueError("Correlation threshold must be between 0 and 1")
            if self.config.sector_exposure <= 0 or self.config.sector_exposure > 1:
                raise ValueError("Sector exposure must be between 0 and 1")
            if self.config.kelly_fraction <= 0 or self.config.kelly_fraction > 1:
                raise ValueError("Kelly fraction must be between 0 and 1")
            if self.config.volatility_window < 2:
                raise ValueError("Volatility window must be at least 2")
            if self.config.optimal_f_lookback < 2:
                raise ValueError("Optimal F lookback must be at least 2")
            if self.config.martingale_factor <= 1:
                raise ValueError("Martingale factor must be greater than 1")
            if self.config.anti_martingale_factor <= 1:
                raise ValueError("Anti-Martingale factor must be greater than 1")
            if self.config.risk_parity_target <= 0:
                raise ValueError("Risk parity target must be positive")
            if self.config.core_weight <= 0 or self.config.core_weight > 1:
                raise ValueError("Core weight must be between 0 and 1")
            if self.config.dynamic_sensitivity <= 0:
                raise ValueError("Dynamic sensitivity must be positive")
        except Exception as e:
            logger.error(f"Invalid position sizing configuration: {str(e)}")
            raise
    
    def calculate_position_size(self,
                              symbol: str,
                              current_price: float,
                              stop_price: float,
                              portfolio_value: float,
                              historical_data: pd.DataFrame,
                              confidence: float = 1.0,
                              correlation: float = 0.0,
                              sector_exposure: float = 0.0) -> float:
        """
        Calculate position size based on strategy.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            stop_price: Stop loss price
            portfolio_value: Total portfolio value
            historical_data: Historical price data
            confidence: Trade confidence (0-1)
            correlation: Correlation with existing positions
            sector_exposure: Current sector exposure
            
        Returns:
            Position size in units
        """
        try:
            if self.config.sizing_type == PositionSizingType.KELLY:
                return self._calculate_kelly_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence
                )
            elif self.config.sizing_type == PositionSizingType.FIXED_FRACTION:
                return self._calculate_fixed_fraction_size(
                    symbol, current_price, stop_price, portfolio_value,
                    confidence
                )
            elif self.config.sizing_type == PositionSizingType.FIXED_RATIO:
                return self._calculate_fixed_ratio_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence
                )
            elif self.config.sizing_type == PositionSizingType.VOLATILITY:
                return self._calculate_volatility_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence
                )
            elif self.config.sizing_type == PositionSizingType.OPTIMAL_F:
                return self._calculate_optimal_f_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence
                )
            elif self.config.sizing_type == PositionSizingType.MARTINGALE:
                return self._calculate_martingale_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence
                )
            elif self.config.sizing_type == PositionSizingType.ANTI_MARTINGALE:
                return self._calculate_anti_martingale_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence
                )
            elif self.config.sizing_type == PositionSizingType.RISK_PARITY:
                return self._calculate_risk_parity_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence, correlation
                )
            elif self.config.sizing_type == PositionSizingType.CORE_SATELLITE:
                return self._calculate_core_satellite_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence
                )
            elif self.config.sizing_type == PositionSizingType.DYNAMIC:
                return self._calculate_dynamic_size(
                    symbol, current_price, stop_price, portfolio_value,
                    historical_data, confidence, correlation, sector_exposure
                )
            else:
                raise ValueError(f"Unsupported position sizing type: {self.config.sizing_type}")
        except Exception as e:
            logger.error(f"Failed to calculate position size: {str(e)}")
            raise
    
    def _calculate_kelly_size(self,
                            symbol: str,
                            current_price: float,
                            stop_price: float,
                            portfolio_value: float,
                            historical_data: pd.DataFrame,
                            confidence: float) -> float:
        """Calculate position size using Kelly Criterion."""
        try:
            # Calculate win rate and win/loss ratio
            returns = historical_data['Close'].pct_change()
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
            win_loss_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 else 1.0
            
            # Calculate Kelly fraction
            kelly_f = win_rate - (1 - win_rate) / win_loss_ratio
            
            # Apply fraction and confidence
            position_size = kelly_f * self.config.kelly_fraction * confidence
            
            # Apply constraints
            position_size = min(position_size, self.config.max_position_size)
            position_size = max(position_size, self.config.min_position_size)
            
            return position_size * portfolio_value / current_price
            
        except Exception as e:
            logger.error(f"Failed to calculate Kelly size: {str(e)}")
            raise
    
    def _calculate_fixed_fraction_size(self,
                                     symbol: str,
                                     current_price: float,
                                     stop_price: float,
                                     portfolio_value: float,
                                     confidence: float) -> float:
        """Calculate position size using fixed fraction."""
        try:
            # Calculate risk per share
            risk_per_share = current_price - stop_price
            
            # Calculate position size
            position_size = (portfolio_value * self.config.risk_per_trade) / risk_per_share
            
            # Apply confidence scaling
            position_size *= confidence
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate fixed fraction size: {str(e)}")
            raise
    
    def _calculate_fixed_ratio_size(self,
                                  symbol: str,
                                  current_price: float,
                                  stop_price: float,
                                  portfolio_value: float,
                                  historical_data: pd.DataFrame,
                                  confidence: float) -> float:
        """Calculate position size using fixed ratio."""
        try:
            # Calculate volatility
            returns = historical_data['Close'].pct_change()
            volatility = returns.std()
            
            # Calculate position size
            position_size = (portfolio_value * self.config.risk_per_trade) / (volatility * current_price)
            
            # Apply confidence scaling
            position_size *= confidence
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate fixed ratio size: {str(e)}")
            raise
    
    def _calculate_volatility_size(self,
                                 symbol: str,
                                 current_price: float,
                                 stop_price: float,
                                 portfolio_value: float,
                                 historical_data: pd.DataFrame,
                                 confidence: float) -> float:
        """Calculate position size based on volatility."""
        try:
            # Calculate volatility
            returns = historical_data['Close'].pct_change()
            volatility = returns.rolling(window=self.config.volatility_window).std().iloc[-1]
            
            # Calculate position size
            position_size = (portfolio_value * self.config.risk_per_trade) / (volatility * current_price)
            
            # Apply confidence scaling
            position_size *= confidence
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility size: {str(e)}")
            raise
    
    def _calculate_optimal_f_size(self,
                                symbol: str,
                                current_price: float,
                                stop_price: float,
                                portfolio_value: float,
                                historical_data: pd.DataFrame,
                                confidence: float) -> float:
        """Calculate position size using Optimal F."""
        try:
            # Calculate returns
            returns = historical_data['Close'].pct_change()
            
            # Calculate Optimal F
            f_values = []
            for f in np.linspace(0.01, 1.0, 100):
                equity_curve = (1 + returns * f).cumprod()
                f_values.append((f, equity_curve.iloc[-1]))
            
            optimal_f = max(f_values, key=lambda x: x[1])[0]
            
            # Calculate position size
            position_size = optimal_f * confidence
            
            # Apply constraints
            position_size = min(position_size, self.config.max_position_size)
            position_size = max(position_size, self.config.min_position_size)
            
            return position_size * portfolio_value / current_price
            
        except Exception as e:
            logger.error(f"Failed to calculate Optimal F size: {str(e)}")
            raise
    
    def _calculate_martingale_size(self,
                                 symbol: str,
                                 current_price: float,
                                 stop_price: float,
                                 portfolio_value: float,
                                 historical_data: pd.DataFrame,
                                 confidence: float) -> float:
        """Calculate position size using Martingale strategy."""
        try:
            # Get position history
            if symbol in self.position_history:
                last_position = self.position_history[symbol][-1]
                if last_position['unrealized_pnl'] < 0:
                    # Increase position size after loss
                    position_size = last_position['size'] * self.config.martingale_factor
                else:
                    # Reset position size after win
                    position_size = self._calculate_fixed_fraction_size(
                        symbol, current_price, stop_price, portfolio_value, confidence
                    )
            else:
                # Initial position
                position_size = self._calculate_fixed_fraction_size(
                    symbol, current_price, stop_price, portfolio_value, confidence
                )
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate Martingale size: {str(e)}")
            raise
    
    def _calculate_anti_martingale_size(self,
                                      symbol: str,
                                      current_price: float,
                                      stop_price: float,
                                      portfolio_value: float,
                                      historical_data: pd.DataFrame,
                                      confidence: float) -> float:
        """Calculate position size using Anti-Martingale strategy."""
        try:
            # Get position history
            if symbol in self.position_history:
                last_position = self.position_history[symbol][-1]
                if last_position['unrealized_pnl'] > 0:
                    # Increase position size after win
                    position_size = last_position['size'] * self.config.anti_martingale_factor
                else:
                    # Reset position size after loss
                    position_size = self._calculate_fixed_fraction_size(
                        symbol, current_price, stop_price, portfolio_value, confidence
                    )
            else:
                # Initial position
                position_size = self._calculate_fixed_fraction_size(
                    symbol, current_price, stop_price, portfolio_value, confidence
                )
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate Anti-Martingale size: {str(e)}")
            raise
    
    def _calculate_risk_parity_size(self,
                                  symbol: str,
                                  current_price: float,
                                  stop_price: float,
                                  portfolio_value: float,
                                  historical_data: pd.DataFrame,
                                  confidence: float,
                                  correlation: float) -> float:
        """Calculate position size using Risk Parity."""
        try:
            # Calculate volatility
            returns = historical_data['Close'].pct_change()
            volatility = returns.std()
            
            # Calculate correlation-adjusted volatility
            adjusted_volatility = volatility * (1 + correlation)
            
            # Calculate position size
            position_size = (self.config.risk_parity_target * portfolio_value) / (adjusted_volatility * current_price)
            
            # Apply confidence scaling
            position_size *= confidence
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate Risk Parity size: {str(e)}")
            raise
    
    def _calculate_core_satellite_size(self,
                                     symbol: str,
                                     current_price: float,
                                     stop_price: float,
                                     portfolio_value: float,
                                     historical_data: pd.DataFrame,
                                     confidence: float) -> float:
        """Calculate position size using Core-Satellite strategy."""
        try:
            # Calculate base position size
            base_size = self._calculate_fixed_fraction_size(
                symbol, current_price, stop_price, portfolio_value, confidence
            )
            
            # Apply core-satellite weighting
            if confidence > 0.8:
                # Core position
                position_size = base_size * self.config.core_weight
            else:
                # Satellite position
                position_size = base_size * (1 - self.config.core_weight)
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate Core-Satellite size: {str(e)}")
            raise
    
    def _calculate_dynamic_size(self,
                              symbol: str,
                              current_price: float,
                              stop_price: float,
                              portfolio_value: float,
                              historical_data: pd.DataFrame,
                              confidence: float,
                              correlation: float,
                              sector_exposure: float) -> float:
        """Calculate position size using dynamic strategy."""
        try:
            # Calculate base position size
            base_size = self._calculate_fixed_fraction_size(
                symbol, current_price, stop_price, portfolio_value, confidence
            )
            
            # Calculate market conditions
            returns = historical_data['Close'].pct_change()
            volatility = returns.std()
            trend = returns.mean()
            
            # Calculate dynamic adjustment
            volatility_adjustment = 1 / (1 + volatility * self.config.dynamic_sensitivity)
            trend_adjustment = 1 + trend * self.config.dynamic_sensitivity
            correlation_adjustment = 1 / (1 + correlation)
            exposure_adjustment = 1 / (1 + sector_exposure)
            
            # Apply adjustments
            position_size = base_size * volatility_adjustment * trend_adjustment * correlation_adjustment * exposure_adjustment
            
            # Apply constraints
            position_size = min(position_size, portfolio_value * self.config.max_position_size / current_price)
            position_size = max(position_size, portfolio_value * self.config.min_position_size / current_price)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate dynamic size: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.normal(100, 1, 100),
            'High': np.random.normal(101, 1, 100),
            'Low': np.random.normal(99, 1, 100),
            'Close': np.random.normal(100, 1, 100)
        }, index=dates)
        
        # Initialize position sizing manager
        config = PositionSizingConfig(
            sizing_type=PositionSizingType.DYNAMIC,
            max_position_size=0.1,
            risk_per_trade=0.01,
            max_drawdown=0.2,
            position_scaling=0.5,
            min_position_size=0.01,
            max_leverage=1.0,
            correlation_threshold=0.7,
            sector_exposure=0.3,
            kelly_fraction=0.5,
            volatility_window=20,
            optimal_f_lookback=100,
            martingale_factor=2.0,
            anti_martingale_factor=1.5,
            risk_parity_target=0.1,
            core_weight=0.7,
            dynamic_sensitivity=0.5
        )
        
        manager = PositionSizingManager(config)
        
        # Calculate position size
        current_price = data['Close'].iloc[-1]
        stop_price = current_price * 0.95
        portfolio_value = 100000
        position_size = manager.calculate_position_size(
            "TEST", current_price, stop_price, portfolio_value, data,
            confidence=0.8, correlation=0.3, sector_exposure=0.2
        )
        
        print(f"\nPosition Size: {position_size:.2f} units")
        
    except Exception as e:
        print(f"Error: {str(e)}") 