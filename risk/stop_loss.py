# ==========================
# risk/stop_loss.py
# ==========================
# Stop loss management and optimization.

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from utils.logger import get_logger

logger = get_logger("stop_loss")

class StopLossType(Enum):
    """Types of stop loss strategies."""
    FIXED = "fixed"              # Fixed percentage/price stop
    TRAILING = "trailing"        # Trailing stop
    VOLATILITY = "volatility"    # Volatility-based stop
    ATR = "atr"                  # Average True Range stop
    CHANDELIER = "chandelier"    # Chandelier stop
    PARABOLIC = "parabolic"      # Parabolic SAR stop
    QUANTUM = "quantum"          # Quantum-enhanced stop
    ADAPTIVE = "adaptive"        # Adaptive stop based on market conditions

@dataclass
class StopLossConfig:
    """Configuration for stop loss strategies."""
    stop_type: StopLossType
    initial_stop: float = 0.02  # Initial stop loss percentage
    trailing_stop: float = 0.01  # Trailing stop percentage
    volatility_window: int = 20  # Window for volatility calculation
    atr_period: int = 14  # Period for ATR calculation
    atr_multiplier: float = 2.0  # Multiplier for ATR stop
    chandelier_period: int = 22  # Period for Chandelier stop
    chandelier_multiplier: float = 3.0  # Multiplier for Chandelier stop
    acceleration_factor: float = 0.02  # Acceleration factor for Parabolic SAR
    max_acceleration: float = 0.2  # Maximum acceleration for Parabolic SAR
    quantum_amplitude: float = 0.7  # Quantum amplitude for quantum stops
    quantum_phase_shift: float = 0.1  # Quantum phase shift
    adaptive_threshold: float = 0.5  # Threshold for adaptive stops
    market_regime: str = "normal"  # Market regime (normal, volatile, trending)
    time_horizon: str = "1d"  # Time horizon for stop loss
    risk_free_rate: float = 0.02  # Annual risk-free rate

@dataclass
class PositionConfig:
    """Configuration for position sizing and management."""
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    risk_per_trade: float = 0.01  # Risk per trade as fraction of portfolio
    max_drawdown: float = 0.2  # Maximum drawdown allowed
    position_scaling: float = 0.5  # Position scaling factor based on confidence
    min_position_size: float = 0.01  # Minimum position size
    max_leverage: float = 1.0  # Maximum leverage allowed
    correlation_threshold: float = 0.7  # Maximum correlation for position overlap
    sector_exposure: float = 0.3  # Maximum sector exposure
    market_cap_weight: bool = True  # Whether to weight by market cap
    volatility_weight: bool = True  # Whether to weight by volatility
    momentum_weight: bool = True  # Whether to weight by momentum

@dataclass
class PortfolioConfig:
    """Configuration for portfolio-level risk management."""
    max_portfolio_risk: float = 0.2  # Maximum portfolio risk as fraction of total value
    max_sector_risk: float = 0.3  # Maximum risk per sector
    max_correlation_risk: float = 0.4  # Maximum risk from correlated positions
    risk_budget: float = 0.1  # Risk budget per trade
    risk_decay: float = 0.95  # Risk budget decay factor
    min_risk_budget: float = 0.01  # Minimum risk budget
    max_leverage: float = 1.0  # Maximum portfolio leverage
    max_drawdown: float = 0.2  # Maximum portfolio drawdown
    risk_metrics_window: int = 20  # Window for risk metrics calculation
    rebalance_threshold: float = 0.1  # Threshold for portfolio rebalancing
    stop_loss_threshold: float = 0.05  # Threshold for stop loss adjustment

@dataclass
class RiskMetricsConfig:
    """Configuration for advanced risk metrics."""
    var_confidence: float = 0.95  # Confidence level for VaR
    cvar_confidence: float = 0.99  # Confidence level for CVaR
    lookback_window: int = 252  # Lookback window for risk metrics
    monte_carlo_sims: int = 10000  # Number of Monte Carlo simulations
    correlation_window: int = 60  # Window for correlation calculation
    regime_threshold: float = 0.2  # Threshold for regime detection
    regime_window: int = 20  # Window for regime detection
    rebalance_threshold: float = 0.1  # Threshold for portfolio rebalancing
    min_rebalance_interval: int = 5  # Minimum days between rebalancing
    max_position_change: float = 0.2  # Maximum position change in rebalancing

class PositionManager:
    """Manages position sizing and tracking."""
    
    def __init__(self, config: PositionConfig):
        self.config = config
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: Dict[str, List[Dict[str, Any]]] = {}
        self._setup_position_manager()
    
    def _setup_position_manager(self) -> None:
        """Initialize position manager."""
        try:
            logger.info("Initializing position manager")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup position manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate position configuration."""
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
        except Exception as e:
            logger.error(f"Invalid position configuration: {str(e)}")
            raise
    
    def calculate_position_size(self,
                              symbol: str,
                              current_price: float,
                              stop_price: float,
                              portfolio_value: float,
                              confidence: float = 1.0,
                              correlation: float = 0.0,
                              sector_exposure: float = 0.0) -> float:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            stop_price: Stop loss price
            portfolio_value: Total portfolio value
            confidence: Trade confidence (0-1)
            correlation: Correlation with existing positions
            sector_exposure: Current sector exposure
            
        Returns:
            Position size in units
        """
        try:
            # Calculate risk per share
            risk_per_share = current_price - stop_price
            
            # Calculate maximum position size based on risk
            max_risk_size = (portfolio_value * self.config.risk_per_trade) / risk_per_share
            
            # Apply position scaling based on confidence
            scaled_size = max_risk_size * (self.config.position_scaling * confidence)
            
            # Apply correlation adjustment
            if correlation > self.config.correlation_threshold:
                scaled_size *= (1 - correlation)
            
            # Apply sector exposure adjustment
            if sector_exposure > self.config.sector_exposure:
                scaled_size *= (1 - (sector_exposure - self.config.sector_exposure))
            
            # Apply maximum position size constraint
            max_size = portfolio_value * self.config.max_position_size / current_price
            scaled_size = min(scaled_size, max_size)
            
            # Apply minimum position size constraint
            min_size = portfolio_value * self.config.min_position_size / current_price
            scaled_size = max(scaled_size, min_size)
            
            return scaled_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {str(e)}")
            raise
    
    def update_position(self,
                       symbol: str,
                       position_size: float,
                       entry_price: float,
                       stop_price: float,
                       current_price: float,
                       timestamp: datetime = None) -> None:
        """
        Update position information.
        
        Args:
            symbol: Trading symbol
            position_size: Position size in units
            entry_price: Entry price
            stop_price: Stop loss price
            current_price: Current price
            timestamp: Position update timestamp
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Calculate position metrics
            unrealized_pnl = (current_price - entry_price) * position_size
            risk_amount = (entry_price - stop_price) * position_size
            risk_reward_ratio = abs(unrealized_pnl / risk_amount) if risk_amount != 0 else 0
            
            # Update position
            self.positions[symbol] = {
                'size': position_size,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'risk_amount': risk_amount,
                'risk_reward_ratio': risk_reward_ratio,
                'last_updated': timestamp
            }
            
            # Update history
            if symbol not in self.position_history:
                self.position_history[symbol] = []
            
            self.position_history[symbol].append({
                'timestamp': timestamp,
                'size': position_size,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'risk_amount': risk_amount,
                'risk_reward_ratio': risk_reward_ratio
            })
            
            # Keep only last 1000 entries
            if len(self.position_history[symbol]) > 1000:
                self.position_history[symbol] = self.position_history[symbol][-1000:]
                
        except Exception as e:
            logger.error(f"Failed to update position: {str(e)}")
            raise
    
    def get_position_statistics(self, symbol: str) -> Dict[str, Any]:
        """
        Get position statistics for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with position statistics
        """
        try:
            if symbol not in self.position_history:
                return {}
            
            history = self.position_history[symbol]
            pnls = [entry['unrealized_pnl'] for entry in history]
            risk_rewards = [entry['risk_reward_ratio'] for entry in history]
            
            return {
                'total_trades': len(history),
                'avg_pnl': np.mean(pnls),
                'max_pnl': np.max(pnls),
                'min_pnl': np.min(pnls),
                'std_pnl': np.std(pnls),
                'avg_risk_reward': np.mean(risk_rewards),
                'max_risk_reward': np.max(risk_rewards),
                'min_risk_reward': np.min(risk_rewards),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0,
                'recent_trades': history[-10:]
            }
        except Exception as e:
            logger.error(f"Failed to get position statistics: {str(e)}")
            raise

class PortfolioManager:
    """Manages portfolio-level risk and position sizing."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.risk_budget: float = config.risk_budget
        self.portfolio_value: float = 0.0
        self._setup_portfolio_manager()
    
    def _setup_portfolio_manager(self) -> None:
        """Initialize portfolio manager."""
        try:
            logger.info("Initializing portfolio manager")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup portfolio manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate portfolio configuration."""
        try:
            if self.config.max_portfolio_risk <= 0 or self.config.max_portfolio_risk > 1:
                raise ValueError("Max portfolio risk must be between 0 and 1")
            if self.config.max_sector_risk <= 0 or self.config.max_sector_risk > 1:
                raise ValueError("Max sector risk must be between 0 and 1")
            if self.config.max_correlation_risk <= 0 or self.config.max_correlation_risk > 1:
                raise ValueError("Max correlation risk must be between 0 and 1")
            if self.config.risk_budget <= 0 or self.config.risk_budget > 1:
                raise ValueError("Risk budget must be between 0 and 1")
            if self.config.risk_decay <= 0 or self.config.risk_decay > 1:
                raise ValueError("Risk decay must be between 0 and 1")
            if self.config.min_risk_budget <= 0:
                raise ValueError("Min risk budget must be positive")
            if self.config.max_leverage <= 0:
                raise ValueError("Max leverage must be positive")
            if self.config.max_drawdown <= 0 or self.config.max_drawdown > 1:
                raise ValueError("Max drawdown must be between 0 and 1")
        except Exception as e:
            logger.error(f"Invalid portfolio configuration: {str(e)}")
            raise
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value and risk budget."""
        try:
            self.portfolio_value = value
            # Update risk budget based on decay
            self.risk_budget = max(
                self.risk_budget * self.config.risk_decay,
                self.config.min_risk_budget
            )
        except Exception as e:
            logger.error(f"Failed to update portfolio value: {str(e)}")
            raise
    
    def calculate_position_risk(self,
                              symbol: str,
                              position_size: float,
                              current_price: float,
                              stop_price: float,
                              sector: str = None,
                              correlations: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate position risk metrics.
        
        Args:
            symbol: Trading symbol
            position_size: Position size in units
            current_price: Current price
            stop_price: Stop loss price
            sector: Position sector
            correlations: Correlations with other positions
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            # Calculate position risk
            position_value = position_size * current_price
            position_risk = (current_price - stop_price) * position_size
            
            # Calculate sector risk
            sector_risk = 0.0
            if sector:
                sector_positions = {
                    s: p for s, p in self.positions.items()
                    if p.get('sector') == sector
                }
                sector_risk = sum(p['risk_amount'] for p in sector_positions.values())
                sector_risk += position_risk
            
            # Calculate correlation risk
            correlation_risk = 0.0
            if correlations:
                for other_symbol, correlation in correlations.items():
                    if other_symbol in self.positions:
                        other_risk = self.positions[other_symbol]['risk_amount']
                        correlation_risk += abs(correlation * other_risk)
                correlation_risk += position_risk
            
            return {
                'position_risk': position_risk,
                'sector_risk': sector_risk,
                'correlation_risk': correlation_risk,
                'risk_ratio': position_risk / self.portfolio_value if self.portfolio_value > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to calculate position risk: {str(e)}")
            raise
    
    def validate_position_risk(self, risk_metrics: Dict[str, float]) -> bool:
        """
        Validate position risk against portfolio constraints.
        
        Args:
            risk_metrics: Position risk metrics
            
        Returns:
            Whether position risk is within constraints
        """
        try:
            # Check portfolio risk
            if risk_metrics['position_risk'] > self.portfolio_value * self.config.max_portfolio_risk:
                return False
            
            # Check sector risk
            if risk_metrics['sector_risk'] > self.portfolio_value * self.config.max_sector_risk:
                return False
            
            # Check correlation risk
            if risk_metrics['correlation_risk'] > self.portfolio_value * self.config.max_correlation_risk:
                return False
            
            # Check risk budget
            if risk_metrics['position_risk'] > self.portfolio_value * self.risk_budget:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to validate position risk: {str(e)}")
            raise
    
    def update_position(self,
                       symbol: str,
                       position_size: float,
                       current_price: float,
                       stop_price: float,
                       sector: str = None,
                       correlations: Dict[str, float] = None) -> None:
        """
        Update position in portfolio.
        
        Args:
            symbol: Trading symbol
            position_size: Position size in units
            current_price: Current price
            stop_price: Stop loss price
            sector: Position sector
            correlations: Correlations with other positions
        """
        try:
            # Calculate risk metrics
            risk_metrics = self.calculate_position_risk(
                symbol, position_size, current_price, stop_price,
                sector, correlations
            )
            
            # Validate risk
            if not self.validate_position_risk(risk_metrics):
                raise ValueError("Position risk exceeds portfolio constraints")
            
            # Update position
            self.positions[symbol] = {
                'size': position_size,
                'current_price': current_price,
                'stop_price': stop_price,
                'sector': sector,
                'risk_amount': risk_metrics['position_risk'],
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to update position: {str(e)}")
            raise
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio risk metrics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        try:
            total_risk = sum(p['risk_amount'] for p in self.positions.values())
            sector_risks = {}
            correlation_risks = {}
            
            for symbol, position in self.positions.items():
                sector = position.get('sector')
                if sector:
                    sector_risks[sector] = sector_risks.get(sector, 0) + position['risk_amount']
            
            return {
                'total_risk': total_risk,
                'risk_ratio': total_risk / self.portfolio_value if self.portfolio_value > 0 else 0,
                'sector_risks': sector_risks,
                'correlation_risks': correlation_risks,
                'risk_budget': self.risk_budget,
                'positions': len(self.positions)
            }
        except Exception as e:
            logger.error(f"Failed to get portfolio metrics: {str(e)}")
            raise

class RiskMetricsManager:
    """Manages advanced risk metrics and portfolio rebalancing."""
    
    def __init__(self, config: RiskMetricsConfig):
        self.config = config
        self.last_rebalance: Optional[datetime] = None
        self._setup_risk_metrics()
    
    def _setup_risk_metrics(self) -> None:
        """Initialize risk metrics manager."""
        try:
            logger.info("Initializing risk metrics manager")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup risk metrics manager: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate risk metrics configuration."""
        try:
            if self.config.var_confidence <= 0 or self.config.var_confidence >= 1:
                raise ValueError("VaR confidence must be between 0 and 1")
            if self.config.cvar_confidence <= 0 or self.config.cvar_confidence >= 1:
                raise ValueError("CVaR confidence must be between 0 and 1")
            if self.config.lookback_window < 2:
                raise ValueError("Lookback window must be at least 2")
            if self.config.monte_carlo_sims < 100:
                raise ValueError("Monte Carlo simulations must be at least 100")
            if self.config.correlation_window < 2:
                raise ValueError("Correlation window must be at least 2")
            if self.config.regime_threshold <= 0:
                raise ValueError("Regime threshold must be positive")
            if self.config.regime_window < 2:
                raise ValueError("Regime window must be at least 2")
            if self.config.rebalance_threshold <= 0:
                raise ValueError("Rebalance threshold must be positive")
            if self.config.min_rebalance_interval < 1:
                raise ValueError("Minimum rebalance interval must be at least 1")
            if self.config.max_position_change <= 0 or self.config.max_position_change > 1:
                raise ValueError("Maximum position change must be between 0 and 1")
        except Exception as e:
            logger.error(f"Invalid risk metrics configuration: {str(e)}")
            raise
    
    def calculate_var(self, returns: pd.Series) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            
        Returns:
            VaR value
        """
        try:
            return np.percentile(returns, (1 - self.config.var_confidence) * 100)
        except Exception as e:
            logger.error(f"Failed to calculate VaR: {str(e)}")
            raise
    
    def calculate_cvar(self, returns: pd.Series) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series of returns
            
        Returns:
            CVaR value
        """
        try:
            var = self.calculate_var(returns)
            return returns[returns <= var].mean()
        except Exception as e:
            logger.error(f"Failed to calculate CVaR: {str(e)}")
            raise
    
    def calculate_portfolio_var(self,
                              positions: Dict[str, Dict[str, Any]],
                              historical_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate portfolio VaR using Monte Carlo simulation.
        
        Args:
            positions: Dictionary of positions
            historical_data: Dictionary of historical data
            
        Returns:
            Portfolio VaR
        """
        try:
            # Calculate returns and correlations
            returns = {}
            for symbol, data in historical_data.items():
                if symbol in positions:
                    returns[symbol] = data['Close'].pct_change().dropna()
            
            # Calculate correlation matrix
            corr_matrix = pd.DataFrame(returns).corr()
            
            # Generate correlated random returns
            np.random.seed(42)
            correlated_returns = np.random.multivariate_normal(
                mean=np.zeros(len(returns)),
                cov=corr_matrix,
                size=self.config.monte_carlo_sims
            )
            
            # Calculate portfolio returns
            portfolio_returns = []
            for sim in range(self.config.monte_carlo_sims):
                sim_return = 0
                for i, (symbol, position) in enumerate(positions.items()):
                    position_value = position['size'] * position['current_price']
                    sim_return += position_value * correlated_returns[sim, i]
                portfolio_returns.append(sim_return)
            
            return np.percentile(portfolio_returns, (1 - self.config.var_confidence) * 100)
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio VaR: {str(e)}")
            raise
    
    def detect_market_regime(self, historical_data: pd.DataFrame) -> str:
        """
        Detect current market regime.
        
        Args:
            historical_data: Historical price data
            
        Returns:
            Market regime (normal, volatile, trending)
        """
        try:
            returns = historical_data['Close'].pct_change()
            volatility = returns.rolling(window=self.config.regime_window).std()
            trend = returns.rolling(window=self.config.regime_window).mean()
            
            current_volatility = volatility.iloc[-1]
            current_trend = trend.iloc[-1]
            
            if current_volatility > self.config.regime_threshold:
                return "volatile"
            elif abs(current_trend) > self.config.regime_threshold:
                return "trending"
            else:
                return "normal"
                
        except Exception as e:
            logger.error(f"Failed to detect market regime: {str(e)}")
            raise
    
    def calculate_rebalancing_needs(self,
                                  positions: Dict[str, Dict[str, Any]],
                                  target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate position changes needed for rebalancing.
        
        Args:
            positions: Current positions
            target_weights: Target position weights
            
        Returns:
            Dictionary of position changes
        """
        try:
            # Calculate current weights
            total_value = sum(p['size'] * p['current_price'] for p in positions.values())
            current_weights = {
                symbol: (p['size'] * p['current_price']) / total_value
                for symbol, p in positions.items()
            }
            
            # Calculate needed changes
            changes = {}
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                change = target_weight - current_weight
                
                # Apply maximum change constraint
                if abs(change) > self.config.max_position_change:
                    change = np.sign(change) * self.config.max_position_change
                
                changes[symbol] = change
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to calculate rebalancing needs: {str(e)}")
            raise
    
    def should_rebalance(self,
                        positions: Dict[str, Dict[str, Any]],
                        target_weights: Dict[str, float]) -> bool:
        """
        Check if portfolio should be rebalanced.
        
        Args:
            positions: Current positions
            target_weights: Target position weights
            
        Returns:
            Whether to rebalance
        """
        try:
            # Check minimum interval
            if self.last_rebalance is not None:
                days_since_rebalance = (datetime.now() - self.last_rebalance).days
                if days_since_rebalance < self.config.min_rebalance_interval:
                    return False
            
            # Calculate current weights
            total_value = sum(p['size'] * p['current_price'] for p in positions.values())
            current_weights = {
                symbol: (p['size'] * p['current_price']) / total_value
                for symbol, p in positions.items()
            }
            
            # Check weight deviations
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                if abs(target_weight - current_weight) > self.config.rebalance_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check rebalancing need: {str(e)}")
            raise

class StopLossManager:
    """Manages stop loss strategies and execution."""
    
    def __init__(self,
                 config: StopLossConfig,
                 position_config: PositionConfig = None,
                 portfolio_config: PortfolioConfig = None,
                 risk_metrics_config: RiskMetricsConfig = None):
        self.config = config
        self.position_config = position_config or PositionConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.risk_metrics_config = risk_metrics_config or RiskMetricsConfig()
        self.position_manager = PositionManager(self.position_config)
        self.portfolio_manager = PortfolioManager(self.portfolio_config)
        self.risk_metrics_manager = RiskMetricsManager(self.risk_metrics_config)
        self.current_stops: Dict[str, float] = {}
        self.stop_history: Dict[str, List[Dict[str, Any]]] = {}
        self._setup_stop_loss()
    
    def _setup_stop_loss(self) -> None:
        """Initialize stop loss strategy."""
        try:
            logger.info(f"Initializing stop loss manager with type: {self.config.stop_type}")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup stop loss: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate stop loss configuration."""
        try:
            if self.config.initial_stop <= 0:
                raise ValueError("Initial stop must be positive")
            if self.config.trailing_stop <= 0:
                raise ValueError("Trailing stop must be positive")
            if self.config.volatility_window < 2:
                raise ValueError("Volatility window must be at least 2")
            if self.config.atr_period < 2:
                raise ValueError("ATR period must be at least 2")
            if self.config.atr_multiplier <= 0:
                raise ValueError("ATR multiplier must be positive")
            if self.config.chandelier_period < 2:
                raise ValueError("Chandelier period must be at least 2")
            if self.config.chandelier_multiplier <= 0:
                raise ValueError("Chandelier multiplier must be positive")
            if self.config.acceleration_factor <= 0:
                raise ValueError("Acceleration factor must be positive")
            if self.config.max_acceleration <= 0:
                raise ValueError("Maximum acceleration must be positive")
            if self.config.quantum_amplitude <= 0:
                raise ValueError("Quantum amplitude must be positive")
            if self.config.quantum_phase_shift <= 0:
                raise ValueError("Quantum phase shift must be positive")
            if self.config.adaptive_threshold <= 0:
                raise ValueError("Adaptive threshold must be positive")
        except Exception as e:
            logger.error(f"Invalid stop loss configuration: {str(e)}")
            raise
    
    def calculate_stop_loss(self, 
                          symbol: str, 
                          current_price: float, 
                          historical_data: pd.DataFrame) -> float:
        """
        Calculate stop loss level based on strategy.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            historical_data: Historical price data
            
        Returns:
            Stop loss price
        """
        try:
            if self.config.stop_type == StopLossType.FIXED:
                return self._calculate_fixed_stop(current_price)
            elif self.config.stop_type == StopLossType.TRAILING:
                return self._calculate_trailing_stop(symbol, current_price, historical_data)
            elif self.config.stop_type == StopLossType.VOLATILITY:
                return self._calculate_volatility_stop(current_price, historical_data)
            elif self.config.stop_type == StopLossType.ATR:
                return self._calculate_atr_stop(current_price, historical_data)
            elif self.config.stop_type == StopLossType.CHANDELIER:
                return self._calculate_chandelier_stop(current_price, historical_data)
            elif self.config.stop_type == StopLossType.PARABOLIC:
                return self._calculate_parabolic_stop(current_price, historical_data)
            elif self.config.stop_type == StopLossType.QUANTUM:
                return self._calculate_quantum_stop(current_price, historical_data)
            elif self.config.stop_type == StopLossType.ADAPTIVE:
                return self._calculate_adaptive_stop(current_price, historical_data)
            else:
                raise ValueError(f"Unsupported stop loss type: {self.config.stop_type}")
        except Exception as e:
            logger.error(f"Failed to calculate stop loss: {str(e)}")
            raise
    
    def _calculate_fixed_stop(self, current_price: float) -> float:
        """Calculate fixed stop loss."""
        return current_price * (1 - self.config.initial_stop)
    
    def _calculate_trailing_stop(self, 
                               symbol: str, 
                               current_price: float, 
                               historical_data: pd.DataFrame) -> float:
        """Calculate trailing stop loss."""
        try:
            if symbol not in self.current_stops:
                self.current_stops[symbol] = current_price * (1 - self.config.initial_stop)
            
            if current_price > self.current_stops[symbol] / (1 - self.config.trailing_stop):
                self.current_stops[symbol] = current_price * (1 - self.config.trailing_stop)
            
            return self.current_stops[symbol]
        except Exception as e:
            logger.error(f"Failed to calculate trailing stop: {str(e)}")
            raise
    
    def _calculate_volatility_stop(self, 
                                 current_price: float, 
                                 historical_data: pd.DataFrame) -> float:
        """Calculate volatility-based stop loss."""
        try:
            returns = historical_data['Close'].pct_change()
            volatility = returns.rolling(window=self.config.volatility_window).std()
            current_volatility = volatility.iloc[-1]
            return current_price * (1 - current_volatility * self.config.atr_multiplier)
        except Exception as e:
            logger.error(f"Failed to calculate volatility stop: {str(e)}")
            raise
    
    def _calculate_atr_stop(self, 
                          current_price: float, 
                          historical_data: pd.DataFrame) -> float:
        """Calculate ATR-based stop loss."""
        try:
            high = historical_data['High']
            low = historical_data['Low']
            close = historical_data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=self.config.atr_period).mean()
            
            return current_price - atr.iloc[-1] * self.config.atr_multiplier
        except Exception as e:
            logger.error(f"Failed to calculate ATR stop: {str(e)}")
            raise
    
    def _calculate_chandelier_stop(self, 
                                 current_price: float, 
                                 historical_data: pd.DataFrame) -> float:
        """Calculate Chandelier stop loss."""
        try:
            high = historical_data['High']
            atr = self._calculate_atr(current_price, historical_data)
            highest_high = high.rolling(window=self.config.chandelier_period).max()
            return highest_high.iloc[-1] - atr * self.config.chandelier_multiplier
        except Exception as e:
            logger.error(f"Failed to calculate Chandelier stop: {str(e)}")
            raise
    
    def _calculate_parabolic_stop(self, 
                                current_price: float, 
                                historical_data: pd.DataFrame) -> float:
        """Calculate Parabolic SAR stop loss."""
        try:
            high = historical_data['High']
            low = historical_data['Low']
            
            # Initialize variables
            trend = 1  # 1 for uptrend, -1 for downtrend
            sar = low.iloc[0]
            ep = high.iloc[0]
            af = self.config.acceleration_factor
            
            # Calculate SAR
            for i in range(1, len(historical_data)):
                if trend == 1:
                    sar = sar + af * (ep - sar)
                    if low.iloc[i] < sar:
                        trend = -1
                        sar = ep
                        ep = low.iloc[i]
                        af = self.config.acceleration_factor
                    else:
                        if high.iloc[i] > ep:
                            ep = high.iloc[i]
                            af = min(af + self.config.acceleration_factor, 
                                   self.config.max_acceleration)
                else:
                    sar = sar + af * (ep - sar)
                    if high.iloc[i] > sar:
                        trend = 1
                        sar = ep
                        ep = high.iloc[i]
                        af = self.config.acceleration_factor
                    else:
                        if low.iloc[i] < ep:
                            ep = low.iloc[i]
                            af = min(af + self.config.acceleration_factor, 
                                   self.config.max_acceleration)
            
            return sar
        except Exception as e:
            logger.error(f"Failed to calculate Parabolic SAR stop: {str(e)}")
            raise
    
    def _calculate_quantum_stop(self, 
                              current_price: float, 
                              historical_data: pd.DataFrame) -> float:
        """Calculate quantum-enhanced stop loss."""
        try:
            # Calculate base stop using ATR
            base_stop = self._calculate_atr_stop(current_price, historical_data)
            
            # Apply quantum enhancement
            returns = historical_data['Close'].pct_change()
            volatility = returns.rolling(window=self.config.volatility_window).std()
            
            # Calculate quantum state
            quantum_state = np.exp(1j * self.config.quantum_phase_shift * volatility.iloc[-1])
            quantum_factor = np.abs(quantum_state) * self.config.quantum_amplitude
            
            # Adjust stop based on quantum state
            return base_stop * (1 - quantum_factor)
        except Exception as e:
            logger.error(f"Failed to calculate quantum stop: {str(e)}")
            raise
    
    def _calculate_adaptive_stop(self, 
                               current_price: float, 
                               historical_data: pd.DataFrame) -> float:
        """Calculate adaptive stop loss based on market conditions."""
        try:
            # Calculate market regime indicators
            returns = historical_data['Close'].pct_change()
            volatility = returns.rolling(window=self.config.volatility_window).std()
            trend = returns.rolling(window=self.config.volatility_window).mean()
            
            # Determine market regime
            if volatility.iloc[-1] > self.config.adaptive_threshold:
                # Volatile market - use tighter stops
                return self._calculate_atr_stop(current_price, historical_data)
            elif abs(trend.iloc[-1]) > self.config.adaptive_threshold:
                # Trending market - use trailing stops
                return self._calculate_trailing_stop("adaptive", current_price, historical_data)
            else:
                # Normal market - use fixed stops
                return self._calculate_fixed_stop(current_price)
        except Exception as e:
            logger.error(f"Failed to calculate adaptive stop: {str(e)}")
            raise
    
    def _calculate_atr(self, 
                      current_price: float, 
                      historical_data: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        try:
            high = historical_data['High']
            low = historical_data['Low']
            close = historical_data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=self.config.atr_period).mean().iloc[-1]
        except Exception as e:
            logger.error(f"Failed to calculate ATR: {str(e)}")
            raise
    
    def update_stop_history(self, 
                          symbol: str, 
                          stop_price: float, 
                          current_price: float) -> None:
        """
        Update stop loss history.
        
        Args:
            symbol: Trading symbol
            stop_price: Stop loss price
            current_price: Current price
        """
        try:
            if symbol not in self.stop_history:
                self.stop_history[symbol] = []
            
            self.stop_history[symbol].append({
                'timestamp': datetime.now(),
                'stop_price': stop_price,
                'current_price': current_price,
                'distance': (current_price - stop_price) / current_price
            })
            
            # Keep only last 1000 entries
            if len(self.stop_history[symbol]) > 1000:
                self.stop_history[symbol] = self.stop_history[symbol][-1000:]
        except Exception as e:
            logger.error(f"Failed to update stop history: {str(e)}")
            raise
    
    def get_stop_statistics(self, symbol: str) -> Dict[str, Any]:
        """
        Get stop loss statistics for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with stop loss statistics
        """
        try:
            if symbol not in self.stop_history:
                return {}
            
            history = self.stop_history[symbol]
            distances = [entry['distance'] for entry in history]
            
            return {
                'total_stops': len(history),
                'avg_distance': np.mean(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'std_distance': np.std(distances),
                'recent_stops': history[-10:]
            }
        except Exception as e:
            logger.error(f"Failed to get stop statistics: {str(e)}")
            raise
    
    def manage_position(self,
                       symbol: str,
                       current_price: float,
                       historical_data: pd.DataFrame,
                       portfolio_value: float,
                       confidence: float = 1.0,
                       correlation: float = 0.0,
                       sector_exposure: float = 0.0,
                       sector: str = None,
                       correlations: Dict[str, float] = None,
                       target_weight: float = None) -> Dict[str, Any]:
        """
        Manage position with stop loss and position sizing.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            historical_data: Historical price data
            portfolio_value: Total portfolio value
            confidence: Trade confidence (0-1)
            correlation: Correlation with existing positions
            sector_exposure: Current sector exposure
            sector: Position sector
            correlations: Correlations with other positions
            target_weight: Target portfolio weight
            
        Returns:
            Dictionary with position management details
        """
        try:
            # Update portfolio value
            self.portfolio_manager.update_portfolio_value(portfolio_value)
            
            # Detect market regime
            market_regime = self.risk_metrics_manager.detect_market_regime(historical_data)
            
            # Calculate stop loss
            stop_price = self.calculate_stop_loss(symbol, current_price, historical_data)
            
            # Calculate position size
            position_size = self.position_manager.calculate_position_size(
                symbol, current_price, stop_price, portfolio_value,
                confidence, correlation, sector_exposure
            )
            
            # Update position in portfolio
            self.portfolio_manager.update_position(
                symbol, position_size, current_price, stop_price,
                sector, correlations
            )
            
            # Check rebalancing needs
            if target_weight is not None:
                target_weights = {symbol: target_weight}
                if self.risk_metrics_manager.should_rebalance(
                    self.portfolio_manager.positions, target_weights
                ):
                    changes = self.risk_metrics_manager.calculate_rebalancing_needs(
                        self.portfolio_manager.positions, target_weights
                    )
                    position_size *= (1 + changes.get(symbol, 0))
            
            # Update position history
            self.position_manager.update_position(
                symbol, position_size, current_price, stop_price, current_price
            )
            
            # Update stop history
            self.update_stop_history(symbol, stop_price, current_price)
            
            # Calculate risk metrics
            returns = historical_data['Close'].pct_change()
            var = self.risk_metrics_manager.calculate_var(returns)
            cvar = self.risk_metrics_manager.calculate_cvar(returns)
            
            return {
                'stop_price': stop_price,
                'position_size': position_size,
                'risk_amount': (current_price - stop_price) * position_size,
                'position_stats': self.position_manager.get_position_statistics(symbol),
                'stop_stats': self.get_stop_statistics(symbol),
                'portfolio_metrics': self.portfolio_manager.get_portfolio_metrics(),
                'risk_metrics': {
                    'var': var,
                    'cvar': cvar,
                    'market_regime': market_regime
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to manage position: {str(e)}")
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
        
        # Initialize configurations
        stop_config = StopLossConfig(
            stop_type=StopLossType.ADAPTIVE,
            initial_stop=0.02,
            trailing_stop=0.01,
            volatility_window=20,
            atr_period=14,
            atr_multiplier=2.0,
            chandelier_period=22,
            chandelier_multiplier=3.0,
            acceleration_factor=0.02,
            max_acceleration=0.2,
            quantum_amplitude=0.7,
            quantum_phase_shift=0.1,
            adaptive_threshold=0.5,
            market_regime="normal",
            time_horizon="1d",
            risk_free_rate=0.02
        )
        
        position_config = PositionConfig(
            max_position_size=0.1,
            risk_per_trade=0.01,
            max_drawdown=0.2,
            position_scaling=0.5,
            min_position_size=0.01,
            max_leverage=1.0,
            correlation_threshold=0.7,
            sector_exposure=0.3
        )
        
        portfolio_config = PortfolioConfig(
            max_portfolio_risk=0.2,
            max_sector_risk=0.3,
            max_correlation_risk=0.4,
            risk_budget=0.1,
            risk_decay=0.95,
            min_risk_budget=0.01,
            max_leverage=1.0,
            max_drawdown=0.2
        )
        
        risk_metrics_config = RiskMetricsConfig(
            var_confidence=0.95,
            cvar_confidence=0.99,
            lookback_window=252,
            monte_carlo_sims=10000,
            correlation_window=60,
            regime_threshold=0.2,
            regime_window=20,
            rebalance_threshold=0.1,
            min_rebalance_interval=5,
            max_position_change=0.2
        )
        
        manager = StopLossManager(
            stop_config,
            position_config,
            portfolio_config,
            risk_metrics_config
        )
        
        # Manage position
        current_price = data['Close'].iloc[-1]
        portfolio_value = 100000
        position_details = manager.manage_position(
            "TEST", current_price, data, portfolio_value,
            confidence=0.8, correlation=0.3, sector_exposure=0.2,
            sector="Technology", correlations={"AAPL": 0.6, "MSFT": 0.5},
            target_weight=0.1
        )
        
        print("\nPosition Management Details:", json.dumps(position_details, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}") 