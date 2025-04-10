# ==========================
# risk/risk_monitor.py
# ==========================
# Real-time risk monitoring and alerts.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from scipy import stats
from utils.logger import get_logger
from risk.risk_metrics import RiskMetricsManager, RiskMetricType
from risk.portfolio_optimizer import PortfolioOptimizer, OptimizationType
from risk.risk_allocator import RiskAllocator, AllocationType
from risk.performance_analyzer import PerformanceAnalyzer, AttributionType

logger = get_logger("risk_monitor")

class AlertType(Enum):
    """Types of risk alerts."""
    VOLATILITY = "volatility"        # Volatility alert
    DRAWDOWN = "drawdown"           # Drawdown alert
    CORRELATION = "correlation"      # Correlation alert
    LIQUIDITY = "liquidity"         # Liquidity alert
    CONCENTRATION = "concentration"  # Concentration alert
    LEVERAGE = "leverage"           # Leverage alert
    MARGIN = "margin"              # Margin alert
    EXPOSURE = "exposure"          # Exposure alert
    REGIME = "regime"             # Regime change alert
    CUSTOM = "custom"             # Custom alert

@dataclass
class AlertConfig:
    """Configuration for risk alerts."""
    alert_type: AlertType = AlertType.VOLATILITY
    threshold: float = 0.02  # Alert threshold
    lookback_window: int = 20  # Lookback window
    min_observations: int = 10  # Minimum observations
    confidence_level: float = 0.95  # Confidence level
    alert_frequency: str = "D"  # Alert frequency
    alert_channels: List[str] = None  # Alert channels
    custom_conditions: Dict[str, Any] = None  # Custom conditions
    severity_level: str = "WARNING"  # Severity level
    notification_template: str = None  # Notification template

@dataclass
class MonitorConfig:
    """Configuration for risk monitoring."""
    alert_configs: List[AlertConfig] = None  # Alert configurations
    monitoring_frequency: str = "1H"  # Monitoring frequency
    real_time: bool = True  # Real-time monitoring
    historical_analysis: bool = True  # Historical analysis
    risk_metrics_manager: RiskMetricsManager = None  # Risk metrics manager
    portfolio_optimizer: PortfolioOptimizer = None  # Portfolio optimizer
    risk_allocator: RiskAllocator = None  # Risk allocator
    performance_analyzer: PerformanceAnalyzer = None  # Performance analyzer
    custom_metrics: Dict[str, Any] = None  # Custom metrics
    monitoring_thresholds: Dict[str, float] = None  # Monitoring thresholds

class RiskMonitor:
    """Manages real-time risk monitoring and alerts."""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        if self.config.alert_configs is None:
            self.config.alert_configs = []
        if self.config.custom_metrics is None:
            self.config.custom_metrics = {}
        if self.config.monitoring_thresholds is None:
            self.config.monitoring_thresholds = {}
        self._setup_risk_monitor()
    
    def _setup_risk_monitor(self) -> None:
        """Initialize risk monitor."""
        try:
            logger.info("Initializing risk monitor")
            self._validate_config()
        except Exception as e:
            logger.error(f"Failed to setup risk monitor: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate monitor configuration."""
        try:
            for alert_config in self.config.alert_configs:
                if alert_config.threshold <= 0:
                    raise ValueError("Alert threshold must be positive")
                if alert_config.lookback_window < alert_config.min_observations:
                    raise ValueError("Lookback window must be at least min_observations")
                if alert_config.confidence_level <= 0 or alert_config.confidence_level >= 1:
                    raise ValueError("Confidence level must be between 0 and 1")
        except Exception as e:
            logger.error(f"Invalid monitor configuration: {str(e)}")
            raise
    
    def monitor_risk(self,
                    portfolio_data: Dict[str, Any],
                    market_data: Dict[str, Any],
                    risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor portfolio risk.
        
        Args:
            portfolio_data: Dictionary containing portfolio data
            market_data: Dictionary containing market data
            risk_data: Dictionary containing risk data
            
        Returns:
            Dictionary containing monitoring results and alerts
        """
        try:
            # Monitor risk metrics
            risk_metrics = self._monitor_risk_metrics(portfolio_data, market_data, risk_data)
            
            # Monitor alerts
            alerts = self._monitor_alerts(portfolio_data, market_data, risk_data)
            
            # Monitor custom metrics
            custom_metrics = self._monitor_custom_metrics(portfolio_data, market_data, risk_data)
            
            return {
                'risk_metrics': risk_metrics,
                'alerts': alerts,
                'custom_metrics': custom_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor risk: {str(e)}")
            raise
    
    def _monitor_risk_metrics(self,
                            portfolio_data: Dict[str, Any],
                            market_data: Dict[str, Any],
                            risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor risk metrics."""
        try:
            if self.config.risk_metrics_manager is None:
                raise ValueError("Risk metrics manager is required for risk monitoring")
            
            # Calculate risk metrics
            risk_metrics = self.config.risk_metrics_manager.calculate_risk_metrics(
                portfolio_data['returns'],
                portfolio_data['weights']
            )
            
            # Check against thresholds
            threshold_violations = {}
            for metric, value in risk_metrics.items():
                if metric in self.config.monitoring_thresholds:
                    if value > self.config.monitoring_thresholds[metric]:
                        threshold_violations[metric] = {
                            'value': value,
                            'threshold': self.config.monitoring_thresholds[metric]
                        }
            
            return {
                'metrics': risk_metrics,
                'threshold_violations': threshold_violations
            }
            
        except Exception as e:
            logger.error(f"Failed to monitor risk metrics: {str(e)}")
            raise
    
    def _monitor_alerts(self,
                       portfolio_data: Dict[str, Any],
                       market_data: Dict[str, Any],
                       risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor alerts."""
        try:
            alerts = []
            
            for alert_config in self.config.alert_configs:
                if alert_config.alert_type == AlertType.VOLATILITY:
                    alert = self._monitor_volatility_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.DRAWDOWN:
                    alert = self._monitor_drawdown_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.CORRELATION:
                    alert = self._monitor_correlation_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.LIQUIDITY:
                    alert = self._monitor_liquidity_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.CONCENTRATION:
                    alert = self._monitor_concentration_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.LEVERAGE:
                    alert = self._monitor_leverage_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.MARGIN:
                    alert = self._monitor_margin_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.EXPOSURE:
                    alert = self._monitor_exposure_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.REGIME:
                    alert = self._monitor_regime_alert(portfolio_data, alert_config)
                elif alert_config.alert_type == AlertType.CUSTOM:
                    alert = self._monitor_custom_alert(portfolio_data, alert_config)
                else:
                    raise ValueError(f"Unsupported alert type: {alert_config.alert_type}")
                
                if alert is not None:
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to monitor alerts: {str(e)}")
            raise
    
    def _monitor_volatility_alert(self,
                                portfolio_data: Dict[str, Any],
                                alert_config: AlertConfig) -> Optional[Dict[str, Any]]:
        """Monitor volatility alert."""
        try:
            # Calculate volatility
            volatility = portfolio_data['returns'].std() * np.sqrt(252)
            
            # Check against threshold
            if volatility > alert_config.threshold:
                return {
                    'type': AlertType.VOLATILITY,
                    'severity': alert_config.severity_level,
                    'value': volatility,
                    'threshold': alert_config.threshold,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to monitor volatility alert: {str(e)}")
            raise
    
    def _monitor_custom_metrics(self,
                              portfolio_data: Dict[str, Any],
                              market_data: Dict[str, Any],
                              risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor custom metrics."""
        try:
            custom_metrics = {}
            
            for metric_name, metric_config in self.config.custom_metrics.items():
                if metric_name == 'custom_metric_1':
                    custom_metrics[metric_name] = self._calculate_custom_metric_1(portfolio_data, metric_config)
                elif metric_name == 'custom_metric_2':
                    custom_metrics[metric_name] = self._calculate_custom_metric_2(portfolio_data, metric_config)
                else:
                    raise ValueError(f"Unsupported custom metric: {metric_name}")
            
            return custom_metrics
            
        except Exception as e:
            logger.error(f"Failed to monitor custom metrics: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        n_assets = 5
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (252, n_assets)),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        weights = pd.DataFrame(
            np.random.dirichlet(np.ones(n_assets), size=252),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Initialize risk monitor
        alert_configs = [
            AlertConfig(
                alert_type=AlertType.VOLATILITY,
                threshold=0.02,
                lookback_window=20,
                min_observations=10,
                confidence_level=0.95,
                alert_frequency="D",
                alert_channels=["email", "slack"],
                severity_level="WARNING"
            ),
            AlertConfig(
                alert_type=AlertType.DRAWDOWN,
                threshold=0.05,
                lookback_window=20,
                min_observations=10,
                confidence_level=0.95,
                alert_frequency="D",
                alert_channels=["email", "slack"],
                severity_level="CRITICAL"
            )
        ]
        
        config = MonitorConfig(
            alert_configs=alert_configs,
            monitoring_frequency="1H",
            real_time=True,
            historical_analysis=True,
            monitoring_thresholds={
                'volatility': 0.02,
                'drawdown': 0.05,
                'correlation': 0.8,
                'concentration': 0.2
            }
        )
        
        monitor = RiskMonitor(config)
        
        # Monitor risk
        portfolio_data = {
            'returns': returns,
            'weights': weights
        }
        
        market_data = {}
        risk_data = {}
        
        result = monitor.monitor_risk(portfolio_data, market_data, risk_data)
        
        # Print results
        print("\nRisk Monitoring Results:")
        print("\nRisk Metrics:")
        print(result['risk_metrics'])
        
        print("\nAlerts:")
        print(result['alerts'])
        
        print("\nCustom Metrics:")
        print(result['custom_metrics'])
        
    except Exception as e:
        print(f"Error: {str(e)}") 