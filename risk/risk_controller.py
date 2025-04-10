# ==========================
# risk/risk_controller.py
# ==========================
# Central risk management controller.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from utils.logger import get_logger
from risk.risk_metrics import RiskMetricsManager
from risk.portfolio_optimizer import PortfolioOptimizer
from risk.market_regime import MarketRegimeManager
from risk.risk_allocator import RiskAllocator
from risk.performance_analyzer import PerformanceAnalyzer
from risk.correlation_analyzer import CorrelationAnalyzer
from risk.volatility_manager import VolatilityManager
from risk.drawdown_manager import DrawdownManager

logger = get_logger("risk_controller")

class RiskControlType(Enum):
    """Types of risk control."""
    POSITION = "position"          # Position-level risk control
    PORTFOLIO = "portfolio"        # Portfolio-level risk control
    MARKET = "market"             # Market-level risk control
    LIQUIDITY = "liquidity"       # Liquidity risk control
    LEVERAGE = "leverage"         # Leverage risk control
    MARGIN = "margin"             # Margin risk control
    EXPOSURE = "exposure"         # Exposure risk control
    REGIME = "regime"             # Regime-based risk control
    CORRELATION = "correlation"   # Correlation-based risk control
    VOLATILITY = "volatility"     # Volatility-based risk control
    DRAWDOWN = "drawdown"         # Drawdown-based risk control
    CONCENTRATION = "concentration"  # Concentration risk control
    FACTOR = "factor"             # Factor risk control
    STRESS = "stress"             # Stress test control
    SCENARIO = "scenario"         # Scenario analysis control
    CUSTOM = "custom"             # Custom risk control

class RiskAction(Enum):
    """Risk control actions."""
    MONITOR = "monitor"           # Monitor risk levels
    ALERT = "alert"              # Generate risk alerts
    REDUCE = "reduce"            # Reduce risk exposure
    INCREASE = "increase"        # Increase risk exposure
    HEDGE = "hedge"              # Apply hedging strategies
    REBALANCE = "rebalance"      # Rebalance portfolio
    LIQUIDATE = "liquidate"      # Liquidate positions
    STOP_TRADING = "stop_trading"  # Stop trading
    CUSTOM = "custom"             # Custom action

@dataclass
class RiskControlConfig:
    """Configuration for risk control."""
    control_type: RiskControlType = RiskControlType.PORTFOLIO
    risk_limits: Dict[str, float] = None  # Risk limits
    alert_thresholds: Dict[str, float] = None  # Alert thresholds
    action_thresholds: Dict[str, float] = None  # Action thresholds
    monitoring_frequency: str = "1D"  # Monitoring frequency
    lookback_window: int = 252  # Lookback window
    confidence_level: float = 0.95  # Confidence level
    position_limits: Dict[str, float] = None  # Position limits
    portfolio_limits: Dict[str, float] = None  # Portfolio limits
    leverage_limits: Dict[str, float] = None  # Leverage limits
    margin_requirements: Dict[str, float] = None  # Margin requirements
    concentration_limits: Dict[str, float] = None  # Concentration limits
    factor_limits: Dict[str, float] = None  # Factor exposure limits
    stress_scenarios: Dict[str, Dict[str, float]] = None  # Stress scenarios
    custom_limits: Dict[str, Any] = None  # Custom limits

@dataclass
class ControllerConfig:
    """Configuration for risk controller."""
    risk_controls: Dict[RiskControlType, RiskControlConfig] = None  # Risk control configurations
    risk_actions: Dict[RiskAction, Dict[str, Any]] = None  # Risk action configurations
    monitoring_config: Dict[str, Any] = None  # Monitoring configuration
    alert_config: Dict[str, Any] = None  # Alert configuration
    reporting_config: Dict[str, Any] = None  # Reporting configuration
    integration_config: Dict[str, Any] = None  # Integration configuration
    custom_config: Dict[str, Any] = None  # Custom configuration

class RiskController:
    """Central risk management controller."""
    
    def __init__(self, config: ControllerConfig):
        """Initialize risk controller."""
        self.config = config
        self._setup_risk_controller()
        
    def _setup_risk_controller(self) -> None:
        """Setup risk controller components."""
        try:
            logger.info("Initializing risk controller")
            self._validate_config()
            self._setup_components()
            self._setup_risk_controls()
            self._setup_risk_actions()
            self._setup_monitoring()
            self._setup_alerts()
            self._setup_reporting()
            self._setup_integration()
        except Exception as e:
            logger.error(f"Failed to setup risk controller: {str(e)}")
            raise
            
    def _validate_config(self) -> None:
        """Validate controller configuration."""
        try:
            if not self.config.risk_controls:
                raise ValueError("Risk controls configuration is required")
            if not self.config.risk_actions:
                raise ValueError("Risk actions configuration is required")
            self._validate_risk_controls()
            self._validate_risk_actions()
            self._validate_monitoring_config()
            self._validate_alert_config()
            self._validate_reporting_config()
            self._validate_integration_config()
        except Exception as e:
            logger.error(f"Invalid controller configuration: {str(e)}")
            raise
            
    def _setup_components(self) -> None:
        """Setup risk management components."""
        try:
            # Initialize risk management components
            self.risk_metrics = RiskMetricsManager(self.config.integration_config.get('risk_metrics'))
            self.portfolio_optimizer = PortfolioOptimizer(self.config.integration_config.get('portfolio_optimizer'))
            self.market_regime = MarketRegimeManager(self.config.integration_config.get('market_regime'))
            self.risk_allocator = RiskAllocator(self.config.integration_config.get('risk_allocator'))
            self.performance_analyzer = PerformanceAnalyzer(self.config.integration_config.get('performance_analyzer'))
            self.correlation_analyzer = CorrelationAnalyzer(self.config.integration_config.get('correlation_analyzer'))
            self.volatility_manager = VolatilityManager(self.config.integration_config.get('volatility_manager'))
            self.drawdown_manager = DrawdownManager(self.config.integration_config.get('drawdown_manager'))
        except Exception as e:
            logger.error(f"Failed to setup components: {str(e)}")
            raise
            
    def control_risk(self,
                    positions: pd.DataFrame,
                    returns: pd.DataFrame,
                    market_data: Optional[pd.DataFrame] = None,
                    risk_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Control portfolio risk.
        
        Args:
            positions: Current portfolio positions
            returns: Historical returns
            market_data: Market data (optional)
            risk_data: Additional risk data (optional)
            
        Returns:
            Dictionary containing risk control results
        """
        try:
            # Monitor risk levels
            risk_levels = self._monitor_risk_levels(positions, returns, market_data, risk_data)
            
            # Check risk limits
            limit_breaches = self._check_risk_limits(risk_levels)
            
            # Generate alerts
            alerts = self._generate_alerts(limit_breaches)
            
            # Determine required actions
            actions = self._determine_actions(limit_breaches, alerts)
            
            # Execute risk control actions
            results = self._execute_actions(actions, positions, returns, market_data, risk_data)
            
            # Update risk status
            status = self._update_risk_status(results)
            
            # Generate report
            report = self._generate_report(status)
            
            return {
                'risk_levels': risk_levels,
                'limit_breaches': limit_breaches,
                'alerts': alerts,
                'actions': actions,
                'results': results,
                'status': status,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Failed to control risk: {str(e)}")
            raise
            
    def _monitor_risk_levels(self,
                           positions: pd.DataFrame,
                           returns: pd.DataFrame,
                           market_data: Optional[pd.DataFrame] = None,
                           risk_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Monitor risk levels."""
        try:
            risk_levels = {}
            
            # Monitor position-level risk
            risk_levels['position'] = self._monitor_position_risk(positions, returns)
            
            # Monitor portfolio-level risk
            risk_levels['portfolio'] = self._monitor_portfolio_risk(positions, returns)
            
            # Monitor market risk
            risk_levels['market'] = self._monitor_market_risk(market_data)
            
            # Monitor liquidity risk
            risk_levels['liquidity'] = self._monitor_liquidity_risk(positions, market_data)
            
            # Monitor leverage risk
            risk_levels['leverage'] = self._monitor_leverage_risk(positions)
            
            # Monitor margin risk
            risk_levels['margin'] = self._monitor_margin_risk(positions, market_data)
            
            # Monitor exposure risk
            risk_levels['exposure'] = self._monitor_exposure_risk(positions)
            
            # Monitor regime risk
            risk_levels['regime'] = self._monitor_regime_risk(returns, market_data)
            
            # Monitor correlation risk
            risk_levels['correlation'] = self._monitor_correlation_risk(returns)
            
            # Monitor volatility risk
            risk_levels['volatility'] = self._monitor_volatility_risk(returns)
            
            # Monitor drawdown risk
            risk_levels['drawdown'] = self._monitor_drawdown_risk(returns)
            
            # Monitor concentration risk
            risk_levels['concentration'] = self._monitor_concentration_risk(positions)
            
            # Monitor factor risk
            risk_levels['factor'] = self._monitor_factor_risk(positions, returns, market_data)
            
            # Monitor stress test results
            risk_levels['stress'] = self._monitor_stress_test(positions, market_data)
            
            # Monitor scenario analysis
            risk_levels['scenario'] = self._monitor_scenario_analysis(positions, market_data)
            
            return risk_levels
            
        except Exception as e:
            logger.error(f"Failed to monitor risk levels: {str(e)}")
            raise
            
    def _check_risk_limits(self, risk_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Check risk limits."""
        try:
            limit_breaches = {}
            
            for control_type in RiskControlType:
                if control_type.value in risk_levels:
                    control_config = self.config.risk_controls.get(control_type)
                    if control_config and control_config.risk_limits:
                        breaches = {}
                        for metric, limit in control_config.risk_limits.items():
                            if metric in risk_levels[control_type.value]:
                                value = risk_levels[control_type.value][metric]
                                if value > limit:
                                    breaches[metric] = {
                                        'limit': limit,
                                        'value': value,
                                        'excess': value - limit
                                    }
                        if breaches:
                            limit_breaches[control_type.value] = breaches
            
            return limit_breaches
            
        except Exception as e:
            logger.error(f"Failed to check risk limits: {str(e)}")
            raise
            
    def _generate_alerts(self, limit_breaches: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk alerts."""
        try:
            alerts = {}
            
            for control_type, breaches in limit_breaches.items():
                control_config = self.config.risk_controls.get(RiskControlType(control_type))
                if control_config and control_config.alert_thresholds:
                    control_alerts = {}
                    for metric, breach in breaches.items():
                        threshold = control_config.alert_thresholds.get(metric)
                        if threshold and breach['excess'] > threshold:
                            control_alerts[metric] = {
                                'severity': 'high' if breach['excess'] > 2 * threshold else 'medium',
                                'message': f"{metric} exceeds limit by {breach['excess']:.2f}",
                                'breach': breach
                            }
                    if control_alerts:
                        alerts[control_type] = control_alerts
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to generate alerts: {str(e)}")
            raise
            
    def _determine_actions(self,
                         limit_breaches: Dict[str, Any],
                         alerts: Dict[str, Any]) -> Dict[str, Any]:
        """Determine required risk actions."""
        try:
            actions = {}
            
            for control_type, breaches in limit_breaches.items():
                control_config = self.config.risk_controls.get(RiskControlType(control_type))
                if control_config and control_config.action_thresholds:
                    control_actions = {}
                    for metric, breach in breaches.items():
                        threshold = control_config.action_thresholds.get(metric)
                        if threshold and breach['excess'] > threshold:
                            control_actions[metric] = {
                                'action': RiskAction.REDUCE.value,
                                'magnitude': min(1.0, breach['excess'] / threshold),
                                'breach': breach
                            }
                    if control_actions:
                        actions[control_type] = control_actions
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to determine actions: {str(e)}")
            raise
            
    def _execute_actions(self,
                        actions: Dict[str, Any],
                        positions: pd.DataFrame,
                        returns: pd.DataFrame,
                        market_data: Optional[pd.DataFrame] = None,
                        risk_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute risk control actions."""
        try:
            results = {}
            
            for control_type, control_actions in actions.items():
                control_config = self.config.risk_controls.get(RiskControlType(control_type))
                if control_config:
                    control_results = {}
                    for metric, action in control_actions.items():
                        if action['action'] == RiskAction.REDUCE.value:
                            control_results[metric] = self._reduce_risk(
                                control_type,
                                metric,
                                action['magnitude'],
                                positions,
                                returns,
                                market_data,
                                risk_data
                            )
                    if control_results:
                        results[control_type] = control_results
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute actions: {str(e)}")
            raise
            
    def _reduce_risk(self,
                    control_type: str,
                    metric: str,
                    magnitude: float,
                    positions: pd.DataFrame,
                    returns: pd.DataFrame,
                    market_data: Optional[pd.DataFrame] = None,
                    risk_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reduce risk exposure."""
        try:
            result = {
                'control_type': control_type,
                'metric': metric,
                'magnitude': magnitude,
                'status': 'pending'
            }
            
            # Implement risk reduction logic based on control type
            if control_type == RiskControlType.POSITION.value:
                result.update(self._reduce_position_risk(metric, magnitude, positions))
            elif control_type == RiskControlType.PORTFOLIO.value:
                result.update(self._reduce_portfolio_risk(metric, magnitude, positions, returns))
            elif control_type == RiskControlType.MARKET.value:
                result.update(self._reduce_market_risk(metric, magnitude, positions, market_data))
            elif control_type == RiskControlType.LIQUIDITY.value:
                result.update(self._reduce_liquidity_risk(metric, magnitude, positions, market_data))
            elif control_type == RiskControlType.LEVERAGE.value:
                result.update(self._reduce_leverage_risk(metric, magnitude, positions))
            elif control_type == RiskControlType.MARGIN.value:
                result.update(self._reduce_margin_risk(metric, magnitude, positions, market_data))
            elif control_type == RiskControlType.EXPOSURE.value:
                result.update(self._reduce_exposure_risk(metric, magnitude, positions))
            elif control_type == RiskControlType.REGIME.value:
                result.update(self._reduce_regime_risk(metric, magnitude, positions, returns, market_data))
            elif control_type == RiskControlType.CORRELATION.value:
                result.update(self._reduce_correlation_risk(metric, magnitude, positions, returns))
            elif control_type == RiskControlType.VOLATILITY.value:
                result.update(self._reduce_volatility_risk(metric, magnitude, positions, returns))
            elif control_type == RiskControlType.DRAWDOWN.value:
                result.update(self._reduce_drawdown_risk(metric, magnitude, positions, returns))
            elif control_type == RiskControlType.CONCENTRATION.value:
                result.update(self._reduce_concentration_risk(metric, magnitude, positions))
            elif control_type == RiskControlType.FACTOR.value:
                result.update(self._reduce_factor_risk(metric, magnitude, positions, returns, market_data))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to reduce risk: {str(e)}")
            raise
            
    def _update_risk_status(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Update risk status."""
        try:
            status = {
                'timestamp': datetime.now(),
                'risk_level': 'normal',
                'alerts': [],
                'actions': [],
                'metrics': {}
            }
            
            # Update status based on results
            for control_type, control_results in results.items():
                for metric, result in control_results.items():
                    if result['status'] == 'completed':
                        status['metrics'][f"{control_type}_{metric}"] = result['final_value']
                        if result['final_value'] > result['target_value']:
                            status['risk_level'] = 'elevated'
                            status['alerts'].append({
                                'control_type': control_type,
                                'metric': metric,
                                'message': f"{metric} remains above target after risk reduction"
                            })
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to update risk status: {str(e)}")
            raise
            
    def _generate_report(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk report."""
        try:
            report = {
                'timestamp': status['timestamp'],
                'summary': {
                    'risk_level': status['risk_level'],
                    'alert_count': len(status['alerts']),
                    'action_count': len(status['actions'])
                },
                'alerts': status['alerts'],
                'actions': status['actions'],
                'metrics': status['metrics'],
                'recommendations': []
            }
            
            # Generate recommendations
            if status['risk_level'] == 'elevated':
                report['recommendations'].append({
                    'priority': 'high',
                    'message': 'Review portfolio risk exposure and consider additional risk reduction'
                })
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        n_assets = 5
        
        positions = pd.DataFrame(
            np.random.uniform(0, 1, (1, n_assets)),
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (252, n_assets)),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        market_data = pd.DataFrame(
            np.random.normal(0.0003, 0.015, (252, n_assets)),
            index=dates,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        # Initialize controller configuration
        controller_config = ControllerConfig(
            risk_controls={
                RiskControlType.PORTFOLIO: RiskControlConfig(
                    control_type=RiskControlType.PORTFOLIO,
                    risk_limits={'volatility': 0.2, 'var': 0.1},
                    alert_thresholds={'volatility': 0.22, 'var': 0.12},
                    action_thresholds={'volatility': 0.25, 'var': 0.15}
                )
            },
            risk_actions={
                RiskAction.REDUCE: {
                    'max_reduction': 0.5,
                    'min_holding': 0.1
                }
            },
            monitoring_config={
                'frequency': '1D',
                'metrics': ['volatility', 'var', 'sharpe']
            },
            alert_config={
                'email': True,
                'slack': False,
                'threshold': 0.1
            },
            reporting_config={
                'frequency': '1D',
                'format': 'pdf',
                'delivery': ['email']
            },
            integration_config={
                'risk_metrics': {},
                'portfolio_optimizer': {},
                'market_regime': {},
                'risk_allocator': {},
                'performance_analyzer': {},
                'correlation_analyzer': {},
                'volatility_manager': {},
                'drawdown_manager': {}
            }
        )
        
        # Initialize risk controller
        controller = RiskController(controller_config)
        
        # Control risk
        result = controller.control_risk(positions, returns, market_data)
        
        # Print results
        print("\nRisk Control Results:")
        print("\nRisk Levels:")
        print(result['risk_levels'])
        
        print("\nLimit Breaches:")
        print(result['limit_breaches'])
        
        print("\nAlerts:")
        print(result['alerts'])
        
        print("\nActions:")
        print(result['actions'])
        
        print("\nResults:")
        print(result['results'])
        
        print("\nStatus:")
        print(result['status'])
        
        print("\nReport:")
        print(result['report'])
        
    except Exception as e:
        print(f"Error: {str(e)}") 