# ==========================
# risk/risk_reporting.py
# ==========================
# Risk reporting and visualization.

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from utils.logger import get_logger
from risk.risk_metrics import RiskMetricsManager
from risk.portfolio_optimizer import PortfolioOptimizer
from risk.market_regime import MarketRegimeManager
from risk.risk_allocator import RiskAllocator
from risk.performance_analyzer import PerformanceAnalyzer
from risk.correlation_analyzer import CorrelationAnalyzer
from risk.volatility_manager import VolatilityManager
from risk.drawdown_manager import DrawdownManager

logger = get_logger("risk_reporting")

class ReportType(Enum):
    """Types of risk reports."""
    DAILY = "daily"              # Daily risk report
    WEEKLY = "weekly"            # Weekly risk report
    MONTHLY = "monthly"          # Monthly risk report
    QUARTERLY = "quarterly"      # Quarterly risk report
    ANNUAL = "annual"            # Annual risk report
    ADHOC = "adhoc"             # Ad-hoc risk report
    ALERT = "alert"             # Alert-based risk report
    AUDIT = "audit"             # Audit risk report
    REGULATORY = "regulatory"    # Regulatory risk report
    EXECUTIVE = "executive"      # Executive risk report
    DETAILED = "detailed"        # Detailed risk report
    SUMMARY = "summary"          # Summary risk report
    CUSTOM = "custom"           # Custom risk report

class ReportFormat(Enum):
    """Report output formats."""
    PDF = "pdf"                 # PDF format
    HTML = "html"               # HTML format
    EXCEL = "excel"             # Excel format
    JSON = "json"               # JSON format
    CSV = "csv"                 # CSV format
    EMAIL = "email"             # Email format
    DASHBOARD = "dashboard"     # Dashboard format
    API = "api"                 # API format
    CUSTOM = "custom"           # Custom format

@dataclass
class ReportConfig:
    """Configuration for risk reporting."""
    report_type: ReportType = ReportType.DAILY
    report_format: ReportFormat = ReportFormat.PDF
    frequency: str = "1D"  # Reporting frequency
    lookback_window: int = 252  # Lookback window
    confidence_level: float = 0.95  # Confidence level
    sections: List[str] = None  # Report sections
    metrics: List[str] = None  # Report metrics
    visualizations: List[str] = None  # Report visualizations
    delivery_methods: List[str] = None  # Delivery methods
    recipients: List[str] = None  # Report recipients
    templates: Dict[str, str] = None  # Report templates
    custom_config: Dict[str, Any] = None  # Custom configuration

class RiskReporter:
    """Manages risk reporting and visualization."""
    
    def __init__(self, config: ReportConfig):
        """Initialize risk reporter."""
        self.config = config
        if self.config.sections is None:
            self.config.sections = [
                'summary', 'risk_metrics', 'portfolio_analysis',
                'market_analysis', 'alerts', 'recommendations'
            ]
        if self.config.metrics is None:
            self.config.metrics = [
                'volatility', 'var', 'es', 'sharpe', 'sortino',
                'beta', 'correlation', 'drawdown'
            ]
        if self.config.visualizations is None:
            self.config.visualizations = [
                'risk_heatmap', 'time_series', 'scatter_plot',
                'histogram', 'box_plot', 'correlation_matrix'
            ]
        if self.config.delivery_methods is None:
            self.config.delivery_methods = ['email', 'dashboard']
        self._setup_risk_reporter()
    
    def _setup_risk_reporter(self) -> None:
        """Setup risk reporter."""
        try:
            logger.info("Initializing risk reporter")
            self._validate_config()
            self._setup_templates()
            self._setup_visualizations()
        except Exception as e:
            logger.error(f"Failed to setup risk reporter: {str(e)}")
            raise
    
    def _validate_config(self) -> None:
        """Validate reporter configuration."""
        try:
            if not self.config.sections:
                raise ValueError("Report sections are required")
            if not self.config.metrics:
                raise ValueError("Report metrics are required")
            if not self.config.visualizations:
                raise ValueError("Report visualizations are required")
            if not self.config.delivery_methods:
                raise ValueError("Report delivery methods are required")
        except Exception as e:
            logger.error(f"Invalid reporter configuration: {str(e)}")
            raise
    
    def generate_report(self,
                       risk_data: Dict[str, Any],
                       positions: Optional[pd.DataFrame] = None,
                       returns: Optional[pd.DataFrame] = None,
                       market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate risk report.
        
        Args:
            risk_data: Risk analysis data
            positions: Portfolio positions (optional)
            returns: Historical returns (optional)
            market_data: Market data (optional)
            
        Returns:
            Dictionary containing report content and metadata
        """
        try:
            # Generate report sections
            sections = {}
            
            # Generate summary section
            if 'summary' in self.config.sections:
                sections['summary'] = self._generate_summary_section(risk_data)
            
            # Generate risk metrics section
            if 'risk_metrics' in self.config.sections:
                sections['risk_metrics'] = self._generate_risk_metrics_section(risk_data)
            
            # Generate portfolio analysis section
            if 'portfolio_analysis' in self.config.sections:
                sections['portfolio_analysis'] = self._generate_portfolio_analysis_section(
                    risk_data, positions, returns
                )
            
            # Generate market analysis section
            if 'market_analysis' in self.config.sections:
                sections['market_analysis'] = self._generate_market_analysis_section(
                    risk_data, market_data
                )
            
            # Generate alerts section
            if 'alerts' in self.config.sections:
                sections['alerts'] = self._generate_alerts_section(risk_data)
            
            # Generate recommendations section
            if 'recommendations' in self.config.sections:
                sections['recommendations'] = self._generate_recommendations_section(risk_data)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(risk_data, positions, returns, market_data)
            
            # Compile report
            report = self._compile_report(sections, visualizations)
            
            # Format report
            formatted_report = self._format_report(report)
            
            # Deliver report
            delivery_status = self._deliver_report(formatted_report)
            
            return {
                'report': formatted_report,
                'sections': sections,
                'visualizations': visualizations,
                'metadata': {
                    'timestamp': datetime.now(),
                    'type': self.config.report_type.value,
                    'format': self.config.report_format.value,
                    'delivery_status': delivery_status
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    def _generate_summary_section(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary section."""
        try:
            summary = {
                'timestamp': datetime.now(),
                'risk_level': risk_data.get('status', {}).get('risk_level', 'normal'),
                'alert_count': len(risk_data.get('alerts', [])),
                'action_count': len(risk_data.get('actions', [])),
                'key_metrics': {},
                'highlights': [],
                'concerns': []
            }
            
            # Add key metrics
            metrics = risk_data.get('risk_levels', {}).get('portfolio', {})
            for metric in ['volatility', 'var', 'sharpe']:
                if metric in metrics:
                    summary['key_metrics'][metric] = metrics[metric]
            
            # Add highlights and concerns
            for alert in risk_data.get('alerts', []):
                if alert.get('severity') == 'high':
                    summary['concerns'].append(alert.get('message'))
                else:
                    summary['highlights'].append(alert.get('message'))
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary section: {str(e)}")
            raise
    
    def _generate_risk_metrics_section(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk metrics section."""
        try:
            metrics = {
                'portfolio_metrics': {},
                'position_metrics': {},
                'market_metrics': {},
                'risk_decomposition': {},
                'historical_metrics': {},
                'stress_metrics': {}
            }
            
            # Add portfolio metrics
            portfolio_metrics = risk_data.get('risk_levels', {}).get('portfolio', {})
            for metric in self.config.metrics:
                if metric in portfolio_metrics:
                    metrics['portfolio_metrics'][metric] = portfolio_metrics[metric]
            
            # Add position metrics
            position_metrics = risk_data.get('risk_levels', {}).get('position', {})
            metrics['position_metrics'] = position_metrics
            
            # Add market metrics
            market_metrics = risk_data.get('risk_levels', {}).get('market', {})
            metrics['market_metrics'] = market_metrics
            
            # Add risk decomposition
            risk_decomposition = risk_data.get('risk_levels', {}).get('factor', {})
            metrics['risk_decomposition'] = risk_decomposition
            
            # Add stress metrics
            stress_metrics = risk_data.get('risk_levels', {}).get('stress', {})
            metrics['stress_metrics'] = stress_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to generate risk metrics section: {str(e)}")
            raise
    
    def _generate_portfolio_analysis_section(self,
                                          risk_data: Dict[str, Any],
                                          positions: Optional[pd.DataFrame] = None,
                                          returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate portfolio analysis section."""
        try:
            analysis = {
                'composition': {},
                'performance': {},
                'risk_contribution': {},
                'factor_exposure': {},
                'scenario_analysis': {}
            }
            
            # Add portfolio composition
            if positions is not None:
                analysis['composition'] = {
                    'weights': positions.to_dict(),
                    'concentration': self._calculate_concentration(positions)
                }
            
            # Add performance analysis
            if returns is not None:
                analysis['performance'] = {
                    'returns': returns.mean().to_dict(),
                    'volatility': returns.std().to_dict(),
                    'sharpe': self._calculate_sharpe_ratio(returns)
                }
            
            # Add risk contribution
            risk_contribution = risk_data.get('risk_levels', {}).get('portfolio', {}).get('risk_contribution', {})
            analysis['risk_contribution'] = risk_contribution
            
            # Add factor exposure
            factor_exposure = risk_data.get('risk_levels', {}).get('factor', {})
            analysis['factor_exposure'] = factor_exposure
            
            # Add scenario analysis
            scenario_analysis = risk_data.get('risk_levels', {}).get('scenario', {})
            analysis['scenario_analysis'] = scenario_analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate portfolio analysis section: {str(e)}")
            raise
    
    def _generate_market_analysis_section(self,
                                       risk_data: Dict[str, Any],
                                       market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate market analysis section."""
        try:
            analysis = {
                'market_regime': {},
                'correlation': {},
                'volatility': {},
                'liquidity': {},
                'trends': {}
            }
            
            # Add market regime analysis
            regime = risk_data.get('risk_levels', {}).get('regime', {})
            analysis['market_regime'] = regime
            
            # Add correlation analysis
            correlation = risk_data.get('risk_levels', {}).get('correlation', {})
            analysis['correlation'] = correlation
            
            # Add volatility analysis
            volatility = risk_data.get('risk_levels', {}).get('volatility', {})
            analysis['volatility'] = volatility
            
            # Add liquidity analysis
            liquidity = risk_data.get('risk_levels', {}).get('liquidity', {})
            analysis['liquidity'] = liquidity
            
            # Add market trends
            if market_data is not None:
                analysis['trends'] = self._analyze_market_trends(market_data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate market analysis section: {str(e)}")
            raise
    
    def _generate_alerts_section(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alerts section."""
        try:
            alerts = {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': [],
                'summary': {}
            }
            
            # Categorize alerts by priority
            for alert in risk_data.get('alerts', []):
                if alert.get('severity') == 'high':
                    alerts['high_priority'].append(alert)
                elif alert.get('severity') == 'medium':
                    alerts['medium_priority'].append(alert)
                else:
                    alerts['low_priority'].append(alert)
            
            # Generate alert summary
            alerts['summary'] = {
                'total_alerts': len(risk_data.get('alerts', [])),
                'high_priority_count': len(alerts['high_priority']),
                'medium_priority_count': len(alerts['medium_priority']),
                'low_priority_count': len(alerts['low_priority'])
            }
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to generate alerts section: {str(e)}")
            raise
    
    def _generate_recommendations_section(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations section."""
        try:
            recommendations = {
                'immediate_actions': [],
                'short_term': [],
                'medium_term': [],
                'long_term': [],
                'monitoring': []
            }
            
            # Generate recommendations based on risk data
            status = risk_data.get('status', {})
            if status.get('risk_level') == 'elevated':
                recommendations['immediate_actions'].extend([
                    'Review high-risk positions',
                    'Consider risk reduction strategies',
                    'Implement hedging measures'
                ])
            
            # Add monitoring recommendations
            recommendations['monitoring'].extend([
                'Continue monitoring risk metrics',
                'Track market regime changes',
                'Monitor correlation structure'
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations section: {str(e)}")
            raise
    
    def _generate_visualizations(self,
                               risk_data: Dict[str, Any],
                               positions: Optional[pd.DataFrame] = None,
                               returns: Optional[pd.DataFrame] = None,
                               market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate report visualizations."""
        try:
            visualizations = {}
            
            # Generate risk heatmap
            if 'risk_heatmap' in self.config.visualizations:
                visualizations['risk_heatmap'] = self._generate_risk_heatmap(risk_data)
            
            # Generate time series plots
            if 'time_series' in self.config.visualizations and returns is not None:
                visualizations['time_series'] = self._generate_time_series_plots(returns)
            
            # Generate scatter plots
            if 'scatter_plot' in self.config.visualizations and returns is not None:
                visualizations['scatter_plot'] = self._generate_scatter_plots(returns)
            
            # Generate histograms
            if 'histogram' in self.config.visualizations and returns is not None:
                visualizations['histogram'] = self._generate_histograms(returns)
            
            # Generate box plots
            if 'box_plot' in self.config.visualizations and returns is not None:
                visualizations['box_plot'] = self._generate_box_plots(returns)
            
            # Generate correlation matrix
            if 'correlation_matrix' in self.config.visualizations and returns is not None:
                visualizations['correlation_matrix'] = self._generate_correlation_matrix(returns)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
            raise
    
    def _compile_report(self,
                       sections: Dict[str, Any],
                       visualizations: Dict[str, Any]) -> Dict[str, Any]:
        """Compile report from sections and visualizations."""
        try:
            report = {
                'metadata': {
                    'timestamp': datetime.now(),
                    'type': self.config.report_type.value,
                    'format': self.config.report_format.value
                },
                'content': {
                    'sections': sections,
                    'visualizations': visualizations
                },
                'summary': sections.get('summary', {}),
                'appendix': {
                    'methodology': self._get_methodology(),
                    'glossary': self._get_glossary(),
                    'disclaimers': self._get_disclaimers()
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to compile report: {str(e)}")
            raise
    
    def _format_report(self, report: Dict[str, Any]) -> Any:
        """Format report according to specified format."""
        try:
            if self.config.report_format == ReportFormat.PDF:
                return self._format_pdf_report(report)
            elif self.config.report_format == ReportFormat.HTML:
                return self._format_html_report(report)
            elif self.config.report_format == ReportFormat.EXCEL:
                return self._format_excel_report(report)
            elif self.config.report_format == ReportFormat.JSON:
                return self._format_json_report(report)
            elif self.config.report_format == ReportFormat.CSV:
                return self._format_csv_report(report)
            elif self.config.report_format == ReportFormat.EMAIL:
                return self._format_email_report(report)
            elif self.config.report_format == ReportFormat.DASHBOARD:
                return self._format_dashboard_report(report)
            elif self.config.report_format == ReportFormat.API:
                return self._format_api_report(report)
            else:
                raise ValueError(f"Unsupported report format: {self.config.report_format}")
            
        except Exception as e:
            logger.error(f"Failed to format report: {str(e)}")
            raise
    
    def _deliver_report(self, report: Any) -> Dict[str, Any]:
        """Deliver report using specified delivery methods."""
        try:
            delivery_status = {}
            
            for method in self.config.delivery_methods:
                if method == 'email':
                    delivery_status['email'] = self._deliver_email_report(report)
                elif method == 'dashboard':
                    delivery_status['dashboard'] = self._deliver_dashboard_report(report)
                elif method == 'api':
                    delivery_status['api'] = self._deliver_api_report(report)
                else:
                    raise ValueError(f"Unsupported delivery method: {method}")
            
            return delivery_status
            
        except Exception as e:
            logger.error(f"Failed to deliver report: {str(e)}")
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
        
        risk_data = {
            'status': {
                'risk_level': 'elevated',
                'timestamp': datetime.now()
            },
            'alerts': [
                {
                    'severity': 'high',
                    'message': 'Portfolio volatility exceeds threshold'
                },
                {
                    'severity': 'medium',
                    'message': 'Increased correlation among assets'
                }
            ],
            'actions': [
                {
                    'type': 'reduce_risk',
                    'message': 'Reduce exposure to high-volatility assets'
                }
            ],
            'risk_levels': {
                'portfolio': {
                    'volatility': 0.25,
                    'var': 0.15,
                    'sharpe': 0.8
                }
            }
        }
        
        # Initialize reporter configuration
        config = ReportConfig(
            report_type=ReportType.DAILY,
            report_format=ReportFormat.PDF,
            frequency="1D",
            lookback_window=252,
            confidence_level=0.95,
            sections=[
                'summary', 'risk_metrics', 'portfolio_analysis',
                'market_analysis', 'alerts', 'recommendations'
            ],
            metrics=[
                'volatility', 'var', 'es', 'sharpe', 'sortino',
                'beta', 'correlation', 'drawdown'
            ],
            visualizations=[
                'risk_heatmap', 'time_series', 'scatter_plot',
                'histogram', 'box_plot', 'correlation_matrix'
            ],
            delivery_methods=['email', 'dashboard'],
            recipients=['risk@example.com', 'portfolio@example.com']
        )
        
        # Initialize risk reporter
        reporter = RiskReporter(config)
        
        # Generate report
        result = reporter.generate_report(risk_data, positions, returns, market_data)
        
        # Print results
        print("\nRisk Report Results:")
        print("\nReport Metadata:")
        print(result['metadata'])
        
        print("\nReport Sections:")
        print(result['sections'])
        
        print("\nReport Visualizations:")
        print(result['visualizations'])
        
    except Exception as e:
        print(f"Error: {str(e)}") 