from abc import ABC, abstractmethod
from typing import Dict, List, Any

class AlgorithmBase(ABC):
    @abstractmethod
    def get_default_config(self) -> Dict:
        """Get default configuration for the algorithm."""
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get algorithm version."""
        pass

    @abstractmethod
    def run(self, api: 'AnalysisAPI', data: Dict[str, Dict[str, float]], 
            interval: str, config: Dict, 
            support_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the algorithm with given parameters."""
        pass 