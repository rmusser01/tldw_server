# __init__.py
# Prompt Studio module initialization

"""
Prompt Studio - A structured prompt engineering platform for tldw_server

This module provides comprehensive prompt engineering capabilities including:
- Project and prompt management with versioning
- Test case creation and management
- Automated testing and evaluation
- Prompt optimization using various strategies
- Real-time job processing and monitoring
"""

# Core managers
from .bootstrap_manager import BootstrapManager
from .evaluation_manager import EvaluationManager
from .evaluation_metrics import EvaluationMetrics
from .evaluation_reports import EvaluationReportGenerator

# Event handling and monitoring
from .event_broadcaster import EventBroadcaster, EventType
from .job_processor import JobProcessor
from .job_types import JobStatus, JobType
from .monitoring import PromptStudioMetrics

# Optimization
from .optimization_engine import OptimizationEngine
from .optimization_strategies import HyperparameterOptimizer
from .prompt_executor import PromptExecutor

# Prompt generation and improvement
from .prompt_generator import PromptGenerator
from .prompt_improver import PromptImprover
from .test_case_generator import TestCaseGenerator
from .test_case_io import TestCaseIO
from .test_case_manager import TestCaseManager

# Testing and evaluation
from .test_runner import TestRunner

__all__ = [
    # Core managers
    'TestCaseManager',
    'TestCaseIO',
    'TestCaseGenerator',
    'JobType',
    'JobStatus',
    'JobProcessor',

    # Prompt generation and improvement
    'PromptGenerator',
    'PromptImprover',
    'BootstrapManager',

    # Testing and evaluation
    'TestRunner',
    'PromptExecutor',
    'EvaluationMetrics',
    'EvaluationManager',
    'EvaluationReportGenerator',

    # Optimization
    'OptimizationEngine',
    'HyperparameterOptimizer',

    # Event handling and monitoring
    'EventBroadcaster',
    'EventType',
    'PromptStudioMetrics',

]

__version__ = '0.1.0'
__author__ = 'tldw_server Development Team'
