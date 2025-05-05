"""
Manifold Partition package for partitioning manifolds.
"""

__version__ = "0.1.0"

from .mesh import TorusMesh
from .slsqp_optimizer import SLSQPOptimizer
from .slsqp_optimizer_analytic import SLSQPOptimizerAnalytic
from .visualization import PartitionVisualizer

__all__ = ['TorusMesh', 'SLSQPOptimizer', 'SLSQPOptimizerAnalytic', 'PartitionVisualizer']

# This file makes the src directory a Python package 