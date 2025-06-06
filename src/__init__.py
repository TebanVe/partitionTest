"""
Manifold Partition package for partitioning manifolds.
"""

__version__ = "0.1.0"

from .mesh import TorusMesh
from .slsqp_optimizer import SLSQPOptimizer
from .config import Config

__all__ = ['TorusMesh', 'SLSQPOptimizer', 'Config']

# This file makes the src directory a Python package 