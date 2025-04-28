"""
Manifold Partition package for partitioning manifolds.
"""

__version__ = "0.1.0"

from .mesh import TorusMesh
from .optimization import PartitionOptimizer
from .visualization import PartitionVisualizer

__all__ = ['TorusMesh', 'PartitionOptimizer', 'PartitionVisualizer']

# This file makes the src directory a Python package 