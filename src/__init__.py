"""
Manifold Partition package for partitioning manifolds.
"""

__version__ = "0.1.0"

from .mesh import TorusMesh
from .slsqp_optimizer import SLSQPOptimizer
from .config import Config
from .find_contours import ContourAnalyzer
from .plot_utils import plot_torus_with_contours_pyvista

__all__ = ['TorusMesh', 'SLSQPOptimizer', 'Config', 'ContourAnalyzer', 'plot_torus_with_contours_pyvista']

# This file makes the src directory a Python package 