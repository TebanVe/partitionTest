import sys
import os
import numpy as np
import datetime
import h5py
from pathlib import Path
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Dict, List, Tuple
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import Config

class ContourAnalyzer:
    """Class to analyze and visualize contours from optimization results using indicator functions."""
    
    def __init__(self, result_path: str):
        """
        Initialize the contour analyzer.
        
        Args:
            result_path: Path to the result directory containing the .h5 file
        """
        self.result_path = Path(result_path)
        self.h5_file = None
        self.x_opt = None
        self.density_functions = None
        self.mesh = None
        self.level = 0.5  # Default level for contour extraction
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_results(self) -> None:
        """Load the optimization results from the HDF5 file."""
        try:
            # First check if the path is a directory
            if not self.result_path.is_dir():
                raise NotADirectoryError(f"Path {self.result_path} is not a directory")
                
            # Look for .h5 files in the directory
            h5_files = list(self.result_path.glob("*.h5"))
            if not h5_files:
                available_files = list(self.result_path.glob("*"))
                self.logger.error(f"No .h5 file found in {self.result_path}")
                self.logger.error(f"Available files: {[f.name for f in available_files]}")
                raise FileNotFoundError(f"No .h5 file found in {self.result_path}. Available files: {[f.name for f in available_files]}")
            
            h5_path = h5_files[0]  # Take the first .h5 file found
            self.logger.info(f"Loading results from {h5_path}")
            
            with h5py.File(h5_path, 'r') as f:
                # Load solution vector
                self.x_opt = f['x_opt'][:]
                # Load mesh information
                self.mesh = {
                    'vertices': f['vertices'][:],
                    'faces': f['faces'][:]
                }
                
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            raise
        
        # Reshape x_opt into density functions
        n_vertices = len(self.mesh['vertices'])
        n_partitions = self.x_opt.shape[0] // n_vertices
        self.density_functions = self.x_opt.reshape(n_vertices, n_partitions)
            
        self.logger.info(f"Loaded solution with {n_partitions} partitions and {n_vertices} vertices")
    
    def compute_indicator_functions(self) -> np.ndarray:
        """
        Compute indicator functions χ_i from density functions using winner-takes-all.
        
        Returns:
            Indicator functions of shape (n_vertices, n_regions)
        """
        if self.density_functions is None:
            raise ValueError("Results must be loaded before computing indicator functions")
            
        n_vertices, n_regions = self.density_functions.shape
        chi = np.zeros_like(self.density_functions)
        
        # Winner-takes-all assignment
        max_indices = np.argmax(self.density_functions, axis=1)
        for i in range(n_vertices):
            chi[i, max_indices[i]] = 1.0
        
        return chi
    
    def _find_intersections(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                          d1: float, d2: float, d3: float) -> List[np.ndarray]:
        """
        Find intersection points of the level set with triangle edges.
        
        Args:
            p1, p2, p3: Triangle vertices
            d1, d2, d3: Density values at vertices
            
        Returns:
            List of contour segments
        """
        segments = []
        
        # Check each edge
        if (d1 > self.level) != (d2 > self.level):
            t = (self.level - d1) / (d2 - d1)
            point = p1 + t * (p2 - p1)
            segments.append(point)
            
        if (d2 > self.level) != (d3 > self.level):
            t = (self.level - d2) / (d3 - d2)
            point = p2 + t * (p3 - p2)
            segments.append(point)
            
        if (d3 > self.level) != (d1 > self.level):
            t = (self.level - d3) / (d1 - d3)
            point = p3 + t * (p1 - p3)
            segments.append(point)
            
        # If we found two points, create a segment
        if len(segments) == 2:
            return [np.array([segments[0], segments[1]])]
        return []
    
    def extract_contours(self, level: float = 0.5) -> Dict[int, List[np.ndarray]]:
        """
        Extract contours using indicator functions.
        
        Args:
            level: Level set value (default: 0.5)
            
        Returns:
            Dictionary mapping region index to list of contour segments
        """
        if self.density_functions is None:
            raise ValueError("Results must be loaded before extracting contours")
            
        self.level = level
        chi = self.compute_indicator_functions()
        contours = {}
        n_regions = chi.shape[1]
        
        for region_idx in range(n_regions):
            chi_region = chi[:, region_idx]
            region_contours = []
            
            for face in self.mesh['faces']:
                v1, v2, v3 = face
                d1, d2, d3 = chi_region[v1], chi_region[v2], chi_region[v3]
                
                if (d1 > level) != (d2 > level) or (d2 > level) != (d3 > level) or (d3 > level) != (d1 > level):
                    p1 = self.mesh['vertices'][v1]
                    p2 = self.mesh['vertices'][v2]
                    p3 = self.mesh['vertices'][v3]
                    
                    segments = self._find_intersections(p1, p2, p3, d1, d2, d3)
                    if segments:
                        region_contours.extend(segments)
            
            contours[region_idx] = region_contours
            self.logger.info(f"Region {region_idx}: extracted {len(region_contours)} contour segments")
        
        return contours