import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import yaml
import datetime
from src.mesh import TorusMesh
from src.config import Config
from src.plot_utils import plot_torus_with_contours_pyvista
from src.find_contours import ContourAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize torus mesh and partitions (PyVista only)')
    parser.add_argument('--input', type=str, help='Path to input YAML file with parameters')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Directory to save visualization plots (default: visualizations)')
    parser.add_argument('--no-triangulation', action='store_true', help='Hide mesh triangulation (show only surface)')
    parser.add_argument('--solution', type=str, help='Path to .h5 solution file to overlay contours (partition visualization)')
    args = parser.parse_args()

    # Load mesh parameters
    if args.input:
        print(f"\nLoading parameters from {args.input}")
        with open(args.input, 'r') as f:
            params = yaml.safe_load(f)
        config = Config(params)
    else:
        print("\nUsing default parameters from Config")
        config = Config()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"torus_mesh_nt{config.n_theta}_np{config.n_phi}_R{config.R}_r{config.r}_{timestamp}.png"
    save_path = os.path.join(args.output_dir, filename)

    # Visualization logic
    if args.solution:
        # Visualize torus with partition contours
        analyzer = ContourAnalyzer(args.solution)
        analyzer.load_results()
        contours = analyzer.extract_contours()
        mesh = analyzer.mesh
        vertices = mesh['vertices']
        faces = mesh['faces']
        plot_torus_with_contours_pyvista(vertices, faces, contours, show_edges=not args.no_triangulation, save_path=save_path)
        print(f"Partition visualization saved to: {save_path}")
    else:
        # Visualize torus mesh (with or without triangulation)
        mesh = TorusMesh(
            n_theta=config.n_theta,
            n_phi=config.n_phi,
            R=config.R,
            r=config.r
        )
        vertices = mesh.vertices
        faces = mesh.triangles
        plot_torus_with_contours_pyvista(vertices, faces, contours=None, show_edges=not args.no_triangulation, save_path=save_path)
        print(f"Mesh visualization saved to: {save_path}")
    
    