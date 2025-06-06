import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def plot_torus_with_contours_pyvista(vertices, faces, contours, show_edges=True, edge_color='white', opacity=1.0, edge_line_width=1, save_path=None):
    """
    Plot a torus mesh with triangulation and overlay contours using PyVista.
    Optionally save a screenshot to save_path.
    """
    def faces_to_pyvista(faces):
        return np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).flatten()
    faces_pv = faces_to_pyvista(faces)
    mesh = pv.PolyData(vertices, faces_pv)
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        color='lightgray',
        show_edges=show_edges,
        edge_color=edge_color,
        opacity=opacity,
        line_width=edge_line_width
    )
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    if contours is not None:
        for region_idx, segments in contours.items():
            color = color_list[region_idx % len(color_list)]
            for seg in segments:
                line = pv.Line(seg[0], seg[1])
                plotter.add_mesh(line, color=color, line_width=4)
    plotter.show()
    if save_path is not None:
        plotter.screenshot(save_path) 