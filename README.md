# Torus Partition Visualization

This project provides tools for creating and visualizing partitions of a torus surface. It includes:

- A mesh generation module for creating torus surfaces
- A visualization module for displaying 3D and 2D representations of partitions
- A main script that demonstrates the functionality

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to see a demonstration:
```bash
python src/main.py
```

This will create a torus mesh, generate a random partition, and display both 3D and 2D visualizations.

## Project Structure

- `src/mesh.py`: Contains the `TorusMesh` class for generating and managing torus meshes
- `src/visualization.py`: Contains the `PartitionVisualizer` class for creating visualizations
- `src/main.py`: Main script demonstrating the functionality
- `requirements.txt`: Project dependencies

## License

MIT License 