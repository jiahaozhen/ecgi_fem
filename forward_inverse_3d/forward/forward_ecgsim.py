from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh
import numpy as np
from mpi4py import MPI
import h5py
from utils.function_tools import eval_function

def load_geometry_data(file_path):
    """Load geometry data from an HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as geom_data:
            v_pts_ecgsim = np.array(geom_data['geom_ventricle']['pts'])
            forward_matrix = np.array(geom_data['ventricles2thorax'])
        return v_pts_ecgsim, forward_matrix
    except Exception as e:
        raise RuntimeError(f"Failed to load geometry data: {e}")

def create_function_space(mesh, cell_markers, marker_value):
    """Create a function space for a specific subdomain."""
    tdim = mesh.topology.dim
    subdomain, _, _, _ = create_submesh(mesh, tdim, cell_markers.find(marker_value))
    return functionspace(subdomain, ("Lagrange", 1))

def compute_surface_potentials(v_data, function_space, eval_points):
    """Compute surface potentials for given data and evaluation points."""
    v = Function(function_space)
    v_data_ecgsim = []

    for data in v_data:
        v.x.array[:] = data
        v_surface = eval_function(v, points=eval_points).reshape(-1)
        v_data_ecgsim.append(v_surface.copy())

    return np.array(v_data_ecgsim)

def forward_tmp(mesh_file, v_data):
    """Perform forward computation to map heart potentials to body potentials."""
    # Load mesh and markers
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)

    # Load geometry data
    geom_file = 'forward_inverse_3d/data/geom_ecgsim.mat'
    v_pts_ecgsim, forward_matrix = load_geometry_data(geom_file)

    # Create function space for the ventricle subdomain
    V = create_function_space(domain, cell_markers, marker_value=2)

    # Compute surface potentials
    v_data_ecgsim = compute_surface_potentials(v_data, V, v_pts_ecgsim)

    # Map surface potentials to body potentials
    d_data_ecgsim = v_data_ecgsim @ forward_matrix

    return d_data_ecgsim

def compute_d_from_tmp(mesh_file, v_data):
    """Wrapper function to compute body potentials from heart potentials."""
    return forward_tmp(mesh_file, v_data)