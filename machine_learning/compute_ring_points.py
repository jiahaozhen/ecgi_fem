import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista

sys.path.append('.')
from utils.helper_function import get_ring_pts, distinguish_ring_pts

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

ring_pts = get_ring_pts(mesh_file, gdim)

left_ventricle, right_ventricle = distinguish_ring_pts(ring_pts)

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
plotter.add_mesh(grid, show_edges=True)
# plotter.add_points(ring_points, color='red', point_size=10, render_points_as_spheres=True)

plotter.add_points(left_ventricle, color='red', point_size=10, render_points_as_spheres=True)
plotter.add_points(right_ventricle, color='green', point_size=10, render_points_as_spheres=True)

plotter.view_yz()
plotter.add_axes()
plotter.show()
