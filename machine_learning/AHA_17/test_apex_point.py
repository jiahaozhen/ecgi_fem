from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista
from utils.ventricular_segmentation_tools import get_ring_pts, get_apex_from_annulus_pts

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

points = subdomain_ventricle.geometry.x
left_ring_index, ring_ring_index, left_ring_pts, right_ring_pts = get_ring_pts(mesh_file, gdim=gdim)
apex_point = get_apex_from_annulus_pts(points, left_ring_pts)

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
plotter.add_mesh(grid, show_edges=True)

plotter.add_points(left_ring_pts, color='red', point_size=10, render_points_as_spheres=True)

plotter.add_points(apex_point, color='red', point_size=10, render_points_as_spheres=True)

plotter.view_yz()
plotter.add_axes()
plotter.show()
