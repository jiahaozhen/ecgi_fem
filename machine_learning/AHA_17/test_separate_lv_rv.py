from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista
from utils.ventricular_segmentation_tools import separate_lv_rv

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3

lv_points, rv_points, _, _ = separate_lv_rv(mesh_file, gdim=gdim)

domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
plotter.add_mesh(grid, show_edges=True)

plotter.add_points(lv_points, color='red', point_size=10, render_points_as_spheres=True)
plotter.add_points(rv_points, color='green', point_size=10, render_points_as_spheres=True)

plotter.view_yz()
plotter.add_axes()
plotter.show()
