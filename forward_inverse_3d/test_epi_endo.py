# 检验内外膜划分是否正确
import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
import pyvista

sys.path.append('.')
from utils.helper_function import distinguish_epi_endo

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim

subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

print("Epi:", len(epi_endo_marker[epi_endo_marker == 1]), "Mid:", len(epi_endo_marker[epi_endo_marker == 0]), "Endo:", len(epi_endo_marker[epi_endo_marker == -1]))

grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid.point_data["f"] = epi_endo_marker
grid.set_active_scalars("f")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.add_title("epi mid endo")
plotter.view_yz()
plotter.add_axes()
plotter.show()