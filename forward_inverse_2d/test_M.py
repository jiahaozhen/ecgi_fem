from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities
from dolfinx.plot import vtk_mesh
import numpy as np
import pyvista
from mpi4py import MPI
import ufl

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain.topology.index_map(tdim-2).size_local
def rho1(x):
    tensor = np.eye(2)
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho2(x):
    tensor = np.eye(2) * 2
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart 
#      M0 in Torso
V = functionspace(domain, ("DG", 0, (2,2)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))

# grid0 = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
# grid0.cell_data['M'] = M_array
# grid0.set_active_scalars("M")

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid0, show_edges=True)
# plotter.show()