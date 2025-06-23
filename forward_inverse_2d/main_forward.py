# compute potential u in B from transmembrane potential v
from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from helper_function import delta_tau
from mpi4py import MPI
from petsc4py import PETSc
import pyvista

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("2d/heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

num_node = domain.topology.index_map(tdim-2).size_local
num_cell = domain.topology.index_map(tdim).size_local

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart
#      M0 in Torso
def rho1(x):
    tensor = np.eye(2) * 2
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho2(x):
    tensor = np.eye(2)
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
V = functionspace(domain, ("DG", 0, (2,2)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))

# M = Constant(domain, default_scalar_type(np.eye(tdim)))
Mi = Constant(subdomain, default_scalar_type(np.eye(tdim)))

# A_u u = b_u
# matrix A
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx",domain=domain)
a = dot(grad(u1), dot(M, grad(v1)))*dx1
bilinear_form_a = form(a)
A = assemble_matrix(bilinear_form_a)
A.assemble()

# phi exact
class exact_solution():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
    def __call__(self, x):
        dist = (x[0]-self.x)**2 + (x[1]-self.y)**2
        return np.sqrt(dist) - self.r

# preparation 
a1 = -60
a2 = -90
tau = 0.3
center = [4, 4]
r = 1
phi = Function(V2)
phi.interpolate(exact_solution(center[0], center[1], r))
# phi.x.array[:] = np.sqrt((subdomain.geometry.x[:,0]-center[0])**2 + (subdomain.geometry.x[:,1]-center[1])**2) - r
delta_phi = Function(V2)
delta_phi.x.array[:] = delta_tau(phi.x.array, tau)

# matrix B
dx2 = Measure("dx",domain=subdomain)
b = (a1-a2) * delta_phi * dot(grad(u1), dot(Mi, grad(phi)))*dx2
entity_map = {domain._cpp_object: sub_to_parent}
linear_form_b = form(b, entity_maps=entity_map)
B = assemble_vector(linear_form_b)
B.assemble()

u = Function(V1)

solver = PETSc.KSP().create()
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.solve(B, u.vector)

np.save('2d/bsp_one_timeframe.npy', u.x.array)
np.save('2d/phi_exact.npy', phi.x.array)

marker = np.zeros(subdomain.topology.index_map(tdim-2).size_local)
marker[phi.x.array < 0] = 1

grid1 = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
grid1.point_data["u"] = u.x.array
grid1.set_active_scalars("u")
grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid2.point_data["phi"] = phi.x.array
grid2.set_active_scalars("phi")
grid3 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
grid3.point_data["marker"] = marker
grid3.set_active_scalars("marker")
grid = (grid1, grid2, grid3)

plotter = pyvista.Plotter(shape=(1, 3))
for i in range(3):
    plotter.subplot(0, i)
    plotter.add_mesh(grid[i], show_edges=True)
    plotter.view_xy()
    plotter.add_axes()
plotter.show()