# compute potential u in B from transmembrane potential v
import sys

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import scipy.io as sio
import h5py

sys.path.append('.')
from utils.helper_function import delta_tau

# mesh of Body
file1 = "3d/mesh_two_conduct_ecgsim.msh"
file2 = "3d/mesh_multi_conduct_ecgsim.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file2, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
# mesh of Lung
subdomain_lung, lung_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(3))
# mesh of Cavity
subdomain_cav, cav_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(4))

num_node = domain.topology.index_map(tdim-3).size_local
num_cell = domain.topology.index_map(tdim).size_local

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

# Mi : intra-cellular conductivity tensor in Heart
# Me : extra-cellular conductivity tensor in Heart
# M0 : conductivity tensor in Torso
# M  : Mi + Me in Heart
#      M0 in Torso
# S/m
M0_torso_value = 0.2
M0_lung_value = M0_torso_value / 5
M0_cav_value = M0_torso_value * 3
Mi_value = 0.1
Me_value = 0.1

def rho1(x):
    tensor = np.eye(tdim) * M0_torso_value
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho2(x):
    tensor = np.eye(tdim) * (Mi_value + Me_value)
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho3(x):
    tensor = np.eye(tdim) * M0_lung_value
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho4(x):
    tensor = np.eye(tdim) * M0_cav_value
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])

V = functionspace(domain, ("DG", 0, (tdim, tdim)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
M.interpolate(rho3, cell_markers.find(3))
M.interpolate(rho4, cell_markers.find(4))
Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * Mi_value))

# A_u u = b_u
# matrix A
u1 = TestFunction(V1)
v1 = TrialFunction(V1)
dx1 = Measure("dx", domain = domain)
a_element = dot(grad(u1), dot(M, grad(v1))) * dx1
bilinear_form_a = form(a_element)
A = assemble_matrix(bilinear_form_a)
A.assemble()

# phi exact
class exact_solution():
    def __init__(self, x, y, z,  r):
        self.x = x
        self.y = y
        self.r = r
        self.z = z
    def __call__(self, x):
        dist = (x[0]-self.x)**2 + (x[1]-self.y)**2 + (x[2]-self.z)**2
        return np.sqrt(dist) - self.r

# preparation 
a1 = -60
a2 = -90
tau = 1
center = [87.9, 69.9, -42.6]
r = 30
phi = Function(V2)
phi.interpolate(exact_solution(center[0], center[1], center[2], r))
# phi.x.array[:] = (subdomain.geometry.x[:,0] - center[0]) ** 2 + \
#                 (subdomain.geometry.x[:,1] - center[1]) ** 2 + \
#                 (subdomain.geometry.x[:,2] - center[2]) ** 2 - r ** 2
delta_phi = Function(V2)
delta_phi.x.array[:] = delta_tau(phi.x.array, tau)

# b
dx2 = Measure("dx", domain = subdomain_ventricle)
b_element = (a1-a2) * delta_phi * dot(grad(u1), dot(Mi, grad(phi)))*dx2
entity_map = {domain._cpp_object: ventricle_to_torso}
linear_form_b = form(b_element, entity_maps = entity_map)
b = assemble_vector(linear_form_b)
b.assemble()

u = Function(V1)

solver = PETSc.KSP().create()
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.solve(b, u.vector)

np.save('3d/bsp_one_timeframe.npy', u.x.array)
np.save('3d/phi_exact.npy', phi.x.array)

marker = np.zeros(subdomain_ventricle.topology.index_map(tdim-3).size_local)
marker[phi.x.array < 0] = 1

grid1 = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
grid1.point_data["u"] = u.x.array
grid1.set_active_scalars("u")
grid2 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid2.point_data["phi"] = phi.x.array
grid2.set_active_scalars("phi")
grid3 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
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

# geom = h5py.File('3d/geom_ecgsim.mat', 'r')
# points = np.array(geom['geom_thorax']['pts'])
# tree = bb_tree(domain, domain.topology.dim)
# cells = []
# points_on_proc = []
# # Find cells whose bounding-box collide with the the points
# cell_candidates = compute_collisions_points(tree, points)
# # Choose one of the cells that contains the point
# colliding_cells = compute_colliding_cells(domain, cell_candidates, points)
# for i, point in enumerate(points):
#     if len(colliding_cells.links(i)) > 0:
#         points_on_proc.append(point)
#         cells.append(colliding_cells.links(i)[0])
# points_on_proc = np.array(points_on_proc, dtype=np.float64)
# u_values = u.eval(points_on_proc, cells)
# sio.savemat('3d/surface_potential.mat', {'surface_potential':u_values})