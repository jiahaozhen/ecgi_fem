import sys
import time

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure, derivative, sqrt, inner
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import matplotlib.pyplot as plt
import multiprocessing

sys.path.append('.')
from utils.helper_function import G_tau, delta_tau, delta_deri_tau, compute_error_with_v, eval_function, compute_phi_with_v_timebased

gdim = 3
if gdim == 2:
    mesh_file = '2d/data/heart_torso.msh'
    v_exact_data_file = '2d/data/v_data_reaction_diffusion.npy'
    d_data_file = '2d/data/u_data_reaction_diffusion.npy'
else:
    mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
    v_exact_data_file = '3d/data/v_data_reaction_diffusion.npy'
    d_data_file = '3d/data/u_data_reaction_diffusion.npy'

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim = gdim)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain_ventricle.topology.index_map(0).size_local
sub_domain_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

# function space
V = functionspace(subdomain_ventricle, ("Lagrange", 1))
v =  Function(V)
phi_1 = Function(V)
phi_2 = Function(V)
delta_phi_1 = Function(V)
delta_phi_2 = Function(V)
alpha1 = 1e-4
alpha2 = 1e-4
dx2 = Measure("dx", domain=subdomain_ventricle)
tau = 1

# load data
v_exact_data = np.load(v_exact_data_file)
# phi_1_data = np.load('3d/data/phi_1_result.npy')
# phi_2_data = np.load('3d/data/phi_2_result.npy')
phi_1_data, phi_2_data = compute_phi_with_v_timebased(v_exact_data, V, -60, 20)
time_total = v_exact_data.shape[0]
reg_element = alpha1 * delta_phi_1 * sqrt(inner(grad(phi_1), grad(phi_1)) + 1e-8) * dx2 + \
                alpha2 * delta_phi_2 * sqrt(inner(grad(phi_2), grad(phi_2)) + 1e-8) * dx2
reg_time = []
for i in range(time_total):
    phi_1.x.array[:] = phi_1_data[i]
    phi_2.x.array[:] = phi_2_data[i]
    delta_phi_1.x.array[:] = delta_tau(phi_1_data[i], tau)
    delta_phi_2.x.array[:] = delta_tau(phi_2_data[i], tau)
    reg_time.append(assemble_scalar(form(reg_element)))

plt.plot(reg_time)
plt.show()
