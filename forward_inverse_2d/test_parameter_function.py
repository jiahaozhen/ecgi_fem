from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.mesh import create_submesh, locate_entities
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, dx, Measure
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI

domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)

V = functionspace(domain, ("Lagrange", 1))

# create function based on the coefficient
phi = Function(V)
phi.vector[:] = np.zeros(len(phi.vector.array))

# G_tau function
def G_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 1
    result[condition2] = 0
    result[condition3] = 0.5 * (1 + s[condition3] / tau + (1 / np.pi) * np.sin(np.pi * s[condition3] / tau))
    return result

def delta_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 0
    result[condition2] = 0
    result[condition3] = (1 / (2 * tau)) * (1 + np.cos(np.pi * s[condition3] / tau))
    return result

def delta_tau_s(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 0
    result[condition2] = 0
    result[condition3] = -(np.pi / (2 * tau**2)) * np.sin(np.pi * s / tau)
    return result

# composite function
composite_f = Function(V)
tau = 0.1
# composite_f.vector[:] = G_tau(phi.vector.array, tau)
composite_f.vector[:] = delta_tau(phi.vector.array, tau)
# composite_f.vector[:] = delta_tau_s(phi.vector.array, tau)

print(composite_f.vector.array)