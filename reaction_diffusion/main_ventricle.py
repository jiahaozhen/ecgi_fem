from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from dolfinx.plot import vtk_mesh
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import matplotlib.pyplot as plt
import pyvista
import sys

sys.path.append('.')
from utils.helper_function import eval_function

mesh_file = '3d/data/mesh_ecgsim_ventricle.msh'
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim

t = 0
T = 200  # Final time
num_steps = T * 5
dt = T / num_steps  # time step size

# A two-current model for the dynamics of cardiac membrane
# D = 1e-3
# tau_in = 0.2
# tau_out = 10
# tau_open = 130
# tau_close = 150
# u_crit = 0.13

# A simple two-variable model of cardiac excitation
D = 4
k = 8
a = 0.17
e = 0.01

# A collocation-Galerkin finite element model of cardiac action potential propagation
# D = 4
# a = 0.12
# b = 1.1
# c1 = 2
# c2 = 0.25
# d = 5.5

V = functionspace(domain, ("Lagrange", 1))

def initial_condition(x):
    return np.where((x[0]-61.8)**2+(x[1]-60.9)**2+(x[2]-26.8)**2 < 25, 1, 0)

u_n = Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

v_n = Function(V)
v_n.name = "v_n"
v_n.interpolate(lambda x : np.full(x.shape[1], 0))

uh = Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

dx1 = Measure("dx", domain=domain)
u, v = TrialFunction(V), TestFunction(V)
a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
# L_u = u_n * v * dx1  + dt * (v_n * (1 - u_n) * u_n * u_n / tau_in - u_n / tau_out) * v * dx1
L_u = u_n * v * dx1 + dt * (k * u_n * (u_n - a) * (1 - u_n) - u_n * v_n) * v * dx1
# L_u = u_n * v * dx1 + dt * (c1 * u_n * (u_n - a) * (1 - u_n) - c2 * u_n * v_n) * v * dx1

bilinear_form = form(a_u)
linear_form_u = form(L_u)

A = assemble_matrix(bilinear_form)
A.assemble()
b_u = create_vector(linear_form_u)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)
solver.getPC().setType(PETSc.PC.Type.HYPRE)
# solver.setType(PETSc.KSP.Type.PREONLY)
# solver.getPC().setType(PETSc.PC.Type.LU)

plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
# grid["u"] = eval_function(u_n, domain.geometry.x)
grid["u"] = u_n.x.array
plotter.add_mesh(grid, scalars="u", cmap="viridis", clim=[0, 1])
plotter.view_isometric()
plotter.add_text('',name="time_text", font_size=18, color="red")
plotter.open_gif("reaction_diffusion/u_data_3d.gif")

u_data = []
u_data.append(u_n.x.array.copy())
for i in range(num_steps):
    t += dt

    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_u)

    solver.solve(b_u, uh.vector)

    u_n.x.array[:] = uh.x.array
    # v_n.x.array[:] = v_n.x.array + dt * np.where(u_n.x.array < u_crit, (1 - v_n.x.array) / tau_open, -v_n.x.array / tau_close)
    v_n.x.array[:] = v_n.x.array - dt * e * (v_n.x.array + k * u_n.x.array * (u_n.x.array - a - 1))
    # v_n.x.array[:] = v_n.x.array + dt * b * (u_n.x.array - d * v_n.x.array)
    u_data.append(u_n.x.array.copy())
    # print(i, '/', num_steps)

    # grid["u"] = eval_function(u_n, domain.geometry.x)
    grid["u"] = u_n.x.array
    plotter.update_scalars(grid["u"])
    plotter.actors["time_text"].SetText(3, f"Time: {t:.1f} ms")
    plotter.write_frame()

plotter.show()
np.save('reaction_diffusion/u_data.npy', u_data)

# u_data = np.array(u_data)[:,50]
# plt.plot(u_data)
# plt.show()