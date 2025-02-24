from dolfinx.mesh import CellType, create_box
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

nx, ny, nz = 40, 40, 40
domain = create_box(MPI.COMM_WORLD, [np.array([-1, -1, -1]), np.array([1, 1, 1])], [nx, ny, nz], CellType.tetrahedron)
t = 0  # Start time
T = 100  # Final time
num_steps = 500
dt = T / num_steps  # time step size

# parameter in two-current model
D = 1e-3
tau_in = 0.2
tau_out = 10
tau_open = 130
tau_close = 150
u_crit = 0.13

V = functionspace(domain, ("Lagrange", 1))

def initial_condition(x):
    return np.where(x[0]**2+x[1]**2+x[2]**2 < 1, 1, 0)

u_n = Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

v_n = Function(V)
v_n.name = "v_n"
v_n.interpolate(lambda x : np.full(x.shape[1], 1))

uh = Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

dx1 = Measure("dx", domain=domain)
u, v = TrialFunction(V), TestFunction(V)
a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
L_u = u_n * v * dx1  + dt * (v_n*(1-u_n)*u_n*u_n/tau_in - u_n/tau_out) * v * dx1

bilinear_form = form(a_u)
linear_form_u = form(L_u)

A = assemble_matrix(bilinear_form)
A.assemble()
b_u = create_vector(linear_form_u)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

u_data = []
u_data.append(u_n.x.array.copy())
for i in range(num_steps):
    t += dt

    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_u)

    solver.solve(b_u, uh.vector)

    u_n.x.array[:] = uh.x.array
    v_n.x.array[:] = v_n.x.array + dt * np.where(u_n.x.array < u_crit, (1 - v_n.x.array) / tau_open, -v_n.x.array / tau_close)
    u_data.append(u_n.x.array.copy())

# points = domain.geometry.x
# x = [1,1,1]
# index = min(range(len(points)), key=lambda i: (points[i][0] - x[0]) ** 2 + (points[i][1] - x[1]) ** 2)
u_data = np.array(u_data)[:-1, 0]
plt.plot(np.arange(0, T, dt), u_data)
plt.show()