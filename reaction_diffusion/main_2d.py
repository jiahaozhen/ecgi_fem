from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista
import numpy as np

# Define temporal parameters
t = 0  # Start time
T = 200  # Final time
num_steps = T * 5
dt = T / num_steps  # time step size

nx, ny = 400, 400
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-4, -4]), np.array([4, 4])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

D = 1e-3
tau_in = 0.2
tau_out = 10
tau_open = 130
tau_close = 150
u_crit = 0.13

# Create initial condition
def initial_condition(x, a=5):
    # return np.where(x[0]**2+x[1]**2 < 0.25, 1, 0)
    return np.exp(-a * (x[0]**2 + x[1]**2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

v_n = fem.Function(V)
v_n.name = "v_n"
v_n.interpolate(lambda x : np.full(x.shape[1], 1))

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

dx1 = Measure("dx", domain=domain)
u, v = TrialFunction(V), TestFunction(V)
a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
L_u = u_n * v * dx1  + dt * (v_n*(1-u_n)*u_n*u_n/tau_in - u_n/tau_out) * v * dx1

bilinear_form = fem.form(a_u)
linear_form_u = fem.form(L_u)

A = assemble_matrix(bilinear_form)
A.assemble()
b_u = create_vector(linear_form_u)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# pyvista.start_xvfb()

# grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

# plotter = pyvista.Plotter()
# plotter.open_gif("reaction_diffusion/u_time.gif", fps=10)

# grid.point_data["uh"] = uh.x.array
# warped = grid.warp_by_scalar("uh", factor=1)

# viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
# sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
#              position_x=0.1, position_y=0.8, width=0.8, height=0.1)

# renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
#                             cmap=viridis, scalar_bar_args=sargs,
#                             clim=[0, max(uh.x.array)])

u_data = []
u_data.append(u_n.x.array.copy())
for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_u)

    # Solve linear problem
    solver.solve(b_u, uh.vector)

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array
    v_n.x.array[:] = v_n.x.array + dt * np.where(u_n.x.array < u_crit, (1 - v_n.x.array) / tau_open, -v_n.x.array / tau_close)

    u_data.append(u_n.x.array.copy())
    # Update plot
    # new_warped = grid.warp_by_scalar("uh", factor=1)
    # warped.points[:, :] = new_warped.points
    # warped.point_data["uh"][:] = uh.x.array
    # plotter.write_frame()
# plotter.close()

points = domain.geometry.x
x = [1,1]
index = min(range(len(points)), key=lambda i: (points[i][0] - x[0]) ** 2 + (points[i][1] - x[1]) ** 2)
u_data = np.array(u_data)[:-1,0]
plt.plot(np.arange(0, T, dt), u_data)
plt.show()