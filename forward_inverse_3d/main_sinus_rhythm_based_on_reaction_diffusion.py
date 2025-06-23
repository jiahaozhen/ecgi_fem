import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import h5py

sys.path.append('.')
from utils.helper_function import eval_function, find_vertex_with_coordinate, fspace2mesh, get_activation_time_from_v, transfer_bsp_to_standard12lead
from main_forward_tmp import compute_d_from_tmp

mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
# mesh of Body
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

# A two-current model for the dynamics of cardiac membrane
D = 1e-1
tau_in = 0.4
tau_out = 10
tau_open = 130
tau_close = 150
u_crit = 0.13

V = functionspace(subdomain_ventricle, ("Lagrange", 1))

u_n = Function(V)
v_n = Function(V)
J_stim = Function(V)
uh = Function(V)
u_n.interpolate(lambda x: np.full(x.shape[1], 0))
uh.interpolate(lambda x: np.full(x.shape[1], 0))
v_n.interpolate(lambda x : np.full(x.shape[1], 1))

T = 500
dt = 0.1
num_steps = int(T / dt)  # time step size
dx1 = Measure("dx", domain=subdomain_ventricle)
u, v = TrialFunction(V), TestFunction(V)
a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
L_u = u_n * v * dx1 + dt * (v_n * (1 - u_n) * u_n * u_n / tau_in - u_n / tau_out + J_stim) * v * dx1

bilinear_form = form(a_u) 
linear_form_u = form(L_u)

A = assemble_matrix(bilinear_form)
A.assemble()
b_u = create_vector(linear_form_u)

solver = PETSc.KSP().create(subdomain_ventricle.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

activation_dict = {
    8 : np.array([57, 51.2, 15]),
    14.4 : np.array([30.2, 45.2, -30]),
    14.5 : np.array([12.8, 54.2, -15]),
    18.7 : np.array([59.4, 29.8, 15]),
    23.5 : np.array([88.3, 41.2, -37.3]),
    34.9 : np.array([69.1, 27.1, -30]),
    45.6 : np.array([48.4, 40.2, -37.5])
}
# apply find_vertex_with_coordinate on the dict key
activation_dict = {k : find_vertex_with_coordinate(subdomain_ventricle, v) for k, v in activation_dict.items()}

functionspace2mesh = fspace2mesh(V)
mesh2functionspace = np.argsort(functionspace2mesh)

activation_dict = {k * 10 : mesh2functionspace[v] for k, v in activation_dict.items()}

u_data = []
u_data.append(u_n.x.array.copy())
t = 0
last_time = 0
for i in range(num_steps):
    t += dt
    if i in activation_dict:
        # u_n.x.array[activation_dict[i]] = 1
        J_stim.x.array[activation_dict[i]] = 0.5
        last_time = 50
    else:
        last_time = last_time - 1
        if last_time < 0:
            J_stim.x.array[:] = np.zeros(J_stim.x.array.shape)
    with b_u.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b_u, linear_form_u)
    solver.solve(b_u, uh.vector)

    u_n.x.array[:] = uh.x.array
    v_n.x.array[:] = v_n.x.array + dt * np.where(u_n.x.array < u_crit, (1 - v_n.x.array) / tau_open, -v_n.x.array / tau_close)
    u_data.append(u_n.x.array.copy())
u_data = np.array(u_data)
u_data = np.where(u_data > 1, 1, u_data)
u_data = np.where(u_data < 0, 0, u_data)
u_data = u_data * 100 - 90
d_data_fem = compute_d_from_tmp(mesh_file, v_data=u_data)
d_data_ecgsim = np.array(h5py.File('3d/data/sinus_rhythm_ecgsim.mat','r')['surface_potential'])

leadIndex = np.array([19, 26, 65, 41, 48, 54, 1, 2, 66]) - 1
standard12Lead_fem = transfer_bsp_to_standard12lead(d_data_fem, leadIndex)
standard12Lead_ecgsim = transfer_bsp_to_standard12lead(d_data_ecgsim, leadIndex)

# Plot the 12-lead ECG
fig, axs = plt.subplots(4, 3, figsize=(15, 10))
leads = [
    "lead I", "lead II", "lead III", "lead V1", "lead V2", "lead V3",
    "lead V4", "lead V5", "lead V6", "lead aVR", "lead aVL", "lead aVF"
]

time_fem = np.arange(0, standard12Lead_fem.shape[0] / 10, 0.1)
time_ecgsim = np.arange(0, standard12Lead_ecgsim.shape[0], 1)
for i, ax in enumerate(axs.flat):
    ax.plot(time_fem, standard12Lead_fem[:, i])
    ax.plot(time_ecgsim, standard12Lead_ecgsim[:, i], linestyle='--')
    ax.legend(['FEM', 'ECGsim'], loc='upper right')
    ax.set_title(leads[i])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Potential (mV)')
    ax.grid(True)

fig.suptitle('12-lead ECG', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

activation_time = get_activation_time_from_v(u_data)
print("the activation last for", (max(activation_time)-min(activation_time)) / 10, "ms.")
tmp_parameter_file = h5py.File('3d/data/tmp_parameter_ecgsim.mat','r')
activation_time_ecgsim = np.array(tmp_parameter_file['tmp_parameter_ecgsim']['dep']).reshape(-1)
at = Function(V)
at.x.array[:] = activation_time
activation_time_fem = eval_function(at, np.array(h5py.File('3d/data/geom_ecgsim.mat', 'r')['geom_ventricle']['pts'])).reshape(-1) / 10
re  = np.linalg.norm(activation_time_ecgsim - activation_time_fem) / np.linalg.norm(activation_time_ecgsim)
cc = np.corrcoef(activation_time_ecgsim, activation_time_fem)[0, 1]
print('Activation Relative error:', re)
print('Activation Correlation coefficient:', cc)
# Plot the activation time on the mesh
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
grid["activation_time"] = eval_function(at, subdomain_ventricle.geometry.x)/10
plotter.add_mesh(grid, scalars="activation_time", cmap="coolwarm", clim=[np.min(activation_time)/10, np.max(activation_time)/10])
plotter.view_isometric()
plotter.add_text("Activation Time", font_size=18, color="black")
plotter.show()

plt.plot(u_data[:, activation_dict[80]], label="activation time 8")
plt.plot(u_data[:, activation_dict[144]], label="activation time 14.4")
plt.plot(u_data[:, activation_dict[145]], label="activation time 14.5")
plt.plot(u_data[:, activation_dict[187]], label="activation time 18.7")
plt.plot(u_data[:, activation_dict[235]], label="activation time 23.5")
plt.plot(u_data[:, activation_dict[349]], label="activation time 34.9")
plt.plot(u_data[:, activation_dict[456]], label="activation time 45.6")
plt.xlabel("time step")
plt.ylabel("u")
plt.title("Transmembrane Potential")
plt.legend()
plt.show()

# plotter = pyvista.Plotter()
# grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
# u = Function(V)
# u.x.array[:] = u_data[0]
# grid["u"] = eval_function(u, subdomain_ventricle.geometry.x)
# plotter.add_mesh(grid, scalars="u", cmap="viridis", clim=[0,1])
# plotter.view_isometric()
# plotter.add_text('',name="time_text", font_size=18, color="red")
# name = 'reaction_diffusion/sinus_rhyme_3d.gif'
# plotter.open_gif(name)

# t = 0
# time_step = 0.1
# for i in range(u_data.shape[0]):
#     u.x.array[:] = u_data[i]
#     grid["u"] = eval_function(u, subdomain_ventricle.geometry.x)
#     plotter.update_scalars(grid["u"])
#     plotter.actors["time_text"].SetText(3, f"Time: {t:.1f} ms")
#     plotter.write_frame()
#     t += time_step

# plotter.close()