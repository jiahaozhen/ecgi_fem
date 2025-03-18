import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np

sys.path.append('.')
from utils.helper_function import v_data_argument

def compute_v_based_on_reaction_diffusion(mesh_file, gdim=3, T=100, step_per_timeframe=5, 
                                          u_peak_ischemia_val=0.7, u_rest_ischemia_val=0.3,
                                          submesh_flag=False, ischemia_flag=False, 
                                          center_activation=np.array([57, 51.2, 15]), radius_activation=5,
                                          center_ischemia=np.array([89.1, 40.9, -13.3]), radius_ischemia=30,
                                          data_argument=False, v_min=-90, v_max=10,
                                          surface_flag=False):
    
    if submesh_flag:
        # mesh of Body
        domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
        tdim = domain.topology.dim
        # mesh of Heart
        subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    else:
        subdomain_ventricle, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)

    t = 0
    num_steps = T * step_per_timeframe
    dt = T / num_steps  # time step size

    # A two-current model for the dynamics of cardiac membrane
    if gdim == 3:
        D = 4
    else:
        D = 0.001
    tau_in = 0.2
    tau_out = 10
    tau_open = 130
    tau_close = 150
    u_crit = 0.13

    # A simple two-variable model of cardiac excitation
    # D = 4
    # k = 8
    # a = 0.17
    # e = 0.01

    # A collocation-Galerkin finite element model of cardiac action potential propagation
    # D = 4
    # a = 0.12
    # b = 1.1
    # c1 = 2
    # c2 = 0.25
    # d = 5.5

    V = functionspace(subdomain_ventricle, ("Lagrange", 1))

    if surface_flag:
    # use ecgsim ischemia data
        geom_data_ecgsim = h5py.File('3d/data/geom_ecgsim.mat', 'r')
        v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])
        file_ecgsim = h5py.File('3d/data/surface_ischemia.mat', 'r')
        v_data_ecgsim = np.array(file_ecgsim['v'])[0]
        v_fem_one = scipy.interpolate.griddata(v_pts_ecgsim, v_data_ecgsim, V.tabulate_dof_coordinates(), method='linear', fill_value=0)

    # assume node 17 is the center of ischemia 
    # radius = 30
    # node 17 coordinates: (89.1, 40.9, -13.3)
    class ischemia_condition():
        def __init__(self, u_ischemia, u_healthy, center=center_ischemia, r=radius_ischemia):
            self.u_ischemia = u_ischemia
            self.u_healthy = u_healthy
            self.center = center
            self.r = r
        def __call__(self, x):
            if gdim == 3:
                return np.where((x[0]-self.center[0])**2 + 
                                (x[1]-self.center[1])**2 +
                                (x[2]-self.center[2])**2 < self.r**2, 
                                self.u_ischemia, self.u_healthy)
            else:
                return np.where((x[0]-self.center[0])**2 + 
                                (x[1]-self.center[1])**2 < self.r**2, 
                                self.u_ischemia, self.u_healthy)
    
    # the earliest node: 146
    # node 146 coordinates: (57, 51.2, 15)
    class activation_initial_condition():
        def __init__(self, u_peak_ischemia, u_rest_ischemia, u_peak_healthy=1, u_rest_healthy=0, 
                    center_a=center_activation, r_a=radius_activation, center_i=center_ischemia, r_i=radius_ischemia):
            self.u_peak_ischemia = u_peak_ischemia
            self.u_peak_healthy = u_peak_healthy
            self.u_rest_ischemia = u_rest_ischemia
            self.u_rest_healthy = u_rest_healthy
            self.center_a = center_a
            self.r_a = r_a
            self.center_i = center_i
            self.r_i = r_i
            if surface_flag:
                self.center_i = np.array([0, 0, 0])
                self.r_i = 0.1
        def __call__(self, x):
            if gdim == 3:
                condition1 = (x[0]-self.center_a[0])**2 + (x[1]-self.center_a[1])**2 + (x[2]-self.center_a[2])**2 < self.r_a**2
                condition2 = (x[0]-self.center_i[0])**2 + (x[1]-self.center_i[1])**2 + (x[2]-self.center_i[2])**2 < self.r_i**2
            else:
                condition1 = (x[0]-self.center_a[0])**2 + (x[1]-self.center_a[1])**2 < self.r_a**2
                condition2 = (x[0]-self.center_i[0])**2 + (x[1]-self.center_i[1])**2 < self.r_i**2
            
            if ischemia_flag:
                return np.where(condition1 & condition2, self.u_peak_ischemia,
                                np.where(~condition1 & condition2, self.u_rest_ischemia, 
                                         np.where( ~condition1 & ~condition2, self.u_rest_healthy, self.u_peak_healthy)))
            else:
                return np.where(condition1, self.u_peak_healthy, self.u_rest_healthy)

    u_peak = Function(V)
    u_rest = Function(V)
    u_n = Function(V)
    v_n = Function(V)
    uh = Function(V)
    if ischemia_flag:
        u_peak.interpolate(ischemia_condition(u_peak_ischemia_val, 1))
        u_rest.interpolate(ischemia_condition(u_rest_ischemia_val, 0))
    else:
        u_peak = 1
        u_rest = 0
    
    u_n.interpolate(activation_initial_condition(u_peak_ischemia_val, u_rest_ischemia_val))
    v_n.interpolate(lambda x : np.full(x.shape[1], 1))
    uh.interpolate(activation_initial_condition(u_peak_ischemia_val, u_rest_ischemia_val))

    if surface_flag:
        u_peak.x.array[:] = np.where(v_fem_one > 0.5, u_peak_ischemia_val, 1)
        u_rest.x.array[:] = np.where(v_fem_one > 0.5, u_rest_ischemia_val, 0)
        u_activation = Function(V)
        u_activation.interpolate(activation_initial_condition(u_peak_ischemia_val, u_rest_ischemia_val))
        u_n.x.array[:] = np.where(v_fem_one > 0.5, u_rest_ischemia_val, 
                                  np.where(u_activation.x.array > 0.5, 1, 0))
        uh.x.array[:] = np.where(v_fem_one > 0.5, u_rest_ischemia_val, 
                                 np.where(u_activation.x.array > 0.5, 1, 0))

    dx1 = Measure("dx", domain=subdomain_ventricle)
    u, v = TrialFunction(V), TestFunction(V)
    a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
    if ischemia_flag:
        L_u = u_n * v * dx1 + dt * (v_n * (u_peak - u_n) * (u_n - u_rest) * (u_n - u_rest) / tau_in - (u_n - u_rest) / tau_out) * v * dx1
    else:
        L_u = u_n * v * dx1 + dt * (v_n * (1 - u_n) * u_n * u_n / tau_in - u_n / tau_out) * v * dx1
    # L_u = u_n * v * dx1 + dt * (k * u_n * (u_n - a) * (1 - u_n) - u_n * v_n) * v * dx1
    # L_u = u_n * v * dx1 + dt * (c1 * u_n * (u_n - a) * (1 - u_n) - c2 * u_n * v_n) * v * dx1

    bilinear_form = form(a_u)
    linear_form_u = form(L_u)

    A = assemble_matrix(bilinear_form)
    A.assemble()
    b_u = create_vector(linear_form_u)

    solver = PETSc.KSP().create(subdomain_ventricle.comm)
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
        # v_n.x.array[:] = v_n.x.array - dt * e * (v_n.x.array + k * u_n.x.array * (u_n.x.array - a - 1))
        # v_n.x.array[:] = v_n.x.array + dt * b * (u_n.x.array - d * v_n.x.array)
        u_data.append(u_n.x.array.copy())
        # print(i, '/', num_steps)

    u_data = np.array(u_data)
    u_data = u_data * (v_max - v_min) + v_min
    if data_argument:
        u_data = v_data_argument(u_data, function_space=V)
    return u_data

if __name__ == '__main__':
    submesh_flag = False
    if submesh_flag:
        mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
        # mesh of Body
        domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
        tdim = domain.topology.dim
        # mesh of Heart
        subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    else:
        mesh_file = '3d/data/mesh_ecgsim_ventricle.msh'
        subdomain_ventricle, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
        tdim = subdomain_ventricle.topology.dim
    v_data = compute_v_based_on_reaction_diffusion(mesh_file, T = 500, submesh_flag=submesh_flag, ischemia_flag=True)
    # find the 2 point nearest to the center of ischemia
    # node 17 coordinates: (89.1, 40.9, -13.3)
    pts = subdomain_ventricle.geometry.x
    center = np.array([89.1, 40.9, -13.3])
    dist = np.linalg.norm(pts - center, axis=1)
    idx = np.argsort(dist)
    plt.plot(np.arange(0, v_data.shape[0]/5, 0.2), v_data[:, idx[-24]])
    plt.xlabel('Time (ms)')
    # plt.plot(v_data[:, idx[1]])
    plt.show()