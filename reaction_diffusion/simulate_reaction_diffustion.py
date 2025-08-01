import sys

from dolfinx.io import gmshio
from dolfinx.plot import vtk_mesh
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
import pyvista

sys.path.append('.')
from utils.helper_function import distinguish_epi_endo, eval_function, v_data_argument, compute_phi_with_v_timebased, find_vertex_with_coordinate, fspace2mesh, assign_function

def compute_v_based_on_reaction_diffusion(mesh_file, gdim=3,
                                          ischemia_flag=True, ischemia_epi_endo=[-1, 0, 1],
                                          center_ischemia=np.array([32.1, 71.7, 15]), radius_ischemia=30,
                                          T=120, step_per_timeframe=10,
                                          u_peak_ischemia_val=0.7, u_rest_ischemia_val=0.3, tau=10,
                                          data_argument=False, v_min=-90, v_max=10):
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    # 0 is mid-myocardial, 1 is epicardium, -1 is endocardium
    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

    t = 0
    num_steps = T * step_per_timeframe
    dt = T / num_steps  # time step size

    D = 1e-1
    tau_in = 0.4
    tau_out = 10
    tau_open = 130
    tau_close = 150
    u_crit = 0.13

    V = functionspace(subdomain_ventricle, ("Lagrange", 1))
    functionspace2mesh = fspace2mesh(V)
    mesh2functionspace = np.argsort(functionspace2mesh)

    marker_function = Function(V)
    marker_function.x.array[:] = mesh2functionspace[epi_endo_marker]
    assign_function(marker_function, np.arange(len(subdomain_ventricle.geometry.x)), epi_endo_marker)

    class ischemia_condition():
        def __init__(self, u_ischemia, u_healthy, center=center_ischemia, r=radius_ischemia):
            self.u_ischemia = u_ischemia
            self.u_healthy = u_healthy
            self.center = center
            self.r = r
        def __call__(self, x):
            if gdim == 3:
                distance_value = (x[0]-self.center[0])**2 + (x[1]-self.center[1])**2 + (x[2]-self.center[2])**2 - self.r**2
            else:
                distance_value = (x[0]-self.center[0])**2 + (x[1]-self.center[1])**2 - self.r**2
            
            marker_value = eval_function(marker_function, x.transpose()).squeeze()
            ret_value = np.full(x.shape[1], self.u_healthy, dtype=np.float64)
            if 0 in ischemia_epi_endo:
                condition = np.where(np.abs(marker_value-0) < 1e-3, distance_value < 0, False)
                ret_value[condition] = self.u_ischemia
            if 1 in ischemia_epi_endo:
                condition = np.where(np.abs(marker_value-1) < 1e-3, distance_value < 0, False)
                ret_value[condition] = self.u_ischemia
            if -1 in ischemia_epi_endo:
                condition = np.where(np.abs(marker_value+1) < 1e-3, distance_value < 0, False)
                ret_value[condition] = self.u_ischemia
            return ret_value
    
    u_peak = Function(V)
    u_rest = Function(V)
    u_n = Function(V)
    v_n = Function(V)
    uh = Function(V)
    J_stim = Function(V)
    if ischemia_flag:
        u_peak.interpolate(ischemia_condition(u_peak_ischemia_val, 1))
        u_rest.interpolate(ischemia_condition(u_rest_ischemia_val, 0))
        u_n.interpolate(ischemia_condition(u_rest_ischemia_val, 0))
        uh.interpolate(ischemia_condition(u_rest_ischemia_val, 0))
    else:
        u_peak = 1
        u_rest = 0
        u_n.interpolate(lambda x: np.full(x.shape[1], 0))
        uh.interpolate(lambda x: np.full(x.shape[1], 0))

    v_n.interpolate(lambda x : np.full(x.shape[1], 1))

    if isinstance(u_peak, Function):
        ischemia_marker = np.where(np.abs(u_peak.x.array - u_peak_ischemia_val) < 1e-3, 1, 0)
    else:
        ischemia_marker = np.full(u_n.x.array.shape, 0)
    
    dx1 = Measure("dx", domain=subdomain_ventricle)
    u, v = TrialFunction(V), TestFunction(V)
    a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
    L_u = u_n * v * dx1 + dt * (v_n * (u_peak - u_n) * (u_n - u_rest) * (u_n - u_rest) / tau_in - (u_n - u_rest) / tau_out  + J_stim) * v * dx1
    
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
    activation_dict = {k : find_vertex_with_coordinate(subdomain_ventricle, v) for k, v in activation_dict.items()}

    functionspace2mesh = fspace2mesh(V)
    mesh2functionspace = np.argsort(functionspace2mesh)

    activation_dict = {k * step_per_timeframe : mesh2functionspace[v] for k, v in activation_dict.items()}

    last_time = 0
    u_data = []
    u_data.append(u_n.x.array.copy())
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
    u_data = u_data * (v_max - v_min) + v_min
    phi_1, phi_2 = compute_phi_with_v_timebased(u_data, V, ischemia_marker)
    if data_argument:
        u_data = v_data_argument(phi_1, phi_2, tau=tau)
    
    return u_data, phi_1, phi_2

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=10)