import sys

from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

sys.path.append('.')
from utils.helper_function import eval_function, v_data_argument, compute_phi_with_v_timebased, find_vertex_with_coordinate, fspace2mesh, assign_function
from utils.ventricular_segmentation_tools import distinguish_epi_endo

def compute_v_based_on_reaction_diffusion(mesh_file, gdim=3,
                                          ischemia_flag=True, ischemia_epi_endo=[-1, 0, 1],
                                          center_ischemia=np.array([32.1, 71.7, 15]), radius_ischemia=30,
                                          T=120, step_per_timeframe=10,
                                          u_peak_ischemia_val=0.7, u_rest_ischemia_val=0.3, tau=10,
                                          data_argument=False, v_min=-90, v_max=10,
                                          plot_flag=False):
    '''
    diff between compute_v_based_on_reaction_diffusion in main_reaction_diffusion_on_ventricle.py:
    1. add ischemia_epi_endo to choose which layer to set ischemia
    2. activation is fixed
    
    diff between compute_v_based_on_reaction_diffusion in main_reaction_diffusion.py:
    1. remove submesh_flag and surface_flag, always use submesh of ventricle
    2. add ischemia_epi_endo to choose which layer to set ischemia
    '''
    if not hasattr(compute_v_based_on_reaction_diffusion, "epi_endo_marker"):
        # mesh of Body
        domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
        tdim = domain.topology.dim
        # mesh of Heart
        subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
        # 1 is epicardium, 0 is mid-myocardial, -1 is endocardium
        epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)
        V = functionspace(subdomain_ventricle, ("Lagrange", 1))
        functionspace2mesh = fspace2mesh(V)
        mesh2functionspace = np.argsort(functionspace2mesh)
        compute_v_based_on_reaction_diffusion.epi_endo_marker = epi_endo_marker
        compute_v_based_on_reaction_diffusion.subdomain_ventricle = subdomain_ventricle
        compute_v_based_on_reaction_diffusion.V = V
        compute_v_based_on_reaction_diffusion.mesh2functionspace = mesh2functionspace
    
    subdomain_ventricle = compute_v_based_on_reaction_diffusion.subdomain_ventricle
    epi_endo_marker = compute_v_based_on_reaction_diffusion.epi_endo_marker
    mesh2functionspace = compute_v_based_on_reaction_diffusion.mesh2functionspace
    V = compute_v_based_on_reaction_diffusion.V

    t = 0
    num_steps = T * step_per_timeframe
    dt = T / num_steps  # time step size

    D = 1e-1
    tau_in = 0.4
    tau_out = 10
    tau_open = 130
    tau_close = 150
    u_crit = 0.13

    marker_function = Function(V)
    assign_function(marker_function, np.arange(len(subdomain_ventricle.geometry.x)), epi_endo_marker)

    class ischemia_condition():
        def __init__(self, u_ischemia, u_healthy, center=center_ischemia, r=radius_ischemia):
            self.u_ischemia = u_ischemia
            self.u_healthy = u_healthy
            self.center = center
            self.r = r
        def __call__(self, x):
            marker_value = eval_function(marker_function, x.T).ravel()
            distance_value = np.sum((x.T - self.center)**2, axis=1) - self.r**2
            mask = (distance_value < 0) & np.isin(marker_value.round(), ischemia_epi_endo)
            ret_value = np.where(mask, self.u_ischemia, self.u_healthy)
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

    if not hasattr(compute_v_based_on_reaction_diffusion, "solver"):
        A = assemble_matrix(bilinear_form)
        A.assemble()
        compute_v_based_on_reaction_diffusion.A = A
        solver = PETSc.KSP().create(subdomain_ventricle.comm)
        solver.setOperators(A)
        solver.setType("cg")  # 改为共轭梯度法
        solver.getPC().setType("hypre")  # 或 "ilu", "gamg"
        solver.setTolerances(rtol=1e-6)
        compute_v_based_on_reaction_diffusion.solver = solver
    else:
        solver = compute_v_based_on_reaction_diffusion.solver
    
    b_u = create_vector(linear_form_u)

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

    activation_dict = {k * step_per_timeframe : mesh2functionspace[v] for k, v in activation_dict.items()}

    last_time = 0
    u_data = []
    u_data.append(u_n.x.array.copy())
    for i in range(num_steps):
        t += dt
        if i in activation_dict:
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
    phi_1 = None
    phi_2 = None
    # phi_1, phi_2 = compute_phi_with_v_timebased(u_data, V, ischemia_marker)
    if data_argument:
        u_data = v_data_argument(phi_1, phi_2, tau=tau)
    
    return u_data, phi_1, phi_2

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=10)