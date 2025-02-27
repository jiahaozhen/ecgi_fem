from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

def compute_v_based_on_reaction_diffusion(mesh_file, T = 100, step_per_timeframe = 5, 
                                          submesh_flag = False, ischemia_flag = False):
    if submesh_flag:
        # mesh of Body
        domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
        tdim = domain.topology.dim
        # mesh of Heart
        subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    else:
        subdomain_ventricle, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)

    t = 0
    num_steps = T * step_per_timeframe
    dt = T / num_steps  # time step size

    # A two-current model for the dynamics of cardiac membrane
    D = 4
    tau_in = 0.2
    tau_out = 10
    tau_open = 130
    tau_close = 150
    u_crit = 0.3

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

    # assume node 17 is the center of ischemia 
    # radius = 30
    # node 17 coordinates: (89.1, 40.9, -13.3)
    node_17 = np.array([89.1, 40.9, -13.3])
    class ischemia_condition():
        def __init__(self, u_ischemic, u_healthy, ischemia_center = node_17, radius = 30):
            self.u_ischemic = u_ischemic
            self.u_healthy = u_healthy
            self.ischemia_center = ischemia_center
            self.radius = radius
        def __call__(self, x):
            return np.where((x[0]-self.ischemia_center[0])**2 + 
                            (x[1]-self.ischemia_center[1])**2 +
                            (x[2]-self.ischemia_center[2])**2 < self.radius**2, 
                            self.u_ischemic, self.u_healthy)
    
    # the earliest node: 146
    # node 146 coordinates: (57, 51.2, 15)
    node_146 = np.array([57, 51.2, 15])
    class activation_initial_condition():
        def __init__(self, u_peak_ischemic, u_peak_healthy, u_rest_ischemic, u_rest_healthy, start = node_146, radius_start = 5, ischemia_center = node_17, radius_ischemia = 30):
            self.start = start
            self.radius_start = radius_start
            self.ischemia_center = ischemia_center
            self.radius_ischemia = radius_ischemia
            self.u_peak_ischemic = u_peak_ischemic
            self.u_peak_healthy = u_peak_healthy
            self.u_rest_ischemic = u_rest_ischemic
            self.u_rest_healthy = u_rest_healthy
        def __call__(self, x):
            if ischemia_flag:
                condition1 = (x[0]-self.start[0])**2 + (x[1]-self.start[1])**2 + (x[2]-self.start[2])**2 < self.radius_start**2
                condition2 = (x[0]-self.ischemia_center[0])**2 + (x[1]-self.ischemia_center[1])**2 + (x[2]-self.ischemia_center[2])**2 < self.radius_ischemia**2
                return np.where(condition1 & condition2, self.u_peak_ischemic,
                                np.where( ~condition1 & condition2, self.u_rest_ischemic,
                                            np.where( ~condition1 & ~condition2, self.u_rest_healthy, self.u_peak_healthy)))
            else:
                condition = (x[0]-self.start[0])**2 + (x[1]-self.start[1])**2 + (x[2]-self.start[2])**2 < self.radius_start**2
                return np.where(condition, self.u_peak_healthy, self.u_rest_healthy)

    if ischemia_flag:
        u_peak = Function(V)
        u_peak.interpolate(ischemia_condition(0.8, 1))
        u_rest = Function(V)
        u_rest.interpolate(ischemia_condition(0.2, 0))
    else:
        u_peak = 1
        u_rest = 0
    
    u_n = Function(V)
    u_n.interpolate(activation_initial_condition(0.8, 1, 0.2, 0))

    v_n = Function(V)
    v_n.interpolate(lambda x : np.full(x.shape[1], 1))

    uh = Function(V)
    uh.interpolate(activation_initial_condition(0.8, 1, 0.2, 0))

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
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    # solver.setType(PETSc.KSP.Type.PREONLY)
    # solver.getPC().setType(PETSc.PC.Type.LU)

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

    return np.array(u_data)

if __name__ == '__main__':
    mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    v_data = compute_v_based_on_reaction_diffusion(mesh_file, 200)