from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

def compute_v_based_on_reaction_diffusion(mesh_file, T=200, step_per_timeframe=5, 
                                          submesh_flag=False, ischemia_flag=False):
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

    def initial_condition(x):
        return np.where((x[0]-61.8)**2+(x[1]-60.9)**2+(x[2]-26.8)**2 < 25, 1, 0)

    u_n = Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)

    v_n = Function(V)
    v_n.name = "v_n"
    v_n.interpolate(lambda x : np.full(x.shape[1], 1))

    uh = Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)

    dx1 = Measure("dx", domain=subdomain_ventricle)
    u, v = TrialFunction(V), TestFunction(V)
    a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
    L_u = u_n * v * dx1  + dt * (v_n * (1 - u_n) * u_n * u_n / tau_in - u_n / tau_out) * v * dx1
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