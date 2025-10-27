import sys

import numpy as np
from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.fem import functionspace, Function
from mpi4py import MPI

sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
from utils.helper_function import compute_grad, compute_normal

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    gdim = 3
    T = 120
    step_per_timeframe = 2
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=T, step_per_timeframe=step_per_timeframe)

    file = "forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh"
    domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
    tdim = domain.topology.dim
    subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V = functionspace(subdomain, ("Lagrange", 1))
    v = Function(V)

    nh_values = compute_normal(subdomain)
    heart_boundary_index = locate_entities_boundary(subdomain, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool))

    # grad(v) * n_h on H
    bc = []

    for i in range(v_data.shape[0]):
        v.x.array[:] = v_data[i]
        grad_v_values = compute_grad(v)
        grad_v_values_exterior = grad_v_values[heart_boundary_index]
        for i, point in enumerate(heart_boundary_index):
            bc.append(np.dot(grad_v_values_exterior[i], nh_values[i]))
        print(f'Frame {i}: grad(v) * n_h on H norm:', np.linalg.norm(bc))
        bc = []
