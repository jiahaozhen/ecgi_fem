from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import create_submesh
import numpy as np
from ufl import TestFunction, dot, grad, Measure
from mpi4py import MPI

import sys
sys.path.append('.')
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion

def compare_b(mesh_file, v_data, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, gdim=3):
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))
    V3 = functionspace(domain, ("DG", 0, (tdim, tdim)))

    def rho1(x):
        tensor = np.eye(tdim) * sigma_t
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    def rho2(x):
        tensor = np.eye(tdim) * (sigma_i + sigma_e)
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    def rho3(x):
        tensor = np.eye(tdim) * sigma_t / 5
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    def rho4(x):
        tensor = np.eye(tdim) * sigma_t * 3
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    
    M = Function(V3)
    M.interpolate(rho1, cell_markers.find(1))
    M.interpolate(rho2, cell_markers.find(2))
    M.interpolate(rho3, cell_markers.find(3))
    M.interpolate(rho4, cell_markers.find(4))

    Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_i))

    # matrix A
    v1 = TestFunction(V1)

    v = Function(V2)
    v.x.array[:] = v_data

    dx2 = Measure("dx", domain=subdomain_ventricle)
    b_element_1 = -dot(grad(v1), dot(Mi, grad(v))) * dx2
    entity_map = {domain._cpp_object: ventricle_to_torso}
    linear_form_b_1 = form(b_element_1, entity_maps=entity_map)
    b_1 = assemble_vector(linear_form_b_1)

    v2 = TestFunction(V2)

    b_element_2 = -dot(grad(v2), dot(Mi, grad(v))) * dx2
    linear_form_b_2 = form(b_element_2)
    b_2 = assemble_vector(linear_form_b_2)

    b_1_array = b_1.array
    b_2_array = b_2.array

    def transfer_index(pts_1, pts_2):
        index = []
        for row in pts_1:
            # 计算当前 row 到 pts_2 所有点的欧式距离
            dist = np.linalg.norm(pts_2 - row, axis=1)
            # 找到距离最近的点索引
            idx = np.argmin(dist)
            index.append(idx)
        return index

    index = transfer_index(V2.tabulate_dof_coordinates(), V1.tabulate_dof_coordinates())
    b_1_reduced = b_1_array[index]

    diff = b_1_reduced - b_2_array
    print("Max difference between two methods for b:", np.max(np.abs(diff)))
    print("Mean difference between two methods for b:", np.mean(np.abs(diff)))
    print("Min difference between two methods for b:", np.min(np.abs(diff)))


if __name__ == "__main__": 
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=50, step_per_timeframe=2)
    compare_b(mesh_file, v_data[-1])