import sys

from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh, locate_entities_boundary
import numpy as np
from mpi4py import MPI
import scipy.io as sio

sys.path.append('.')
from utils.helper_function import compute_grad, submesh_node_index, compute_normal
from main_create_mesh_ecgsim_multi_conduct import create_mesh
from main_ecgsim2fem import ecgsim2fem
from main_forward_tmp import forward_tmp

def boundary_condition(u, v, cell_markers):
    domain = u.function_space.mesh
    subdomain = v.function_space.mesh
    sub_to_parent_node = submesh_node_index(domain, cell_markers, 2)
    nt_values = compute_normal(domain)
    nh_values = compute_normal(subdomain)
    grad_u_values = compute_grad(u)
    grad_v_values = compute_grad(v)

    # grad_u_norm = np.linalg.norm(grad_u_values, axis=1, keepdims=True)
    # grad_u_norm[grad_u_norm == 0] = 1e-8
    # grad_u_values = grad_u_values/grad_u_norm
    # grad_v_norm = np.linalg.norm(grad_v_values, axis=1, keepdims=True)
    # grad_v_norm[grad_v_norm == 0] = 1e-8
    # grad_v_values = grad_v_values/grad_v_norm

    # print(np.linalg.norm(grad_u_values, axis=1, keepdims=True))
    # print(np.linalg.norm(grad_v_values, axis=1, keepdims=True))
    # u_nan_mask = np.where(np.isnan(grad_u_values))
    # v_nan_mask = np.where(np.isnan(grad_v_values))
    # print(u_nan_mask)
    # print(v_nan_mask)

    body_boundary_index = locate_entities_boundary(domain, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool))
    heart_boundary_index = locate_entities_boundary(subdomain, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool))

    grad_u_values_exterior = grad_u_values[body_boundary_index]
    grad_v_values_exterior = grad_v_values[heart_boundary_index]
    grad_u_values_interior = grad_u_values[sub_to_parent_node[heart_boundary_index]]

    # sigma_i * grad(u_e) * n_h = - sigma_i * grad(v) * n_h on H
    bc1 = []
    # sigma_t * grad(u_t) * n_h = sigma_e * grad(u_e) * n_h on H
    bc2 = []
    # sigma_t * grad(u_t) * n_t = 0 on T
    bc3 = []
    # (sigma_i + sigma_e) * grad(u) * n_h + sigma_i * grad(v) * n_h = sigma_t * grad(u) * n_h on H
    bc4 = []

    for i, point in enumerate(heart_boundary_index):
        bc1.append(np.dot(grad_u_values_interior[i] + grad_v_values_exterior[i], nh_values[i])*sigma_i)
        bc2.append(np.dot(sigma_t*grad_u_values_interior[i] - sigma_e*grad_u_values_interior[i], nh_values[i]))
        bc4.append(np.dot((sigma_i+sigma_e-sigma_t)*grad_u_values_interior[i] + sigma_i*grad_v_values_exterior[i], nh_values[i]))
    for i, point in enumerate(body_boundary_index):
        bc3.append(np.dot(grad_u_values_exterior[i], nt_values[i])*sigma_t)

    # print('grad(u) on T:', grad_u_values_exterior)
    # print('grad(u) on H:', grad_u_values_interior)
    # print('grad(v)', grad_v_values)
    # print('nt:', nt_values)
    # print('nh:', nh_values)

    # print(np.linalg.norm(bc1))
    # print(np.linalg.norm(bc2))
    # print(np.linalg.norm(bc3))
    # print(np.linalg.norm(bc4))

    return (np.linalg.norm(bc1), np.linalg.norm(bc2), np.linalg.norm(bc3), np.linalg.norm(bc4))

if __name__ == '__main__':
    lc = 60
    sigma_i = 0.4
    sigma_e = 0.8
    sigma_t = 0.8
    file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    create_mesh(file, lc, multi_flag=False)
    ecgsim2fem(file, ischemic=False)
    v_data = np.load('3d/data/v_all.npy')
    u_data = forward_tmp(file, v_data, multi_flag=False, sigma_e=sigma_e, sigma_t=sigma_t, sigma_i=sigma_i)
    if v_data.ndim == 1:
        v_data = v_data.reshape(1,-1)
    if u_data.ndim == 1:
        u_data = u_data.reshape(1,-1)

    # mesh of Body
    domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain, ("Lagrange", 1))

    u = Function(V1)
    v = Function(V2)
    total_num = len(u_data)
    bc = []
    for i in range(total_num):
        u.x.array[:] = u_data[i]
        v.x.array[:] = v_data[i]
        bc.append(boundary_condition(u, v, cell_markers))
    bc = np.array(bc)
    print(bc)
    # sio.savemat('3d/data/boundary_condition.mat', {'bc': bc})