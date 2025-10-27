import sys

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, dirichletbc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, LinearProblem
from dolfinx.mesh import create_submesh, locate_entities_boundary
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import h5py

sys.path.append('.')
from utils.function_tools import extract_data_from_function
from reaction_diffusion.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion

def forward_tmp2ue(mesh_file, v_data, sigma_i=0.4, sigma_e=0.8, gdim=3, heart_marker=2):
    r'''
    \nabla \cdot ((M_i+M_e) \nabla u_e) + \nabla \cdot (M_i \nabla v) = 0 x \in H
    (M_i \nabla v + (M_i+M_e) \nabla u_e) \cdot \boldsymbol{n}_H = 0 x \in \partial H
    '''
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(heart_marker))

    V = functionspace(subdomain_ventricle, ("Lagrange", 1))

    u = Function(V) # ue
    v = Function(V) # tmp

    Me = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_e))
    Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_i))

    u1 = TrialFunction(V)
    v1 = TestFunction(V)

    dx = Measure("dx", domain=subdomain_ventricle)
    a_element = dot(grad(v1), dot(Mi+Me, grad(u1))) * dx
    bilinear_form_a = form(a_element)
    A = assemble_matrix(bilinear_form_a)
    A.assemble()

    b_element = -dot(grad(v1), dot(Mi, grad(v))) * dx
    linear_form_b = form(b_element)
    b = create_vector(linear_form_b)

    solver = PETSc.KSP().create()
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.ILU)

    if v_data.ndim == 1:
        v_data = v_data.reshape(1,-1)
    total_num = len(v_data)
    u_data = []
    for i in range(total_num):
        v.x.array[:] = v_data[i]
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form_b)
        solver.solve(b, u.vector)
        u_data.append(u.x.array.copy())
    return np.array(u_data), V

def inner_boundary_pts_from_mesh(mesh_file, gdim=3, marker=2):
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(marker))
    boundary_index = locate_entities_boundary(subdomain_ventricle, 0, lambda x: np.full(x.shape[1], True, dtype=bool))
    coords = subdomain_ventricle.geometry.x[boundary_index]
    return coords

def forward_ue2ut(mesh_file, ue_boundary_pts, ue_boundary_val, gt_val=0, sigma_t=0.8, multi_flag=True, gdim=3):
    r'''
    \nabla \cdot (M_T \nabla u_T) = 0 x \in T
    u_e = u_T x \in \partial H
    (M_T \nabla u_T) \cdot n_T = 0 x \in \partial B
    '''
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    if multi_flag:
        torso_marker = [1, 3, 4]
    else:
        torso_marker = [1]
    # mesh of torso
    cell_indices = np.concatenate([cell_markers.find(i) for i in torso_marker])
    subdomain_torso, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_indices)

    V = functionspace(subdomain_torso, ("Lagrange", 1))
    
    V1 = functionspace(domain, ("DG", 0, (tdim, tdim)))
    V2 = functionspace(subdomain_torso, ("DG", 0, (tdim, tdim)))
    def rho1(x):
        tensor = np.eye(tdim) * sigma_t
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
    
    M_expand = Function(V1)
    M_expand.interpolate(rho1, cell_markers.find(1))
    if cell_markers.find(3).any():
        if multi_flag == True:
            M_expand.interpolate(rho3, cell_markers.find(3))
        else:
            M_expand.interpolate(rho1, cell_markers.find(3))
    if cell_markers.find(4).any():
        if multi_flag == True:
            M_expand.interpolate(rho4, cell_markers.find(4))
        else:
            M_expand.interpolate(rho1, cell_markers.find(4))

    M = Function(V2)
    # 获取子网格单元对应的主单元索引
    parent_cells = sub_to_parent

    # 取出 M_expand 在主域上的值
    parent_values = M_expand.x.array.reshape(-1, tdim, tdim)

    # 直接按映射复制（DG0 情况下单元常值，对应关系一一对应）
    sub_values = parent_values[parent_cells]
    M.x.array[:] = sub_values.flatten()
    
    u_bc = Function(V)
    pts_f = V.tabulate_dof_coordinates()
    diff = ue_boundary_pts[:, None, :] - pts_f[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    # 每行最小值位置（即最近的行号）
    idx = np.argmin(dist, axis=1)

    gt = Constant(subdomain_torso, PETSc.ScalarType(0.0))

    dx = Measure("dx", domain=subdomain_torso)
    ds = Measure("ds", domain=subdomain_torso)
    
    v = TestFunction(V)
    u = TrialFunction(V)

    a_element = dot(grad(v), dot(M, grad(u))) * dx
    L = gt * v * ds

    u_bc = Function(V)
    u_bc.x.array[:] = 0.0 
    ut_f_data = []
    if ue_boundary_val.ndim == 1:
        ue_boundary_val  = ue_boundary_val.reshape(1,-1)
    total_num = len(ue_boundary_val)
    for i in range(total_num):
        u_bc.x.array[idx] = ue_boundary_val[i].reshape(-1)
        bc = dirichletbc(u_bc, idx)
        bcs = [bc]
        problem = LinearProblem(a_element, L, bcs=bcs, 
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_f = problem.solve()
        ut_f_data.append(u_f.x.array.copy())
    ut_f_data = np.array(ut_f_data)
    return ut_f_data, V

def forward_tmp(mesh_file, v_data, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, multi_flag=True, gdim=3):
    ue_f_data, ue_functionspace = forward_tmp2ue(mesh_file, v_data, sigma_i=sigma_i, sigma_e=sigma_e, gdim=gdim)
    ue_boundary_pts = inner_boundary_pts_from_mesh(mesh_file, gdim=gdim)
    ue_boundary_val = extract_data_from_function(ue_f_data, ue_functionspace, ue_boundary_pts)
    ut_f_data, ut_functionspace = forward_ue2ut(mesh_file, ue_boundary_pts, ue_boundary_val, sigma_t=sigma_t, multi_flag=multi_flag, gdim=gdim)
    return ut_f_data, ut_functionspace

def compute_d_from_tmp(mesh_file, v_data, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, multi_flag=True):
    ut_f_data, ut_functionspace = forward_tmp(mesh_file, v_data, sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t, multi_flag=multi_flag)
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    points = np.array(geom['geom_thorax']['pts'])
    d_data = extract_data_from_function(ut_f_data, ut_functionspace, points)
    return d_data

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    gdim = 3
    T = 100
    step_per_timeframe = 2
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=T, step_per_timeframe=step_per_timeframe)
    ue_f_data, ue_functionspace = forward_tmp2ue(mesh_file, v_data, gdim=gdim)
    ue_boundary_pts = inner_boundary_pts_from_mesh(mesh_file, gdim=gdim)
    ue_boundary_val = extract_data_from_function(ue_f_data, ue_functionspace, ue_boundary_pts)
    ut_f_data, ut_functionspace = forward_ue2ut(mesh_file, ue_boundary_pts, ue_boundary_val, gdim=gdim)