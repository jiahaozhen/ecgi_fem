import sys

from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import h5py

sys.path.append('.')
from forward_inverse_3d.simulate_ischemia.simulate_tools import build_Mi, build_M
from utils.function_tools import extract_data_from_function, assign_function, eval_function
from utils.ventricular_segmentation_tools import distinguish_epi_endo

def forward_tmp(mesh_file, v_data, 
                sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, 
                multi_flag=True, gdim=3, 
                ischemia_flag=False, scar_flag=False, 
                center_ischemia=np.array([32.1, 71.7, 15]), 
                radius_ischemia=10, 
                ischemia_epi_endo=[-1, 0, 1],
                solver=None):  # 添加 solver 参数
    '''
    consider influence of ischemia or scar on forward simulation
    different conductivity in different regions
    '''
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

    # 1 is epicardium, 0 is mid-myocardial, -1 is endocardium
    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)
    marker_function = Function(V2)
    assign_function(marker_function, np.arange(len(subdomain_ventricle.geometry.x)), epi_endo_marker)

    class ischemia_condition():
        def __init__(self, u_ischemia, u_healthy, center=center_ischemia, r=radius_ischemia, sigma=3):
            self.u_ischemia = u_ischemia
            self.u_healthy = u_healthy
            self.center = center
            self.r = r
            self.sigma = sigma
        def __call__(self, x):
            marker_value = eval_function(marker_function, x.T).ravel()
            distance = np.sqrt(np.sum((x.T - self.center)**2, axis=1))

            # 只在选定层参与缺血（如 -1,0,1）
            layer_mask = np.isin(marker_value.round(), ischemia_epi_endo)

            # 高斯平滑权重 (0~1)，sigma 控制过渡宽度
            # exp[-0.5*((d - r)/sigma)^2] → 软边界
            smooth_mask = np.exp(-0.5 * ((distance - self.r) / self.sigma) ** 2)
            smooth_mask[distance < self.r] = 1.0
            smooth_mask[distance > self.r + 3 * self.sigma] = 0.0

            # 只在目标层起效
            smooth_mask *= layer_mask.astype(float)

            # 输出连续电生理参数
            ret_value = self.u_healthy + (self.u_ischemia - self.u_healthy) * smooth_mask

            # ischemia_mask = (distance <= self.r) & layer_mask
            # ret_value = np.where(ischemia_mask, self.u_ischemia, self.u_healthy)

            return ret_value

    Mi = build_Mi(subdomain_ventricle, ischemia_condition, sigma_i=sigma_i, scar=scar_flag, ischemia=ischemia_flag)
    M = build_M(domain, cell_markers, multi_flag=multi_flag, condition=ischemia_condition, sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t, scar=scar_flag, ischemia=ischemia_flag)

    u = Function(V1)
    v = Function(V2)

    # A u = b
    # matrix A
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    dx1 = Measure("dx", domain = domain)
    a_element = dot(grad(v1), dot(M, grad(u1))) * dx1
    bilinear_form_a = form(a_element)
    A = assemble_matrix(bilinear_form_a)
    A.assemble()

    # b
    dx2 = Measure("dx", domain=subdomain_ventricle)
    b_element = -dot(grad(v1), dot(Mi, grad(v))) * dx2
    entity_map = {domain._cpp_object: ventricle_to_torso}
    linear_form_b = form(b_element, entity_maps=entity_map)
    b = create_vector(linear_form_b)

    # 使用传入的 solver
    if solver is None:
        solver = PETSc.KSP().create()
        solver.setType(PETSc.KSP.Type.CG)
        solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setOperators(A)

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
    return np.array(u_data), V1

def compute_d_from_tmp(mesh_file, v_data, 
                       sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, 
                       multi_flag=True, gdim=3, 
                       ischemia_flag=False, scar_flag=False, 
                       center_ischemia=np.array([32.1, 71.7, 15]), 
                       radius_ischemia=10, 
                       ischemia_epi_endo=[-1, 0, 1]):
    u_f_data, u_functionspace = forward_tmp(mesh_file, v_data, 
                                            sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t, 
                                            multi_flag=multi_flag, gdim=gdim,
                                            ischemia_flag=ischemia_flag, scar_flag=scar_flag, 
                                            center_ischemia=center_ischemia, 
                                            radius_ischemia=radius_ischemia, 
                                            ischemia_epi_endo=ischemia_epi_endo)
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    points = np.array(geom['geom_thorax']['pts'])
    d_data = extract_data_from_function(u_f_data, u_functionspace, points)
    return d_data

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    T = 500
    step_per_timeframe = 4
    from forward_inverse_3d.simulate_ischemia.simulate_reaction_diffustion import compute_v_based_on_reaction_diffusion
    import time
    start_time = time.time()
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                         T=T, 
                                                         step_per_timeframe=step_per_timeframe, 
                                                         ischemia_flag=False)
    end_time = time.time()
    print(f"Reaction-diffusion simulation time: {end_time - start_time} seconds")
    d_data = compute_d_from_tmp(mesh_file, v_data, ischemia_flag=False)
    end_time = time.time()
    print(f"Forward TMP to BSP simulation time: {end_time - start_time} seconds")
    from utils.visualize_tools import plot_bsp_on_standard12lead
    plot_bsp_on_standard12lead(d_data, step_per_timeframe=step_per_timeframe)