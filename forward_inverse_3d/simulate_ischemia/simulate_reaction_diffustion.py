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
from utils.function_tools import eval_function, assign_function
from utils.ventricular_segmentation_tools import distinguish_epi_endo
from forward_inverse_3d.simulate_ischemia.simulate_tools import build_tau_close, build_tau_in, build_D, get_activation_dict, ischemia_condition

def compute_v_based_on_reaction_diffusion(mesh_file, gdim=3,
                                          ischemia_flag=False, scar_flag=False,
                                          center_ischemia=np.array([32.1, 71.7, 15]), 
                                          radius_ischemia=20,
                                          ischemia_epi_endo=[-1, 0, 1],
                                          u_peak_ischemia_val=0.9, u_rest_ischemia_val=0.1,
                                          T=120, step_per_timeframe=10,
                                          v_min=-90, v_max=10,
                                          tau_close_vary=False):
    '''
    ischemia affect to D tau_close tau_in
    '''
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    V = functionspace(subdomain_ventricle, ("Lagrange", 1))
    V_piecewise = functionspace(subdomain_ventricle, ("DG", 0))

    t = 0
    num_steps = T * step_per_timeframe
    dt = T / num_steps  # time step size
    
    # 1 is epicardium, 0 is mid-myocardial, -1 is endocardium
    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)
    marker_function = Function(V)
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
    
    D = build_D(V_piecewise, condition=ischemia_condition, scar=scar_flag, ischemia=ischemia_flag)
    tau_out = 10
    tau_open = 130
    tau_close = build_tau_close(marker_function, ischemia_condition, ischemia=ischemia_flag, vary=tau_close_vary)
    tau_in = build_tau_in(V, ischemia_condition, ischemia=ischemia_flag)
    u_crit = 0.13

    u_peak = Function(V)
    u_rest = Function(V)
    u_n = Function(V)
    v_n = Function(V)
    uh = Function(V)
    J_stim = Function(V)
    J_stim_plus = Function(V)
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
    
    dx1 = Measure("dx", domain=subdomain_ventricle)
    u, v = TrialFunction(V), TestFunction(V)
    a_u = u * v * dx1 + dt * D * dot(grad(u), grad(v)) * dx1
    L_u = u_n * v * dx1 + dt * (v_n * (u_peak - u_n) * (u_n - u_rest) * (u_n - u_rest) / tau_in - (u_n - u_rest) / tau_out  + J_stim) * v * dx1
    
    bilinear_form = form(a_u)
    linear_form_u = form(L_u)

    A = assemble_matrix(bilinear_form)
    A.assemble()
    solver = PETSc.KSP().create(subdomain_ventricle.comm)
    solver.setOperators(A)
    solver.setType("cg")  # 改为共轭梯度法
    solver.getPC().setType("gamg")  # 或 "ilu", "gamg"
    solver.setTolerances(rtol=1e-6)
    
    b_u = create_vector(linear_form_u)

    # ---------------------------
    # activation 座标（物理坐标）
    # 保持原始字典的物理坐标定义，不再把它映射到单个 DOF/节点索引
    activation_centers_original = get_activation_dict()
    # 把时间转换为步数索引，但保留物理坐标作为激活中心
    
    activation_centers = { int(k * step_per_timeframe) : v for k, v in activation_centers_original.items() }

    # 刺激的物理半径和强度（你可以根据需要调整）
    stim_radius = 5.0         # mm（或与网格同单位）----> 保持为物理长度，不随网格变化
    stim_strength = 0.5       # A/m^3 或数值强度（与原来数值对齐）

    # ---------------------------
    u_data = []
    u_data.append(u_n.x.array.copy())
    # 初始化激活计时器
    activation_timers = {}
    for i in range(num_steps):
        t += dt

        # 清零 J_stim
        J_stim.x.array[:] = np.zeros(J_stim.x.array.shape)

        # 检查是否有新的激活点
        if i in activation_centers:
            activation_timers[i] = 5 * step_per_timeframe

        # 更新激活计时器并累加刺激电流
        for key in list(activation_timers.keys()):
            activation_timers[key] -= 1
            if activation_timers[key] <= 0:
                del activation_timers[key]
            else:
                center_coord = activation_centers[key].astype(float)

                # interpolate 要求返回每个查询点的值（x 的形状为 (gdim, npoints)）
                def stim_indicator(x, c=center_coord, r=stim_radius, s=stim_strength):
                    # x: (gdim, npoints)
                    pts = x.T  # (npoints, gdim)
                    d2 = np.sum((pts - c)**2, axis=1)
                    return np.where(d2 <= r**2, np.full(pts.shape[0], s), np.zeros(pts.shape[0]))

                # 累加刺激电流
                J_stim_plus.interpolate(stim_indicator)
                J_stim.x.array[:] = J_stim.x.array + J_stim_plus.x.array
                

        # 组装并求解
        with b_u.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b_u, linear_form_u)
        solver.solve(b_u, uh.vector)
        
        if i > 150 * step_per_timeframe:
            uh.x.array[:] = np.where(uh.x.array-u_n.x.array > 0, u_n.x.array, uh.x.array)
        
        u_n.x.array[:] = uh.x.array
        v_n.x.array[:] = v_n.x.array + dt * np.where(u_n.x.array < u_crit, (1 - v_n.x.array) / tau_open, -v_n.x.array / tau_close.x.array)
        u_data.append(u_n.x.array.copy())

    u_data = np.array(u_data)
    u_data = np.where(u_data > 1, 1, u_data)
    u_data = np.where(u_data < 0, 0, u_data)
    u_data = u_data * (v_max - v_min) + v_min
    
    return u_data, None, None

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    step_per_timeframe = 4
    import time
    start_time = time.time()
    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, ischemia_flag=True, T=2000, step_per_timeframe=step_per_timeframe)
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time} seconds")
    from utils.visualize_tools import plot_v_random
    plot_v_random(v_data, step_per_timeframe=step_per_timeframe)