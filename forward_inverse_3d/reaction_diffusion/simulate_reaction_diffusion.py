from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from dolfinx.fem import functionspace, Function, form
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector
from ufl import TrialFunction, TestFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from utils.function_tools import assign_function
from utils.ventricular_segmentation_tools import distinguish_epi_endo
from utils.simulate_tools import build_tau_close, build_tau_in, build_D, get_activation_dict, ischemia_condition

def compute_v_based_on_reaction_diffusion(mesh_file, gdim=3,
                                          ischemia_flag=False, scar_flag=False,
                                          center_ischemia=np.array([80.4, 19.7, -15.0]), 
                                          radius_ischemia=30,
                                          ischemia_epi_endo=[-1],
                                          u_peak_ischemia_val=0.9, u_rest_ischemia_val=0.1,
                                          T=500, step_per_timeframe=4,
                                          v_min=-90, v_max=10,
                                          tau_close_vary=False,
                                          affect_D=True, affect_tau_in=True, affect_tau_close=True,
                                          activation_dict_origin=None,
                                          stim_radius=5.0, stim_strength=0.5,
                                          tau_close_endo=155, tau_close_mid=150,
                                          tau_close_epi=145, tau_close_shift=20,
                                          tau_in_val=0.4, tau_in_ischemia=1,
                                          D_val=1e-1, D_val_ischemia=5e-2, D_val_scar=0
                                          ):
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

    condition = ischemia_condition(u_ischemia=1.0, u_healthy=0.0,
                                   center=center_ischemia,
                                   r=radius_ischemia,
                                   marker_function=marker_function,
                                   ischemia_epi_endo=ischemia_epi_endo)
    
    D = build_D(V_piecewise, 
                condition=condition, 
                scar=scar_flag and affect_D, 
                ischemia=ischemia_flag and affect_D,
                D_val=D_val, 
                D_val_ischemia=D_val_ischemia, 
                D_val_scar=D_val_scar)
    tau_out = 10
    tau_open = 130
    tau_close = build_tau_close(marker_function, 
                                condition, 
                                ischemia=ischemia_flag and affect_tau_close, 
                                vary=tau_close_vary,
                                tau_close_endo=tau_close_endo,
                                tau_close_mid=tau_close_mid,
                                tau_close_epi=tau_close_epi,
                                tau_close_shift=tau_close_shift)
    tau_in = build_tau_in(V, 
                          condition, 
                          ischemia=ischemia_flag and affect_tau_in,
                          tau_in_val=tau_in_val, tau_in_ischemia=tau_in_ischemia)
    u_crit = 0.13

    u_peak = Function(V)
    u_rest = Function(V)
    u_n = Function(V)
    v_n = Function(V)
    uh = Function(V)
    J_stim = Function(V)
    J_stim_plus = Function(V)
    if ischemia_flag:
        condition.u_ischemia = u_peak_ischemia_val
        condition.u_healthy = 1.0
        u_peak.interpolate(condition)
        
        condition.u_ischemia = u_rest_ischemia_val
        condition.u_healthy = 0.0
        u_rest.interpolate(condition)
        u_n.interpolate(condition)
        uh.interpolate(condition)

        condition.u_ischemia = 1.0
        condition.u_healthy = 0.0
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

    
    if activation_dict_origin is None:
        activation_dict_origin = get_activation_dict(mesh_file)
        
    from collections import defaultdict
    activation_dict = defaultdict(list)

    for k, v in activation_dict_origin.items():
        new_k = int(k * step_per_timeframe)
        if not isinstance(v, list):
            v = [v]
        activation_dict[new_k].extend(v)

    # ---------------------------
    u_data = []

    u_data.append(u_n.x.array.copy())
    # 初始化激活计时器
    activation_timers = {}
    # 新增：初始化已激活标志数组和激活时刻数组
    activated = np.zeros_like(u_n.x.array, dtype=bool)
    activation_time = np.full_like(u_n.x.array, np.nan, dtype=float)

    for i in range(num_steps):
        t += dt

        # 清零 J_stim
        J_stim.x.array[:] = np.zeros(J_stim.x.array.shape)

        # 检查是否有新的激活点
        if i in activation_dict:
            activation_timers[i] = 5 * step_per_timeframe

        # 更新激活计时器并累加刺激电流
        for key in list(activation_timers.keys()):
            activation_timers[key] -= 1
            if activation_timers[key] <= 0:
                del activation_timers[key]
            else:
                center_coords = activation_dict[key]  # Now a list of coordinates

                # interpolate 要求返回每个查询点的值（x 的形状为 (gdim, npoints)）
                def stim_indicator(x, centers=center_coords, r=stim_radius, s=stim_strength):
                    pts = x.T  # (npoints, gdim)
                    centers_arr = np.array(centers)  # (ncenters, gdim)
                    diff = pts[:, None, :] - centers_arr[None, :, :]
                    d2 = np.sum(diff**2, axis=2)  # (npoints, ncenters)

                    # 找每个点与最近中心的距离平方
                    dmin = np.min(d2, axis=1)  # (npoints,)

                    return np.where(dmin <= r**2, s, 0)

                # 累加刺激电流
                J_stim_plus.interpolate(stim_indicator)
                J_stim.x.array[:] += J_stim_plus.x.array[:]

        # 组装并求解
        with b_u.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b_u, linear_form_u)
        solver.solve(b_u, uh.vector)


        # 防止二次激活：只允许未激活点被激活，并记录激活时刻
        newly_activated = (~activated) & (uh.x.array > 0.8)
        activated[newly_activated] = True
        activation_time[newly_activated] = t  # 记录首次激活时刻

        # 激活1秒后不许u值上升
        lock_mask = activated & (t - activation_time >= 10)
        # uh.x.array[lock_mask & (uh.x.array > u_n.x.array)] = u_n.x.array[lock_mask & (uh.x.array > u_n.x.array)]

        u_n.x.array[:] = uh.x.array
        v_n.x.array[:] = v_n.x.array + dt * np.where(u_n.x.array < u_crit, (1 - v_n.x.array) / tau_open, -v_n.x.array / tau_close.x.array)

        u_data.append(u_n.x.array.copy())

    u_data = np.array(u_data)
    # u_data = np.where(u_data > 1, 1, u_data)
    # u_data = np.where(u_data < 0, 0, u_data)
    u_data = u_data * (v_max - v_min) + v_min
    
    return u_data, None, None