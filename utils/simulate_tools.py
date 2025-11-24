import numpy as np
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import Mesh
from utils.function_tools import eval_function

MARKER_EPI = 1
MARKER_MID = 0
MARKER_ENDO = -1

MI_ISCHEMIA_FACTOR = 0.9
MI_SCAR_FACTOR = 0.0
ME_ISCHEMIA_FACTOR = 0.6
ME_SCAR_FACTOR = 0.3

CAVITY_FACTOR = 3
LUNG_FACTOR = 0.2


class ischemia_condition():
    def __init__(self, u_ischemia, u_healthy, center, r, marker_function, ischemia_epi_endo,sigma=3):
        self.u_ischemia = u_ischemia
        self.u_healthy = u_healthy
        self.center = center
        self.r = r
        self.sigma = sigma
        self.marker_function = marker_function
        self.ischemia_epi_endo = ischemia_epi_endo
    def __call__(self, x):
        marker_value = eval_function(self.marker_function, x.T).ravel()
        distance = np.sqrt(np.sum((x.T - self.center)**2, axis=1))

        # 只在选定层参与缺血（如 -1,0,1）
        layer_mask = np.isin(marker_value.round(), self.ischemia_epi_endo)

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

def build_tau_close(marker_function: Function, 
                    condition: ischemia_condition, 
                    ischemia=False, 
                    vary=False,
                    tau_close_endo=155,
                    tau_close_mid=150,
                    tau_close_epi=145,
                    tau_close_shift=20):
    f_space = marker_function.function_space
    tau_close = Function(f_space)

    if vary:
        marker_f = marker_function.x.array.round()
        tau_close.x.array[:] = np.where(marker_f == MARKER_EPI,
                                        tau_close_epi,
                                        np.where(marker_f == MARKER_MID,
                                                 tau_close_mid,
                                                 tau_close_endo))
        
        # from utils.ventricular_segmentation_tools import separate_lv_rv
        # from utils.function_tools import fspace2mesh
        # _, _, _, rv_mask = separate_lv_rv(r"machine_learning/data/mesh/mesh_multi_conduct_ecgsim.msh",3)
        
        # fspace2mesh_map = fspace2mesh(f_space)
        # mesh2fspace_map = np.argsort(fspace2mesh_map)
        # rv_idx = np.where(rv_mask)[0]
        # rv_dof_idx = mesh2fspace_map[rv_idx]
        # tau_close.x.array[rv_dof_idx] = tau_close_mid

        # from utils.visualize_tools import plot_f_on_domain
        # plot_f_on_domain(f_space.mesh, tau_close, title="tau_close")

    else:
        tau_close.x.array[:] = tau_close_mid

    if ischemia:
        coords = f_space.tabulate_dof_coordinates().T

        mask = condition(coords)
        tau_close.x.array[:] = tau_close.x.array + tau_close_shift * mask

    return tau_close


def build_tau_in(f_space: functionspace, 
                 condition: ischemia_condition, 
                 ischemia=False,
                 tau_in_val=0.4,
                 tau_in_ischemia=1):
    tau_in = Function(f_space)

    if ischemia:
        condition.u_ischemia = tau_in_ischemia
        condition.u_healthy = tau_in_val
        coords = f_space.tabulate_dof_coordinates().T
        tau_in_smooth = condition(coords)
        tau_in.x.array[:] = tau_in_smooth
    else:
        tau_in.x.array[:] = tau_in_val

    condition.u_ischemia = 1.0
    condition.u_healthy = 0.0

    return tau_in


def build_D(f_space: functionspace, 
            condition: ischemia_condition, 
            scar=False, ischemia=False,
            D_val=1e-1, D_val_ischemia=5e-2, D_val_scar=0):
    D = Function(f_space)
    coords = f_space.tabulate_dof_coordinates().T

    if scar:
        condition.u_ischemia = D_val_scar
        condition.u_healthy = D_val
        D_smooth = condition(coords)
        D.x.array[:] = D_smooth
    elif ischemia:
        condition.u_ischemia = D_val_ischemia
        condition.u_healthy = D_val
        D_smooth = condition(coords)
        D.x.array[:] = D_smooth
    else:
        D.x.array[:] = D_val

    condition.u_ischemia = 1.0
    condition.u_healthy = 0.0

    return D


def build_Mi(domain: Mesh, 
             condition: ischemia_condition, 
             sigma_i=0.4, 
             scar=False, ischemia=False):
    tdim = domain.topology.dim
    f_space = functionspace(domain, ("DG", 0, (tdim, tdim)))
    coords = f_space.tabulate_dof_coordinates().T

    tensor = np.eye(tdim) * sigma_i
    n_dofs = coords.shape[1]
    values = np.tile(tensor.reshape(-1, 1), n_dofs)

    if ischemia:
        mask = condition(coords)  # 连续0~1 掩码
        factor = 1 - (1 - MI_ISCHEMIA_FACTOR) * mask
    elif scar:
        mask = condition(coords)
        factor = 1 - (1 - MI_SCAR_FACTOR) * mask
    else:
        factor = np.ones(coords.shape[1])

    values *= factor
    Mi = Function(f_space)
    Mi.x.array[:] = values.flatten(order='F')
    return Mi


def build_Me(domain: Mesh, 
             condition: ischemia_condition, 
             sigma_e=0.8, 
             scar=False, ischemia=False):
    tdim = domain.topology.dim
    f_space = functionspace(domain, ("DG", 0, (tdim, tdim)))
    coords = f_space.tabulate_dof_coordinates().T

    tensor = np.eye(tdim) * sigma_e
    n_dofs = coords.shape[1]
    values = np.tile(tensor.reshape(-1, 1), n_dofs)

    if ischemia:
        mask = condition(coords)
        factor = 1 - (1 - ME_ISCHEMIA_FACTOR) * mask
    elif scar:
        mask = condition(coords)
        factor = 1 - (1 - ME_SCAR_FACTOR) * mask
    else:
        factor = np.ones(coords.shape[1])

    values *= factor
    Me = Function(f_space)
    Me.x.array[:] = values.flatten(order='F')
    return Me


def build_M(domain: Mesh, 
            cell_markers, multi_flag,
            condition: ischemia_condition, 
            sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, 
            scar=False, ischemia=False):
    tdim = domain.topology.dim
    f_space = functionspace(domain, ("DG", 0, (tdim, tdim)))
    M = Function(f_space)

    def rho1(x):
        tensor = np.eye(tdim) * sigma_t
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    def rho2(x):
        tensor1 = np.eye(tdim) * sigma_i
        tensor2 = np.eye(tdim) * sigma_e
        n_dofs = x.shape[1]
        values1 = np.tile(tensor1.reshape(-1, 1), n_dofs)
        values2 = np.tile(tensor2.reshape(-1, 1), n_dofs)
        if ischemia:
            mask_local = condition(x)  # 使用局部掩码
            factor_local_1 = 1 - (1 - MI_ISCHEMIA_FACTOR) * mask_local
            factor_local_2 = 1 - (1 - ME_ISCHEMIA_FACTOR) * mask_local
        elif scar:
            mask_local = condition(x)
            factor_local_1 = 1 - (1 - MI_SCAR_FACTOR) * mask_local
            factor_local_2 = 1 - (1 - ME_SCAR_FACTOR) * mask_local
        else:
            factor_local_1 = np.ones(n_dofs)
            factor_local_2 = np.ones(n_dofs)
        values1 *= factor_local_1
        values2 *= factor_local_2
        values = values1 + values2
        return values
    def rho3(x):
        tensor = np.eye(tdim) * sigma_t * LUNG_FACTOR
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    def rho4(x):
        tensor = np.eye(tdim) * sigma_t * CAVITY_FACTOR
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    
    M.interpolate(rho1, cell_markers.find(1))
    M.interpolate(rho2, cell_markers.find(2))
    if cell_markers.find(3).any():
        if multi_flag == True:
            M.interpolate(rho3, cell_markers.find(3))
        else:
            M.interpolate(rho1, cell_markers.find(3))
    if cell_markers.find(4).any():
        if multi_flag == True:
            M.interpolate(rho4, cell_markers.find(4))
        else:
            M.interpolate(rho1, cell_markers.find(4))
    return M


def get_activation_dict(mesh_file, target_marker=2, gdim=3, mode='ENDO', threshold=100):

    from dolfinx.io import gmshio
    from mpi4py import MPI
    from dolfinx.mesh import create_submesh

    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(target_marker))

    import h5py
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    ventricle_pts = np.array(geom['geom_ventricle']['pts'])

    activation_times = h5py.File(r'forward_inverse_3d/data/activation_times_ecgsim.mat', 'r')
    activation = np.array(activation_times['dep']).reshape(-1)

    assert ventricle_pts.shape[0] == activation.shape[0], \
        "Number of ventricle points does not match number of activation times"
    
    assert np.unique(activation).shape[0] == activation.shape[0]

    # create dict time -> coords
    activation_dict = {}
    for i in range(ventricle_pts.shape[0]):
        time = activation[i]
        coord = ventricle_pts[i]
        activation_dict[time] = coord

    if mode == 'FULL':
        target_coords = subdomain_ventricle.geometry.x
    elif mode == 'ENDO':
        from utils.ventricular_segmentation_tools import distinguish_epi_endo
        epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)
        endo_idx = np.where(np.isclose(epi_endo_marker, -1.0))[0]
        target_coords = subdomain_ventricle.geometry.x[endo_idx, :]
    else:
        target_coords = None
    
    if target_coords is not None:
        activation_dict = compute_full_activation_dict(activation_dict, target_coords, threshold)
    
    return activation_dict

def compute_full_activation_dict(activation_dict, pts, threshold, power=2.0):
    """
    使用 IDW (Inverse Distance Weighting) 对 pts 进行激活时间插值。
    不再使用 activation_dict 的 key 做插值点，只用它存的三维坐标。
    
    参数：
        activation_dict : dict
            {activation_time : point3d}
        pts : array-like
            待插值的点 (N,3)
        power : float
            IDW 幂指数, 默认 2
        
    返回：
        dict, 结构为 { interpolated_time : point }
    """

    # 空输入直接返回
    if len(activation_dict) == 0:
        return {}

    # ---- 提取已知的时间与坐标 ----
    known_times = np.array(list(activation_dict.keys()), dtype=float)   # (M,)
    known_pts   = np.array(list(activation_dict.values()), dtype=float) # (M,3)

    # ---- 保证 pts 形状正确 ----
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    # ---- 向量化距离计算 ----
    # diff[i,j,:] = pts[i] - known_pts[j]
    diff = pts[:, None, :] - known_pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)   # (N,M)

    eps = 1e-12
    result = {}
    used_keys = set()

    # ---- 对每个目标点做插值 ----
    for i in range(dists.shape[0]):
        row = dists[i]

        # -------- 精确匹配：距离为 0 --------
        exact = np.where(row <= eps)[0]
        if exact.size > 0:
            t_val = float(known_times[exact[0]])

        else:
            # -------- IDW 权值 --------
            weights = 1.0 / (row ** power)
            wsum = weights.sum()

            if wsum <= 0:
                t_val = float(np.mean(known_times))
            else:
                t_val = float((weights * known_times).sum() / wsum)

        # -------- 避免 key 冲突 --------
        key = t_val
        offset = 0
        while key in used_keys:
            offset += 1
            key = t_val + 1e-8 * offset

        used_keys.add(key)
        result[key] = pts[i].copy()

    # save the ones that key less than threshold
    result = {k: v for k, v in result.items() if k < threshold}
    return result