import sys

import numpy as np
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import Mesh

sys.path.append('.')
from utils.function_tools import eval_function

MARKER_EPI = 1
MARKER_MID = 0
MARKER_ENDO = -1

TAU_CLOSE_EPI = 145
TAU_CLOSE_MID = 150
TAU_CLOSE_ENDO = 155
TAU_CLOSE_SHIFT = 20

TAU_IN = 0.4
TAU_IN_ISCHEMIA = 0.3

D_VAL = 1e-1
D_VAL_ISCHEMIA = 5e-2
D_VAL_SCAR = 0

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

def build_tau_close(marker_function: Function, condition: ischemia_condition, ischemia=False, vary=False):
    f_space = marker_function.function_space
    tau_close = Function(f_space)

    if vary:
        marker_f = marker_function.x.array.round()
        tau_close.x.array[:] = np.where(marker_f == MARKER_EPI,
                                        TAU_CLOSE_EPI,
                                        np.where(marker_f == MARKER_MID,
                                                 TAU_CLOSE_MID,
                                                 TAU_CLOSE_ENDO))
    else:
        tau_close.x.array[:] = TAU_CLOSE_MID

    if ischemia:
        coords = f_space.tabulate_dof_coordinates().T

        mask = condition(coords)
        tau_close.x.array[:] = tau_close.x.array + TAU_CLOSE_SHIFT * mask

    return tau_close


def build_tau_in(f_space: functionspace, condition: ischemia_condition, ischemia=False):
    tau_in = Function(f_space)

    if ischemia:
        condition.u_ischemia = TAU_IN_ISCHEMIA
        condition.u_healthy = TAU_IN
        coords = f_space.tabulate_dof_coordinates().T
        tau_in_smooth = condition(coords)
        tau_in.x.array[:] = tau_in_smooth
    else:
        tau_in.x.array[:] = TAU_IN

    condition.u_ischemia = 1.0
    condition.u_healthy = 0.0

    return tau_in


def build_D(f_space: functionspace, condition: ischemia_condition, scar=False, ischemia=False):
    D = Function(f_space)
    coords = f_space.tabulate_dof_coordinates().T

    if scar:
        condition.u_ischemia = D_VAL_SCAR
        condition.u_healthy = D_VAL
        D_smooth = condition(coords)
        D.x.array[:] = D_smooth
    elif ischemia:
        condition.u_ischemia = D_VAL_ISCHEMIA
        condition.u_healthy = D_VAL
        D_smooth = condition(coords)
        D.x.array[:] = D_smooth
    else:
        D.x.array[:] = D_VAL

    condition.u_ischemia = 1.0
    condition.u_healthy = 0.0

    return D


def build_Mi(domain: Mesh, condition: ischemia_condition, sigma_i=0.4, scar=False, ischemia=False):
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

def build_Me(domain: Mesh, condition: ischemia_condition, sigma_e=0.8, scar=False, ischemia=False):
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

def build_M(domain: Mesh, cell_markers, condition: ischemia_condition, multi_flag, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, scar=False, ischemia=False):
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

def get_activation_dict():
    # activation_dict = {
        # 8 : np.array([57, 51.2, 15]),
        # 14.4 : np.array([30.2, 45.2, -30]),
        # 14.5 : np.array([12.8, 54.2, -15]),
        # 18.7 : np.array([59.4, 29.8, 15]),
        # 23.5 : np.array([88.3, 41.2, -37.3]),
        # 34.9 : np.array([69.1, 27.1, -30]),
        # 45.6 : np.array([48.4, 40.2, -37.5])
    # }
    import h5py
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    ventricle_pts = np.array(geom['geom_ventricle']['pts'])
    right_cavity_pts = np.array(geom['geom_rcav']['pts'])
    left_cavity_pts = np.array(geom['geom_lcav']['pts'])

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
        if coord.tolist() in right_cavity_pts.tolist() or coord.tolist() in left_cavity_pts.tolist():
           activation_dict[time] = coord
    
    return activation_dict