import numpy as np
from scipy.spatial import cKDTree
from dolfinx.fem import Function, FunctionSpace


# G function
def G(s):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > 0
    condition2 = s < 0
    result = np.zeros_like(s)
    result[condition1] = 1
    result[condition2] = 0
    return result


# G_tau function
def G_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 1
    result[condition2] = 0
    result[condition3] = 0.5 * (1 + s[condition3] / tau + (1 / np.pi) * np.sin(np.pi * s[condition3] / tau))
    return result


# delta_tau function
def delta_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 0
    result[condition2] = 0
    result[condition3] = (1 / (2 * tau)) * (1 + np.cos(np.pi * s[condition3] / tau))
    return result


# delta^{'}_tau function
def delta_deri_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 0
    result[condition2] = 0
    result[condition3] = -(np.pi / (2 * tau**2)) * np.sin(np.pi * s[condition3] / tau)
    return result


def get_activation_time_from_v(v_data: np.ndarray):
    v_deriviative = np.diff(v_data, axis=0)
    activation_time = np.argmax(v_deriviative, axis=0)
    return activation_time


def min_distance(coords, mask):
    """快速计算 coords 每个点到 mask==True 点集的最小距离(KDTree 加速)"""
    if not np.any(mask):
        return np.zeros(coords.shape[0])

    tree = cKDTree(coords[mask])
    # 查询每个 coords[i] 到 mask 点集的最近距离
    dist, _ = tree.query(coords, k=1, workers=-1)
    return dist


def compute_phi_with_v(v: np.ndarray, marker_ischemia: np.ndarray, function_space: FunctionSpace):
    num_time, num_nodes = v.shape
    coords = function_space.tabulate_dof_coordinates()

    # ---- 1) preprocess ----
    activation_time = get_activation_time_from_v(v)
    marker_ischemia = (marker_ischemia == 1)

    # ---- 2) 直接向量化构造 marker_activation ----
    # shape: (num_time, num_nodes)
    marker_activation = np.less_equal(
        activation_time[None, :],            # broadcast到时间维度
        np.arange(num_time)[:, None]         # 每个 timeframe
    )

    # ---- 3) 预计算 ischemia 距离（无时间变化）----
    min_iso = min_distance(coords, marker_ischemia)
    min_no_iso = min_distance(coords, ~marker_ischemia)

    # ---- 4) 构造 phi1（与时间无关，只是根据 mask 选取正负距离）----
    phi_1_template = np.where(marker_ischemia, -min_no_iso, min_iso)
    phi_1 = np.tile(phi_1_template, (num_time, 1))

    # ---- 5) phi2 逐时计算（需要根据激活 mask）----
    phi_2 = np.zeros_like(v)

    # 预先找最早激活时间
    min_act_time = np.min(activation_time)

    for t in range(num_time):
        act_mask = marker_activation[t]

        min_act = min_distance(coords, act_mask)
        min_no_act = min_distance(coords, ~act_mask)

        frame = np.where(act_mask, -min_no_act, min_act)

        # 处理全部为 0 的情况
        if np.all(frame == 0):
            if t < min_act_time + 5:
                frame[:] = 20
            else:
                frame[:] = -20

        phi_2[t] = frame

    # ---- 6) phi1 也处理零值（一次性）----
    zero_frame = np.all(phi_1 == 0, axis=1)
    phi_1[zero_frame] = 20

    return phi_1, phi_2


def v_data_argument(phi_1: np.ndarray, phi_2: np.ndarray, tau = 10, a1 = -90, a2 = -60, a3 = 10, a4 = -20):
    G_phi_1 = G_tau(phi_1, tau)
    G_phi_2 = G_tau(phi_2, tau)
    v = ((a1 * G_phi_2 + a3 * (1 - G_phi_2)) * G_phi_1 + 
         (a2 * G_phi_2 + a4 * (1 - G_phi_2)) * (1 - G_phi_1))
    return v


def compute_phi_with_activation(activation_f : Function, duration : int):
    phi = np.zeros((duration, len(activation_f.x.array)))
    activation_time = activation_f.x.array
    marker_activation = np.full_like(phi, False, dtype=bool)
    for i in range(phi.shape[1]):
        marker_activation[int(activation_time[i]):, i] = True
    coords = activation_f.function_space.tabulate_dof_coordinates()
    
    for timeframe in range(duration):
        
        min_act = min_distance(coords, marker_activation[timeframe])
        min_no_act = min_distance(coords, ~marker_activation[timeframe])
    
        phi[timeframe] = np.where(marker_activation[timeframe], -min_no_act, min_act)
        if (phi[timeframe] == 0).all():
            if timeframe < np.min(activation_time) + 5:
                phi[timeframe] = 20
            else:
                phi[timeframe] = -20
    
    return phi