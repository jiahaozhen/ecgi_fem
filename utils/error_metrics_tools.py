import numpy as np
from scipy.stats import pearsonr

def compute_error_with_v(v_exact, v_result, function_space, v_rest_healthy, v_rest_ischemia, v_peak_healthy, v_peak_ischemia):
    #ichemic region
    ischemia_exact_condition = ((v_exact > v_rest_ischemia-5) & (v_exact < v_rest_ischemia+5)| 
                                (v_exact > v_peak_ischemia-5) & (v_exact < v_peak_ischemia+5))
    marker_ischemia_exact = np.where(ischemia_exact_condition, 1, 0)
    ischemia_result_condition = ((v_result > v_rest_ischemia-5) & (v_result < v_rest_ischemia+5)| 
                                (v_result > v_peak_ischemia-5) & (v_result < v_peak_ischemia+5))
    marker_ischemia_result = np.where(ischemia_result_condition, 1, 0)
    #activate region
    activate_exact_condition = v_exact > (v_peak_healthy + v_rest_healthy)/2
    marker_activate_exact = np.where(activate_exact_condition, 1, 0)
    activate_result_condition = v_result > (v_peak_healthy + v_rest_healthy)/2
    marker_activate_result = np.where(activate_result_condition, 1, 0)

    coordinates = function_space.tabulate_dof_coordinates()
    coordinates_ischemia_exact = coordinates[np.where(marker_ischemia_exact == 1)]
    coordinates_ischemia_result = coordinates[np.where(marker_ischemia_result == 1)]
    coordinates_activate_exact = coordinates[np.where(marker_activate_exact == 1)]
    coordinates_activate_result = coordinates[np.where(marker_activate_result == 1)]

    cm_ischemia_exact = np.mean(coordinates_ischemia_exact, axis=0)
    cm_ischemia_result = np.mean(coordinates_ischemia_result, axis=0)
    cm_activate_exact = np.mean(coordinates_activate_exact, axis=0)
    cm_activate_result = np.mean(coordinates_activate_result, axis=0)

    cm_error_ischemia = np.linalg.norm(cm_ischemia_exact-cm_ischemia_result)   
    cm_error_activate = np.linalg.norm(cm_activate_exact-cm_activate_result)

    return (cm_error_ischemia, cm_error_activate)

def compute_error(v_exact, phi_result):
    marker_exact = np.full(v_exact.x.array.shape, 0)
    marker_exact[v_exact.x.array > -89.9] = 1
    marker_result = np.full(phi_result.x.array.shape, 0)
    marker_result[phi_result.x.array < 0] = 1

    coordinates = v_exact.function_space.tabulate_dof_coordinates()
    coordinates_ischemia_exact = coordinates[np.where(marker_exact == 1)]
    coordinates_ischemia_result = coordinates[np.where(marker_result == 1)]

    cm1 = np.mean(coordinates_ischemia_exact, axis=0)
    cm2 = np.mean(coordinates_ischemia_result, axis=0)
    cm = np.linalg.norm(cm1-cm2)

    if (coordinates_ischemia_result.size == 0):
        return (cm, None, None, None)
    
    # HaussDist
    hdxy = 0
    for coordinate in coordinates_ischemia_exact:
        hdy = np.min(np.linalg.norm(coordinate - coordinates_ischemia_result, axis=1))
        hdxy = max(hdxy, hdy)
    hdyx = 0
    for coordinate in coordinates_ischemia_result:
        hdx = np.min(np.linalg.norm(coordinate - coordinates_ischemia_exact, axis=1))
        hdyx = max(hdyx, hdx)
    hd = max(hdxy, hdyx)

    # SN false negative
    marker_exact_index = np.where(marker_exact == 1)[0]
    marker_result_index = np.where(marker_result == 1)[0]
    SN = 0
    for index in marker_exact_index:
        if index not in marker_result_index:
            SN = SN + 1
    SN = SN / np.shape(marker_exact_index)[0]

    # SP false positive
    SP = 0
    for index in marker_result_index:
        if index not in marker_exact_index:
            SP = SP + 1
    SP = SP / np.shape(marker_result_index)[0]

    return (cm, hd, SN, SP)

def compare_phi_one_timeframe(phi_exact, phi_result, coordinates = []):
    marker_exact = np.where(phi_exact < 0, 1, 0)
    marker_result = np.where(phi_result < 0, 1, 0)
    cc = np.corrcoef(marker_exact, marker_result)[0, 1]
    if coordinates != []:
        coordinates_ischemia_exact = coordinates[np.where(marker_exact == 1)]
        coordinates_ischemia_result = coordinates[np.where(marker_result == 1)]
        cm1 = np.mean(coordinates_ischemia_exact, axis=0)
        cm2 = np.mean(coordinates_ischemia_result, axis=0)
        cm = np.linalg.norm(cm1-cm2)
        return cc, cm
    return cc

def compute_cc(exact, result):
    cc = []
    for i in range(exact.shape[0]):
        cc.append(compare_phi_one_timeframe(exact[i], result[i]))
    return np.array(cc)

def compute_error_and_correlation(result: np.ndarray, ref: np.ndarray):
    assert len(result) == len(ref)
    relative_error = 0
    correlation_coefficient = 0
    for i in range(len(result)):
        result[i] += np.mean(ref[i]-result[i])
        relative_error += np.linalg.norm(result[i] - ref[i]) / np.linalg.norm(ref[i])
        correlation_matrix = np.corrcoef(result[i], ref[i])
        correlation_coefficient += correlation_matrix[0, 1]
    relative_error = relative_error/len(result)
    correlation_coefficient = correlation_coefficient/len(result)
    return relative_error, correlation_coefficient

def compute_error_phi(phi_exact: np.ndarray, phi_result: np.ndarray, function_space):
    marker_exact = np.where(phi_exact < 0, 1, 0)
    marker_result = np.where(phi_result < 0, 1, 0)
    coordinates = function_space.tabulate_dof_coordinates()
    coordinates_ischemia_exact = coordinates[np.where(marker_exact == 1)]
    coordinates_ischemia_result = coordinates[np.where(marker_result == 1)]
    cm1 = np.mean(coordinates_ischemia_exact, axis=0)
    cm2 = np.mean(coordinates_ischemia_result, axis=0)
    cm = np.linalg.norm(cm1-cm2)
    return cm

def compute_metrics(data_base, data_other):
    """
    计算单个 lc 和 lc_ratio 相对于基准的多指标相似性。
    data_base, data_other: shape = (n_leads, n_time)
    返回:
        {
          "corr": 平均皮尔逊相关,
          "rmse": 均方根误差,
          "rel_L2": 相对L2误差,
          "peak_shift": 平均峰值时间偏移（采样点数）
        }
    """
    n_leads = data_base.shape[1]
    r_list, rmse_list, relL2_list, peak_shift_list = [], [], [], []

    for i in range(n_leads):
        x, y = data_base[:,i], data_other[:,i]

        # 1. 皮尔逊相关系数
        if np.std(x) == 0 or np.std(y) == 0:
            r = np.nan
        else:
            r, _ = pearsonr(x, y)

        # 2. RMSE
        rmse = np.sqrt(np.mean((x - y) ** 2))

        # 3. 相对L2误差
        rel_L2 = np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-12)

        # 4. 峰值时间偏移
        peak_shift = np.argmax(y) - np.argmax(x)

        r_list.append(r)
        rmse_list.append(rmse)
        relL2_list.append(rel_L2)
        peak_shift_list.append(peak_shift)

    metrics = {
        "corr": np.nanmean(r_list),
        "rmse": np.mean(rmse_list),
        "rel_L2": np.mean(relL2_list),
        "peak_shift": np.mean(np.abs(peak_shift_list))
    }
    return metrics

def compute_metrics_for_timestep(d_base, d_other):
    """
    计算单个时间步长相对于基准的相关系数。
    d_base, d_other: shape = (n_leads, n_time)
    返回:
        {
          "corr": 平均皮尔逊相关
        }
    """
    n_leads = d_base.shape[1]
    r_list = []

    for i in range(n_leads):
        x, y = d_base[:, i], d_other[:, i]

        # 对齐时间序列长度并进行伸缩
        if len(x) != len(y):
            x = np.interp(np.linspace(0, 1, len(y)), np.linspace(0, 1, len(x)), x)

        # 计算皮尔逊相关系数
        if np.std(x) == 0 or np.std(y) == 0:
            r = np.nan
        else:
            r, _ = pearsonr(x, y)

        r_list.append(r)

    metrics = {
        "corr": np.nanmean(r_list)
    }
    return metrics

def compute_convergence_metrics(data_dict, base_lc=20, base_lc_ratio=1):
    """
    计算不同 lc 和 lc_ratio 下信号相对基准的多指标相似度
    """
    base_data = data_dict[(base_lc, base_lc_ratio)]
    summary = {}

    print("\n=== Mesh Convergence Analysis ===")
    for (lc, lc_ratio), data in sorted(data_dict.items()):
        if (lc, lc_ratio) == (base_lc, base_lc_ratio):
            continue

        metrics = compute_metrics(base_data, data)
        summary[(lc, lc_ratio)] = metrics
        print(
            f"lc={lc:>3d}, lc_ratio={lc_ratio:>2d} vs base lc={base_lc}, lc_ratio={base_lc_ratio}: "
            f"corr={metrics['corr']:.3f}, "
            f"relL2={metrics['rel_L2']:.3e}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"Δt_peak={metrics['peak_shift']:.2f}"
        )

    return summary