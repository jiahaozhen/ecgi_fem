import numpy as np

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

# function to compare exact phi and result phi
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