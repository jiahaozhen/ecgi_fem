from ufl import grad
from dolfinx.mesh import locate_entities_boundary, meshtags, exterior_facet_indices, create_submesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.fem import functionspace, Function, Expression
from .normals_and_tangents import facet_vector_approximation
import numpy as np

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

def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s

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

def compute_phi_with_v(v, function_space, v_rest_healthy, v_rest_ischemia, v_peak_healthy, v_peak_ischemia):
    coordinates = function_space.tabulate_dof_coordinates()
    marker_ischemia = (((v > v_rest_ischemia - 5) & (v < v_rest_ischemia + 5)) |
                       ((v > v_peak_ischemia - 5) & (v < v_peak_ischemia + 5)))
    marker_activate = v > ((v_peak_healthy + v_rest_healthy) / 2)
    
    def min_distance(coords, mask):
        if np.any(mask):
            return np.min(np.linalg.norm(coords[:, None, :] - coords[mask], axis=2), axis=1)
        else:
            return np.zeros(len(coords))
    
    min_iso = min_distance(coordinates, marker_ischemia)
    min_no_iso = min_distance(coordinates, ~marker_ischemia)
    min_act = min_distance(coordinates, marker_activate)
    min_no_act = min_distance(coordinates, ~marker_activate)
    
    phi_1 = np.where(marker_ischemia, -min_no_iso, min_iso)
    phi_2 = np.where(marker_activate, -min_no_act, min_act)

    return phi_1, phi_2

def compute_phi_with_v_timebased(v, function_space, v_rest_ischemia, v_peak_ischemia):
    phi_1 = np.full_like(v, 0)
    phi_2 = np.full_like(v, 0)
    activation_time = get_activation_time_from_v(v)

    for i in range(v.shape[1]):
        phi_2[:activation_time[i], i] = 10
        phi_2[activation_time[i]:, i] = -10
        if  (min(v[:, i]) < v_rest_ischemia - 10 or max(v[:, i]) > v_peak_ischemia + 10) :
            phi_1[:, i] = 10
        else:
            phi_1[:, i] = -10
    # coordinates = function_space.tabulate_dof_coordinates()
    # marker_ischemia = np.full(v.shape[1], False,  dtype=bool)
    # marker_activation  = np.full_like(v, False, dtype=bool)
    # for i in range(v.shape[1]):
    #     marker_activation[activation_time[i]:, i] = True
    #     if  (min(v[:, i]) > v_rest_ischemia - 10 and max(v[:, i]) < v_peak_ischemia + 10) :
    #         marker_ischemia[i] = True

    # def min_distance(coords, mask):
    #     if np.any(mask):
    #         return np.min(np.linalg.norm(coords[:, None, :] - coords[mask], axis=2), axis=1)
    #     else:
    #         return np.zeros(len(coords))
    
    # min_iso = min_distance(coordinates, marker_ischemia)
    # min_no_iso = min_distance(coordinates, ~marker_ischemia)
    # for timeframe in range(v.shape[0]):
        
    #     min_act = min_distance(coordinates, marker_activation[timeframe])
    #     min_no_act = min_distance(coordinates, ~marker_activation[timeframe])
    
    #     phi_1[timeframe] = np.where(marker_ischemia[timeframe], -min_no_iso, min_iso)
    #     phi_2[timeframe] = np.where(marker_activation[timeframe], -min_no_act, min_act)

    return phi_1, phi_2

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

# function to comppare exact phi and result phi
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

def compute_normal(domain):
    tdim = domain.topology.dim
    boundary_index = locate_entities_boundary(domain, tdim - 3, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_points = np.array(domain.geometry.x[boundary_index])
    boundary_facets = exterior_facet_indices(domain.topology)
    b_tag = meshtags(domain, tdim - 1, boundary_facets, 1)
    V = functionspace(domain, ("Lagrange", 1, (tdim, )))
    n_f = facet_vector_approximation(V, b_tag, 1)

    return eval_function(n_f, boundary_points)

def compute_grad(u):
    domain = u.function_space.mesh
    tdim = domain.topology.dim
    V1 = functionspace(domain, ("Lagrange", 1))
    grad_u = Expression(grad(u), V1.element.interpolation_points())
    V2 = functionspace(domain, ("Lagrange", 1, (tdim, )))
    grad_u_f = Function(V2)
    grad_u_f.interpolate(grad_u)
    points = domain.geometry.x

    return eval_function(grad_u_f, points)

def submesh_node_index(domain, cell_markers, sub_tag):
    tdim = domain.topology.dim
    subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(sub_tag))
    subdomain_node_num = subdomain.topology.index_map(0).size_local
    subdomain.topology.create_connectivity(tdim, 0)
    domain.topology.create_connectivity(tdim, 0)

    sub2parent_node = np.ones((subdomain_node_num,), dtype=int) * -1

    domain_cell2point = domain.topology.connectivity(tdim, 0)
    subdomain_cell2point = subdomain.topology.connectivity(tdim, 0)
    for i, cell in enumerate(sub_to_parent):
        sub_cell_point = subdomain_cell2point.links(i)
        parent_cell_point = domain_cell2point.links(cell)
        for sub_point, parent_point in zip(sub_cell_point, parent_cell_point):
            if sub2parent_node[sub_point] != -1:
                assert sub2parent_node[sub_point] == parent_point
            else:
                sub2parent_node[sub_point] = parent_point
    return sub2parent_node

def eval_function(u, points):
    if points.ndim == 1:
        points = points.reshape(1, -1)
    domain = u.function_space.mesh
    domain_tree = bb_tree(domain, domain.topology.dim)
    # Find cells whose bounding-box collide with the the points
    cell_candidates = compute_collisions_points(domain_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points)
    cells = []
    points_on_proc = []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)
    return u_values

def compute_error_and_correlation(result, ref):
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

def assign_function(f, idx, val):
    '''idx means the value's order in domain.geometry.x'''
    assert len(val) == len(f.x.array)
    assert len(idx) == len(val)
    functionspace2mesh = fspace2mesh(f.function_space)
    mesh2functionspace = np.argsort(functionspace2mesh)
    f.x.array[mesh2functionspace[idx]] = val

def fspace2mesh(V):
    fspace_cell2point = V.dofmap.list
    subdomain_cell2point = V.mesh.geometry.dofmap

    fspace2mesh = np.ones((len(V.mesh.geometry.x)), dtype=int) * -1
    for cell2point1, cell2point2 in zip(fspace_cell2point, subdomain_cell2point):
        for idx_fspace, idx_submesh in zip(cell2point1, cell2point2):
            if fspace2mesh[idx_fspace] != -1:
                assert fspace2mesh[idx_fspace] == idx_submesh
            else:
                fspace2mesh[idx_fspace] = idx_submesh
    return fspace2mesh

def get_activation_time_from_v(v_data):
    v_deriviative = np.diff(v_data, axis=0)
    # find the time where the v_deriviative is biggest
    activation_time = np.argmax(v_deriviative, axis=0)
    return activation_time

def v_data_argument(v_data, tau = 1, v_rest_healthy = -90, v_ischemia_rest = -60, v_peak_healthy = 10, v_peak_ischemia = -20):
    phi_1, phi_2 = compute_phi_with_v_timebased(v_data, None, v_ischemia_rest, v_peak_ischemia)
    v = []
    for t in range(phi_2.shape[0]):
        G_phi_1 = G_tau(phi_1[t], tau)
        G_phi_2 = G_tau(phi_2[t], tau)
        v_timeframe = (v_rest_healthy * G_phi_2 + v_peak_healthy * (1 - G_phi_2)) * G_phi_1 + \
                        (v_ischemia_rest * G_phi_2 + v_peak_ischemia * (1 - G_phi_2)) * (1 - G_phi_1)
        v.append(v_timeframe)
    v = np.array(v)
    return v