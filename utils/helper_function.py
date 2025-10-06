from ufl import grad
from dolfinx.mesh import locate_entities_boundary, meshtags, exterior_facet_indices, create_submesh, Mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.fem import functionspace, Function, Expression
from dolfinx.io import gmshio
from .normals_and_tangents import facet_vector_approximation
from mpi4py import MPI
import numpy as np
import h5py
import scipy.interpolate

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

def compute_phi_with_v_timebased(v, function_space, marker_ischemia):
    phi_1 = np.full_like(v, 0)
    phi_2 = np.full_like(v, 0)
    activation_time = get_activation_time_from_v(v)
    marker_ischemia = np.where(marker_ischemia==1, True, False)

    coordinates = function_space.tabulate_dof_coordinates()
    marker_activation  = np.full_like(v, False, dtype=bool)
    for i in range(v.shape[1]):
        marker_activation[activation_time[i]:, i] = True

    def min_distance(coords, mask):
        if np.any(mask):
            return np.min(np.linalg.norm(coords[:, None, :] - coords[mask], axis=2), axis=1)
        else:
            return np.zeros(len(coords))
    
    min_iso = min_distance(coordinates, marker_ischemia)
    min_no_iso = min_distance(coordinates, ~marker_ischemia)
    for timeframe in range(v.shape[0]):
        
        min_act = min_distance(coordinates, marker_activation[timeframe])
        min_no_act = min_distance(coordinates, ~marker_activation[timeframe])
    
        phi_1[timeframe] = np.where(marker_ischemia, -min_no_iso, min_iso)
        phi_2[timeframe] = np.where(marker_activation[timeframe], -min_no_act, min_act)
        if (phi_1[timeframe] == 0).all():
            phi_1[timeframe] = 20
        if (phi_2[timeframe] == 0).all():
            if timeframe < np.min(activation_time) + 5:
                phi_2[timeframe] = 20
            else:
                phi_2[timeframe] = -20
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

def eval_function(u: Function, points: np.ndarray):
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

def assign_function(f: Function, idx: np.ndarray, val: np.ndarray):
    '''idx means the value's order in domain.geometry.x'''
    assert len(val) == len(f.x.array)
    assert len(idx) == len(val)
    functionspace2mesh = fspace2mesh(f.function_space)
    mesh2functionspace = np.argsort(functionspace2mesh)
    f.x.array[mesh2functionspace[idx]] = val

def fspace2mesh(V: functionspace):
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

def get_activation_time_from_v(v_data: np.ndarray):
    v_deriviative = np.diff(v_data, axis=0)
    # find the time where the v_deriviative is biggest
    activation_time = np.argmax(v_deriviative, axis=0)
    return activation_time

def v_data_argument(phi_1: np.ndarray, phi_2: np.ndarray, tau = 10, a1 = -90, a2 = -60, a3 = 10, a4 = -20):
    G_phi_1 = G_tau(phi_1, tau)
    G_phi_2 = G_tau(phi_2, tau)
    v = ((a1 * G_phi_2 + a3 * (1 - G_phi_2)) * G_phi_1 + 
         (a2 * G_phi_2 + a4 * (1 - G_phi_2)) * (1 - G_phi_1))
    return v

def compute_error_phi(phi_exact: np.ndarray, phi_result: np.ndarray, function_space: functionspace):
    marker_exact = np.where(phi_exact < 0, 1, 0)
    marker_result = np.where(phi_result < 0, 1, 0)
    coordinates = function_space.tabulate_dof_coordinates()
    coordinates_ischemia_exact = coordinates[np.where(marker_exact == 1)]
    coordinates_ischemia_result = coordinates[np.where(marker_result == 1)]
    cm1 = np.mean(coordinates_ischemia_exact, axis=0)
    cm2 = np.mean(coordinates_ischemia_result, axis=0)
    cm = np.linalg.norm(cm1-cm2)
    return cm

def find_connected_vertex(domain: Mesh, vertex: int):
    
    domain.topology.create_connectivity(0, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, 0)
    
    incident_cells = domain.topology.connectivity(0, domain.topology.dim).links(vertex)
    
    connected_vertices = set()
    for cell in incident_cells:
        cell_vertices = domain.topology.connectivity(domain.topology.dim, 0).links(cell)
        connected_vertices.update(cell_vertices)
    
    connected_vertices.discard(vertex)
    connected_vertices = sorted(list(connected_vertices))
    
    return connected_vertices

def find_all_connected_vertex(domain: Mesh):
    tdim = domain.topology.dim
    domain.topology.create_connectivity(0, tdim)
    domain.topology.create_connectivity(tdim, 0)
    
    conn_v_to_c = domain.topology.connectivity(0, tdim)
    conn_c_to_v = domain.topology.connectivity(tdim, 0)
    
    num_vertices = domain.topology.index_map(0).size_local
    all_neighbors = []

    for v in range(num_vertices):
        incident_cells = conn_v_to_c.links(v)
        neighbors = set()
        for cell in incident_cells:
            cell_vertices = conn_c_to_v.links(cell)
            neighbors.update(cell_vertices)
        neighbors.discard(v)
        all_neighbors.append(sorted(neighbors))

    return all_neighbors

def get_boundary_vertex_connectivity(domain: Mesh):
    """
    获取 dolfinx 网格外表面上的顶点连接关系（邻接表）。
    返回:
        boundary_vertices: list[int] 所有外表面顶点索引
        adjacency: dict[int, set[int]] 顶点到其邻接顶点集合
    """
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim - 1)  # cell→facet
    domain.topology.create_connectivity(tdim - 1, tdim)  # facet→cell
    domain.topology.create_connectivity(tdim - 1, 0)     # facet→vertex

    facet_to_cell = domain.topology.connectivity(tdim - 1, tdim)
    facet_to_vertex = domain.topology.connectivity(tdim - 1, 0)

    num_facets = domain.topology.index_map(tdim - 1).size_local

    boundary_facets = [
        f for f in range(num_facets)
        if len(facet_to_cell.links(f)) == 1
    ]

    adjacency = {}
    for f in boundary_facets:
        vs = facet_to_vertex.links(f)
        for i in range(len(vs)):
            vi = vs[i]
            adjacency.setdefault(vi, set())
            for j in range(len(vs)):
                if i == j:
                    continue
                vj = vs[j]
                adjacency[vi].add(vj)

    boundary_vertices = sorted(adjacency.keys())

    return boundary_vertices, adjacency

def find_vertex_with_neighbour_less_than_0(domain: Mesh, f: Function):
    index_function = np.where(f.x.array < 0)[0]
    f2mesh = fspace2mesh(f.function_space)
    mesh2f = np.argsort(f2mesh)
    index_mesh = f2mesh[index_function]

    neighbor_map = {}
    neighbor_idx = set()

    for i in index_mesh:
        neighbors = find_connected_vertex(domain, i)
        neighbor_idx.update(neighbors)
        for j in neighbors:
            neighbor_map[j] = neighbor_map.get(j, 0) + 1

    neighbor_idx = mesh2f[np.array(list(neighbor_idx), dtype=int)]
    neighbor_map = {mesh2f[k]: v for k, v in neighbor_map.items()}

    return neighbor_idx, neighbor_map

def find_vertex_with_coordinate(domain: Mesh, x: np.ndarray):
    points = domain.geometry.x
    distances = np.linalg.norm(points - x, axis=1)
    closest_vertex = np.argmin(distances)
    return closest_vertex
    
def transfer_bsp_to_standard12lead(bsp_data: np.ndarray, lead_index: np.ndarray):
    standard12Lead = np.zeros((bsp_data.shape[0], 12))
    # I = VL - VR
    standard12Lead[:,0] = bsp_data[:,lead_index[7]] - bsp_data[:,lead_index[6]]
    # II = VF - VR
    standard12Lead[:,1] = bsp_data[:,lead_index[8]] - bsp_data[:,lead_index[6]]
    # III = VF - VL
    standard12Lead[:,2] = bsp_data[:,lead_index[8]] - bsp_data[:,lead_index[7]]
    # Vi = Vi - (VR + VL + VF) / 3
    standard12Lead[:, 3:9] = bsp_data[:, lead_index[0:6]] - np.mean(bsp_data[:, lead_index[6:9]], axis=1, keepdims=True)
    # aVR = VR - (VL + VF) / 2
    standard12Lead[:, 9] = bsp_data[:, lead_index[6]] - np.mean(bsp_data[:, lead_index[7:9]], axis=1)
    # aVL = VL - (VR + VF) / 2
    standard12Lead[:, 10] = bsp_data[:, lead_index[7]] - np.mean(bsp_data[:, [lead_index[6], lead_index[8]]], axis=1)
    # aVF = VF - (VR + VL) / 2
    standard12Lead[:, 11] = bsp_data[:, lead_index[8]] - np.mean(bsp_data[:, lead_index[6:8]], axis=1)
    
    return standard12Lead

def add_noise_based_on_snr(data: np.ndarray, snr: float) -> np.ndarray:
    """
    Add noise to the data based on the specified SNR (Signal-to-Noise Ratio).
    
    Parameters:
    - data: The original data to which noise will be added.
    - snr: The desired SNR in decibels (dB).
    
    Returns:
    - Noisy data with the specified SNR.
    """
    signal_power = np.mean(data**2)
    noise_power = signal_power / (10**(snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    noisy_data = data + noise
    return noisy_data

def check_noise_level_snr(data: np.ndarray, noise: np.ndarray) -> float:
    """
    Check the SNR (Signal-to-Noise Ratio) of the data.
    
    Parameters:
    - data: The original data.
    - noise: The noise added to the data.
    
    Returns:
    - SNR in decibels (dB).
    """
    signal_power = np.mean(data**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_phi_with_activation(activation_f : Function, duration : int):
    phi = np.zeros((duration, len(activation_f.x.array)))
    activation_time = activation_f.x.array
    marker_activation = np.full_like(phi, False, dtype=bool)
    for i in range(phi.shape[1]):
        marker_activation[int(activation_time[i]):, i] = True
    coordinates = activation_f.function_space.tabulate_dof_coordinates()
    
    def min_distance(coords, mask):
        if np.any(mask):
            return np.min(np.linalg.norm(coords[:, None, :] - coords[mask], axis=2), axis=1)
        else:
            return np.zeros(len(coords))
    
    for timeframe in range(duration):
        
        min_act = min_distance(coordinates, marker_activation[timeframe])
        min_no_act = min_distance(coordinates, ~marker_activation[timeframe])
    
        phi[timeframe] = np.where(marker_activation[timeframe], -min_no_act, min_act)
        if (phi[timeframe] == 0).all():
            if timeframe < np.min(activation_time) + 5:
                phi[timeframe] = 20
            else:
                phi[timeframe] = -20
    
    return phi

def extract_data_from_mesh(mesh_file: str, points: np.ndarray, origin_data: np.ndarray) -> np.ndarray:
    domain, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
    V = functionspace(domain, ("Lagrange", 1))
    f = Function(V)
    extracted_data = []
    total_num = len(origin_data)
    for i in range(total_num):
        f.x.array[:] = origin_data[i]
        extracted_data_t = eval_function(f, points).squeeze()
        extracted_data.append(extracted_data_t.copy())
    return np.array(extracted_data)

def extract_data_from_mesh1_to_mesh2(mesh_file1: str, mesh_file2: str, origin_data: np.ndarray) -> np.ndarray:
    domain2, _, _ = gmshio.read_from_msh(mesh_file2, MPI.COMM_WORLD, gdim=3)
    V2 = functionspace(domain2, ("Lagrange", 1))
    points = V2.tabulate_dof_coordinates()
    extract_data = extract_data_from_mesh(mesh_file1, points, origin_data)
    return extract_data

def extract_data_from_submesh1_to_submesh2(mesh_file1: str, mesh_file2: str, origin_data: np.ndarray) -> np.ndarray:
    domain1, cell_markers_1, _ = gmshio.read_from_msh(mesh_file1, MPI.COMM_WORLD, gdim=3)
    domain2, cell_markers_2, _ = gmshio.read_from_msh(mesh_file2, MPI.COMM_WORLD, gdim=3)
    
    subdomain1, _, _, _ = create_submesh(domain1, domain1.topology.dim, cell_markers_1.find(2))
    subdomain2, _, _, _ = create_submesh(domain2, domain2.topology.dim, cell_markers_2.find(2))

    V1 = functionspace(subdomain1, ("Lagrange", 1))
    V2 = functionspace(subdomain2, ("Lagrange", 1))

    points = V2.tabulate_dof_coordinates()
    f = Function(V1)
    extracted_data = []
    total_num = len(origin_data)
    for i in range(total_num):
        f.x.array[:] = origin_data[i]
        extracted_data_t = eval_function(f, points).squeeze()
        extracted_data.append(extracted_data_t.copy())    
    return extracted_data

def distinguish_epi_endo(mesh_file: str, gdim: int) -> np.ndarray:
    """
    Distinguish epi and endo based on the mesh file.
    
    Parameters:
    - mesh_file: Path to the mesh file.
    
    Returns:
    - epi_endo_marker: Array with 1 for epi and -1 for endo.
    """
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    subdomain_cavity, _, _, _ = create_submesh(domain, tdim, cell_markers.find(4))

    epi_endo_marker = np.zeros(subdomain_ventricle.geometry.x.shape[0], dtype=np.int32)

    ventricle_sub2parent = submesh_node_index(domain, cell_markers, 2)
    ventricle_parent2sub = np.zeros(domain.geometry.x.shape[0], dtype=np.int32) - 1
    ventricle_parent2sub[ventricle_sub2parent] = np.arange(len(ventricle_sub2parent))
    cavity_sub2parent = submesh_node_index(domain, cell_markers, 4)
    ventricle_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
    cavity_boundary = locate_entities_boundary(subdomain_cavity, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

    epi_endo_marker[ventricle_boundary] = 1
    for i in range(len(cavity_boundary)):
        node_index_in_ventricle = ventricle_parent2sub[cavity_sub2parent[cavity_boundary[i]]]
        if node_index_in_ventricle != -1:
            epi_endo_marker[node_index_in_ventricle] = -1
    
    return epi_endo_marker.astype(np.int32)

def get_ring_pts(mesh_file: str, gdim: int) -> np.ndarray:
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    points = subdomain_ventricle.geometry.x

    epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)
    boundary_vertices, adjacency = get_boundary_vertex_connectivity(subdomain_ventricle)

    ring_point_index = []

    for v in boundary_vertices:
        adjacency_list = adjacency[v]
        marker = epi_endo_marker[v]
        if marker != 1:
            continue
        for v_n in adjacency_list:
            if epi_endo_marker[v_n] == -marker:
                ring_point_index.append(v_n)

    ring_points = points[ring_point_index]

    return ring_point_index, ring_points

def distinguish_ring_pts(ring_points: np.ndarray):
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=10, min_samples=1).fit(ring_points)  # eps 可调
    labels = clustering.labels_
    group1 = ring_points[labels == 0]
    group2 = ring_points[labels == 1]

    x_mean1 = group1[:, 0].mean()
    x_mean2 = group2[:, 0].mean()

    if x_mean1 < x_mean2:
        left_ventricle = group1
        right_ventricle = group2
    else:
        left_ventricle = group2
        right_ventricle = group1

    return left_ventricle, right_ventricle

def separate_lv_rv(points, mitral_ring, tricuspid_ring, offset_ratio=0.25):
    # 1. 环中心
    cL = mitral_ring.mean(axis=0)
    cR = tricuspid_ring.mean(axis=0)
    
    # 2. 连线方向单位向量
    v = cR - cL
    dist = np.linalg.norm(v)
    v /= dist
    
    # 3. 分界平面中心（向右偏移）
    cM = (cL + cR) / 2
    cM_shift = cM + offset_ratio * dist * v  # 向右心室方向平移
    
    # 4. 点投影与分类
    proj = np.dot(points - cM_shift, v)
    lv_mask = proj < 0
    rv_mask = ~lv_mask

    return points[lv_mask], points[rv_mask], lv_mask, rv_mask

def get_apex_from_annulus_pts(vertices, annulus_pts):
    V = np.asarray(vertices)
    annulus = np.asarray(annulus_pts)
    ann_centroid = annulus.mean(axis=0)
    U, S, VT = np.linalg.svd(annulus - ann_centroid, full_matrices=False)
    normal = VT[-1] / np.linalg.norm(VT[-1])
    centroid = V.mean(axis=0)
    if np.dot(normal, centroid - ann_centroid) < 0:
        normal = -normal
    proj = np.dot(V - ann_centroid, normal)
    idx = np.argmax(proj)
    apex = V[idx]

    return apex