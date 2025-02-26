from ufl import grad
from dolfinx.mesh import locate_entities_boundary, meshtags, exterior_facet_indices, create_submesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.fem import functionspace, Function, Expression
from .normals_and_tangents import facet_vector_approximation
from dolfinx.plot import vtk_mesh
import numpy as np
import pyvista

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

def compute_error(v_exact, phi_result):
    marker_exact = np.full(v_exact.x.array.shape, 0)
    marker_exact[v_exact.x.array > -89.9] = 1
    marker_result = np.full(phi_result.x.array.shape, 0)
    marker_result[phi_result.x.array < 0] = 1

    coordinates = v_exact.function_space.tabulate_dof_coordinates()
    coordinates_ischemic_exact = coordinates[np.where(marker_exact == 1)]
    coordinates_ischemic_result = coordinates[np.where(marker_result == 1)]

    cm1 = np.mean(coordinates_ischemic_exact, axis=0)
    cm2 = np.mean(coordinates_ischemic_result, axis=0)
    cm = np.linalg.norm(cm1-cm2)

    if (coordinates_ischemic_result.size == 0):
        return (cm, None, None, None)
    
    # HaussDist
    hdxy = 0
    for coordinate in coordinates_ischemic_exact:
        hdy = np.min(np.linalg.norm(coordinate - coordinates_ischemic_result, axis=1))
        hdxy = max(hdxy, hdy)
    hdyx = 0
    for coordinate in coordinates_ischemic_result:
        hdx = np.min(np.linalg.norm(coordinate - coordinates_ischemic_exact, axis=1))
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

def get_activation_time_from_v(v_data, threshold):
    activation_time = np.argmax(v_data > threshold, axis=0)
    return activation_time