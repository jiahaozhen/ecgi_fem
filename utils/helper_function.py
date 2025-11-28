from ufl import grad
from dolfinx.mesh import locate_entities_boundary, meshtags, exterior_facet_indices, create_submesh, Mesh
from dolfinx.fem import functionspace, Function, Expression
from dolfinx.io import gmshio
from .normals_and_tangents import facet_vector_approximation
from mpi4py import MPI
import numpy as np
from utils.function_tools import eval_function


def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s


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
    # the idx of submesh node in parent mesh
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
    from utils.function_tools import fspace2mesh
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