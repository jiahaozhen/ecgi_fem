from dolfinx.mesh import locate_entities_boundary, create_submesh
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np

from .helper_function import submesh_node_index, get_boundary_vertex_connectivity

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

def distinguish_left_right_endo_epi(mesh_file: str, gdim: int) -> np.ndarray:
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    subdomain_left_cavity, _, _, _ = create_submesh(domain, tdim, cell_markers.find(5))
    subdomain_right_cavity, _, _, _ = create_submesh(domain, tdim, cell_markers.find(6))

    # 1 epi 0 mid -1 left_endo -2 right_endo
    marker = np.zeros(subdomain_ventricle.geometry.x.shape[0], dtype=np.int32)

    ventricle_sub2parent = submesh_node_index(domain, cell_markers, 2)
    ventricle_parent2sub = np.zeros(domain.geometry.x.shape[0], dtype=np.int32) - 1
    ventricle_parent2sub[ventricle_sub2parent] = np.arange(len(ventricle_sub2parent))

    left_cavity_sub2parent = submesh_node_index(domain, cell_markers, 5)
    right_cavity_sub2parent = submesh_node_index(domain, cell_markers, 6)
    ventricle_boundary = locate_entities_boundary(subdomain_ventricle, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
    left_cavity_boundary = locate_entities_boundary(subdomain_left_cavity, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
    right_cavity_boundary = locate_entities_boundary(subdomain_right_cavity, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

    marker[ventricle_boundary] = 1
    for i in range(len(left_cavity_boundary)):
        node_index_in_ventricle = ventricle_parent2sub[left_cavity_sub2parent[left_cavity_boundary[i]]]
        if node_index_in_ventricle != -1:
            marker[node_index_in_ventricle] = -1
    for i in range(len(right_cavity_boundary)):
        node_index_in_ventricle = ventricle_parent2sub[right_cavity_sub2parent[right_cavity_boundary[i]]]
        if node_index_in_ventricle != -1:
            marker[node_index_in_ventricle] = -2
    
    return marker.astype(np.int32)

def get_ring_pts(mesh_file: str, gdim: int) -> tuple[np.ndarray, np.ndarray]:
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    points = subdomain_ventricle.geometry.x

    marker = distinguish_left_right_endo_epi(mesh_file, gdim=gdim)
    boundary_vertices, adjacency = get_boundary_vertex_connectivity(subdomain_ventricle)

    left_point_index = []
    ring_point_index = []

    for v in boundary_vertices:
        adjacency_list = adjacency[v]
        marker_val = marker[v]
        if marker_val != 1:
            continue
        for v_n in adjacency_list:
            if marker[v_n] == -1:
                left_point_index.append(v_n)
            if marker[v_n] == -2:
                ring_point_index.append(v_n)

    left_points = points[left_point_index]
    ring_points = points[ring_point_index]

    return left_point_index, ring_point_index, left_points, ring_points

def separate_lv_rv(mesh_file, gdim=3):
    from scipy.spatial import cKDTree as KDTree

    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    marker = distinguish_left_right_endo_epi(mesh_file, gdim=gdim)

    coords = subdomain_ventricle.geometry.x
    left_endo = coords[marker == -1]
    right_endo = coords[marker == -2]

    # compute min distances to left and right landmark sets (use KDTree if available)
    # KDTree is available (imported above)

    if left_endo.size == 0 and right_endo.size == 0:
        return np.empty((0, coords.shape[1])), np.empty((0, coords.shape[1])), np.empty((0,), dtype=bool), np.empty((0,), dtype=bool)

    if left_endo.size == 0:
        return np.empty((0, coords.shape[1])), coords.copy(), np.empty((0,), dtype=bool), np.full((coords.shape[0],), True, dtype=bool)
    if right_endo.size == 0:
        return coords.copy(), np.empty((0, coords.shape[1])), np.full((coords.shape[0],), True, dtype=bool), np.empty((0,), dtype=bool)

    if KDTree is not None:
        tree_l = KDTree(left_endo)
        tree_r = KDTree(right_endo)
        d_left, _ = tree_l.query(coords)
        d_right, _ = tree_r.query(coords)
    else:
        # fallback: chunked pairwise computation to avoid huge memory usage
        def min_dists(points, targets):
            n = points.shape[0]
            out = np.empty(n)
            chunk = 10000
            for i in range(0, n, chunk):
                p = points[i:i+chunk]
                diff = p[:, None, :] - targets[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                out[i:i+chunk] = np.sqrt(d2.min(axis=1))
            return out
        d_left = min_dists(coords, left_endo)
        d_right = min_dists(coords, right_endo)

    # assign to side with smaller distance
    left_mask = d_left < d_right
    lv_points = coords[left_mask]
    rv_points = coords[~left_mask]

    return lv_points, rv_points, left_mask, ~left_mask

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

def compute_lv_axis(annulus_points, apex_point):
    annulus_center = annulus_points.mean(axis=0)
    axis = apex_point - annulus_center
    axis /= np.linalg.norm(axis)
    return annulus_center, axis

def compute_lv_height_values(points, annulus_center, axis):
    return np.dot(points - annulus_center, axis)

def assign_segment(hi, theta_deg):
    if hi < 0.33:
        ring = 'basal'
        n_seg, offset = 6, 0
    elif hi < 0.66:
        ring = 'mid'
        n_seg, offset = 6, 6
    elif hi < 0.9:
        ring = 'apical'
        n_seg, offset = 4, 12
    else:
        return 16  # segment 17 (apex cap)

    theta_deg = (theta_deg + 360) % 360
    seg = int(theta_deg / (360 / n_seg)) + offset
    return seg

def lv_17_segmentation(lv_points, annulus_points, apex_point):
    annulus_center, axis = compute_lv_axis(annulus_points, apex_point)

    v1 = annulus_points[0] - annulus_center
    v1 -= np.dot(v1, axis) * axis
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)

    h_values = compute_lv_height_values(lv_points, annulus_center, axis)
    h_min, h_max = h_values.min(), h_values.max()
    h_norm = (h_values - h_min) / (h_max - h_min)

    seg_ids = []
    r_mapped = []
    theta_mapped = []
    for p, h in zip(lv_points, h_norm):
        v = p - annulus_center
        v /= np.linalg.norm(v)
        v -= np.dot(v, axis) * axis
        theta = np.degrees(np.arctan2(np.dot(v, v2), np.dot(v, v1)))
        r_mapped.append(np.linalg.norm(v))
        theta_mapped.append(theta)
        seg_ids.append(assign_segment(h, theta))
    return np.array(seg_ids), np.array(r_mapped), np.array(theta_mapped)

def lv_17_segmentation_from_mesh(mesh_file: str, gdim: int = 3) -> np.ndarray:
    """
    Perform 17-segment segmentation of the left ventricle based on the mesh file.
    
    Parameters:
    - mesh_file: Path to the mesh file.
    - gdim: Geometric dimension of the mesh (default is 3).
    
    Returns:
    - segment_ids: Array of segment IDs for each vertex in the left ventricle.
    """
    # Load mesh and extract ventricle submesh
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    
    points = subdomain_ventricle.geometry.x
    left_ring_index, ring_ring_index, left_ring_pts, right_ring_pts = get_ring_pts(mesh_file, gdim=gdim)
    
    lv_points, rv_points, lv_mask, rv_mask = separate_lv_rv(mesh_file, gdim=gdim)
    
    apex_point = get_apex_from_annulus_pts(lv_points, left_ring_pts)
    
    segment_ids_lv, r_mapped, theta_mapped = lv_17_segmentation(lv_points, left_ring_pts, apex_point)
    
    segment_ids = np.zeros(points.shape[0], dtype=np.int32)
    segment_ids[lv_mask] = segment_ids_lv
    segment_ids[rv_mask] = -1  # RV points marked as -1
    
    return segment_ids, r_mapped, theta_mapped