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

def get_ring_pts(mesh_file: str, gdim: int) -> tuple[np.ndarray, np.ndarray]:
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
        mitral_ring = group1
        tricuspid_ring = group2
    else:
        mitral_ring = group2
        tricuspid_ring = group1

    return mitral_ring, tricuspid_ring

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
        return 17  # segment 17 (apex cap)

    theta_deg = (theta_deg + 360) % 360
    seg = int(theta_deg / (360 / n_seg)) + 1 + offset
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