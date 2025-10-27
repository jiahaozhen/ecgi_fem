from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.fem import Function, functionspace
import numpy as np

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

def extract_data_from_function(f_data: np.ndarray, functionspace: functionspace, coords: np.ndarray):
    f = Function(functionspace)
    if f_data.ndim == 1:
        f_data = f_data.reshape(1,-1)
    data = []
    for i in range(len(f_data)):
        f.x.array[:] = f_data[i]
        data.append(eval_function(f, coords).squeeze())
    return np.array(data)