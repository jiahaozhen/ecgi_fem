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

def fspace2mesh(V: functionspace):
    '''
    return the mapping from function space dof index to mesh vertex index
    fspace2mesh[index in function space] = index in mesh
    '''
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

def assign_function(f: Function, idx: np.ndarray, val: np.ndarray):
    '''idx means the value's order in domain.geometry.x'''
    assert len(val) == len(f.x.array)
    assert len(idx) == len(val)
    functionspace2mesh = fspace2mesh(f.function_space)
    mesh2functionspace = np.argsort(functionspace2mesh)
    f.x.array[mesh2functionspace[idx]] = val