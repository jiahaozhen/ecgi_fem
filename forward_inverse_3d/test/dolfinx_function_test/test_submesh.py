'''
函数空间点序列与网格点序列不一致的问题 
子网格点序列与原网格点序列不一致的问题
'''
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh, locate_entities_boundary
import numpy as np
from mpi4py import MPI
from utils.helper_function import submesh_node_index
from utils.function_tools import assign_function, fspace2mesh, eval_function

file = r"forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

domain_boundary_index = locate_entities_boundary(domain, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
subdomain_boundary_index = locate_entities_boundary(subdomain, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))

domain_pts = domain.geometry.x
subdomain_pts = subdomain.geometry.x

'''test submesh_node_index function'''
submesh2mesh = []
for row in subdomain_pts:
    matches = np.where((domain_pts == row).all(axis=1))[0]
    if len(matches) > 0:
        submesh2mesh.append(matches[0])
    else:
        submesh2mesh.append(-1)
# print('index of submesh''s point in mesh:', submesh2mesh)
submesh2mesh_1 = submesh_node_index(domain, cell_markers, 2)
# print('another way based on cell_marker:', submesh2mesh_1)
print('error in two ways on get index of submesh''s point in main mesh:', np.linalg.norm(submesh2mesh-submesh2mesh_1))

'''assign val in submesh function'''
V2 = functionspace(subdomain, ("Lagrange", 1))
f2 = Function(V2)
val = np.arange(len(subdomain.geometry.x))
assign_function(f2, val, val)

functionspace2mesh = fspace2mesh(V2)
# print('fspace to mesh (point index):', functionspace2mesh)
functionspace_pts_in_order = V2.tabulate_dof_coordinates()
print('error in index match function:', np.linalg.norm(functionspace_pts_in_order - subdomain_pts[functionspace2mesh]))

f2_val1 = eval_function(f2, subdomain.geometry.x).squeeze()
print('error in assignment method 1:', np.linalg.norm(f2_val1 - val))
f2.x.array[:] = functionspace2mesh
f2_val2 = eval_function(f2, subdomain.geometry.x).squeeze()
print('error in assignment method 2:', np.linalg.norm(f2_val2-val))

# domain's cell2point: domain.geometry.dofmap
# functionspace's cell2point: functionspace.dofmap.list