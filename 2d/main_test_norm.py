from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function, Expression
from dolfinx.mesh import create_submesh, locate_entities_boundary, meshtags, exterior_facet_indices
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import numpy as np
from ufl import grad
from mpi4py import MPI
from normals_and_tangents import facet_vector_approximation

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("2d/heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

num_node = domain.topology.index_map(tdim-2).size_local
num_cell = domain.topology.index_map(tdim).size_local

V1 = functionspace(domain, ("Lagrange", 1))
V2 = functionspace(subdomain, ("Lagrange", 1))
u = Function(V1)
v = Function(V2)

u.x.array[:] = np.load(file='2d/u.npy')
v.x.array[:] = np.load(file='2d/v.npy')

body_boundary_index = locate_entities_boundary(domain, tdim - 2, lambda x: np.full(x.shape[1], True, dtype=bool))
heart_boundary_index = locate_entities_boundary(subdomain, tdim - 2, lambda x: np.full(x.shape[1], True, dtype=bool))
body_boundary_points = np.array(domain.geometry.x[body_boundary_index])
heart_boundary_points = np.array(subdomain.geometry.x[heart_boundary_index])

grad_u = Expression(grad(u), V1.element.interpolation_points())
V3 = functionspace(domain, ("Lagrange", 1, (2, )))
grad_u_f = Function(V3)
grad_u_f.interpolate(grad_u)

grad_v = Expression(grad(v), V2.element.interpolation_points())
V4 = functionspace(subdomain, ("Lagrange", 1, (2, )))
grad_v_f = Function(V4)
grad_v_f.interpolate(grad_v)

domain.topology.create_connectivity(tdim-1, tdim)
subdomain.topology.create_connectivity(tdim-1, tdim)
body_boundary_facets = exterior_facet_indices(domain.topology)
heart_boundary_facets = exterior_facet_indices(subdomain.topology)
bb_tag = meshtags(domain, tdim - 1, body_boundary_facets, 1)
hb_tag = meshtags(subdomain, tdim - 1, heart_boundary_facets, 1)
nt_f = facet_vector_approximation(V3, bb_tag, 1)
nh_f = facet_vector_approximation(V4, hb_tag, 1)

domain_tree = bb_tree(domain, domain.topology.dim)
# Find cells whose bounding-box collide with the the points
cell_candidates = compute_collisions_points(domain_tree, body_boundary_points)
# Choose one of the cells that contains the point
colliding_cells = compute_colliding_cells(domain, cell_candidates, body_boundary_points)
cells = []
points_on_proc = []
for i, point in enumerate(body_boundary_points):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)
grad_u_f_values_exterior = grad_u_f.eval(points_on_proc, cells)
nt_f_values = nt_f.eval(points_on_proc, cells)

cell_candidates = compute_collisions_points(domain_tree, heart_boundary_points)
colliding_cells = compute_colliding_cells(domain, cell_candidates, heart_boundary_points)
cells = []
points_on_proc = []
for i, point in enumerate(heart_boundary_points):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)
grad_u_f_values_interior = grad_u_f.eval(points_on_proc, cells)

subdomain_tree = bb_tree(subdomain, subdomain.topology.dim)
cell_candidates = compute_collisions_points(subdomain_tree, heart_boundary_points)
colliding_cells = compute_colliding_cells(subdomain, cell_candidates, heart_boundary_points)
cells = []
points_on_proc = []
for i, point in enumerate(heart_boundary_points):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)
grad_v_f_values = grad_v_f.eval(points_on_proc, cells)
nh_f_values = nh_f.eval(points_on_proc, cells)

# sigma_i * grad(u_e) * n_h = - sigma_i * grad(v) * n_h
bc1 = []
# sigma_t * grad(u_t) * n_h = sigma_e * grad(u_e) * n_h
bc2 = []
# sigma_t * grad(u_t) * n_t = 0
bc3 = []
for i, point in enumerate(heart_boundary_points):
    bc1.append(np.dot(grad_u_f_values_interior[i] + grad_v_f_values[i], nh_f_values[i]))
    bc2.append(np.dot(grad_u_f_values_interior[i], nh_f_values[i]))
for i, point in enumerate(body_boundary_points):
    bc3.append(np.dot(grad_u_f_values_exterior[i], nt_f_values[i]))
# print('grad(u_B) on B:', grad_u_f_values_exterior)
# print('grad(u_B) on T:', grad_u_f_values_interior)
# print('grad(v)', grad_v_f_values)
# print('nT:', nt_f_values)
# print('nh:', nh_f_values)
print(np.linalg.norm(bc1))
print(np.linalg.norm(bc2))
print(np.linalg.norm(bc3))