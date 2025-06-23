from mpi4py import MPI
from dolfinx.fem import form, functionspace
from dolfinx.mesh import create_unit_square, CellType, locate_entities_boundary
from ufl import TestFunction, Measure
import pyvista
import numpy as np
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import assemble_vector
domain = create_unit_square(MPI.COMM_WORLD, 8, 8, CellType.quadrilateral)
tdim = domain.topology.dim
num_node = domain.topology.index_map(tdim-2).size_local

def OuterBoundary(x):
    return x[0] > 0

facets = locate_entities_boundary(domain, domain.topology.dim-1, OuterBoundary)
# print(outerBoundary)

V = functionspace(domain, ("Lagrange", 1))

u = TestFunction(V)
from dolfinx.mesh import meshtags
facet_indices = np.array(facets, dtype=np.int32)
print(facets)
print(facet_indices)
facet_values = np.full(len(facet_indices), 1, dtype=np.int32)
facet_tags = meshtags(domain, domain.topology.dim - 1, facet_indices, facet_values)

# 定义部分线积分 Measure
ds_left = Measure('ds', domain=domain, subdomain_data=facet_tags, subdomain_id=1)
b_w = u*ds_left
# entity_map = {domain._cpp_object: outerBoundary}
# linear_form_b_w = form(b_w, entity_maps=entity_map)
linear_form_b_w = form(b_w)
B_w = assemble_vector(linear_form_b_w)

marker = np.zeros([num_node,1])
marker = B_w.array
plotter = pyvista.Plotter(window_size=[800, 800])
grid = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
grid.point_data["marker"] = marker
grid.set_active_scalars("marker")
plotter.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = plotter.screenshot("u.png")
plotter.show()