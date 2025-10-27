'''the dof in TestFunction TrialFunction'''
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import create_submesh
import numpy as np
from ufl import TestFunction, TrialFunction, Measure
from mpi4py import MPI

file = "forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim
# mesh of Heart
subdomain, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

V = functionspace(subdomain, ("Lagrange", 1))
u = TrialFunction(V)
# u = TestFunction(V)
phi = Function(V)

for index in range(len(V.tabulate_dof_coordinates())):
    phi.x.array[index-1] = 0
    phi.x.array[index] = 1

    dx = Measure('dx', domain=subdomain)
    a = phi * u * dx
    form_a = form(a)
    a_vector = assemble_vector(form_a)
    a_vector.assemble()
    a_array = a_vector.array
    index1 = np.where(a_array != 0)[0]

    functionspace_cell = V.dofmap.list
    index2 = []
    for row in functionspace_cell:
        if index in row:
            for e in row:
                if e not in index2:
                    index2.append(e)
    index2 = np.sort(np.array(index2))

    #should be no error
    if not np.array_equal(index1, index2):
        print('error in row', index)
        print('index1:', index1)
        print('index2:', index2)