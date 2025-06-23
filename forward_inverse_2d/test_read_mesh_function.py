from dolfinx import default_scalar_type, io
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import create_submesh, locate_entities
from dolfinx.plot import vtk_mesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from helper_function import delta_tau
from mpi4py import MPI
from petsc4py import PETSc
import pyvista

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh("heart_torso.msh", MPI.COMM_WORLD, gdim=2)
tdim = domain.topology.dim
# mesh of Heart
subdomain, sub_to_parent, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
V1 = functionspace(domain, ("Lagrange", 1))
u = Function(V1)
xdmf1 = io.XDMFFile(domain.comm, "body_surface_potential_oneframe.xdmf", "r")
xdmf2 = io.XDMFFile(subdomain.comm, "phi_oneframe.xdmf", "r")

xdmf1.read_function(u)

xdmf1.close()
xdmf2.close()