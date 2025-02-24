'''property of transfer matrix A (Au + Rv = 0)'''
import sys

from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Function
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI

sys.path.append('.')
from utils.helper_function import  petsc2array


# mesh of Body
file = "3d/data/mesh_multi_conduct_ecgsim.msh"
domain, cell_markers, facet_markers = gmshio.read_from_msh(file, MPI.COMM_WORLD, gdim=3)
tdim = domain.topology.dim

# function space
V1 = functionspace(domain, ("Lagrange", 1))

# sigma_i : intra-cellular conductivity tensor in Heart
# sigma_e : extra-cellular conductivity tensor in Heart
# sigma_t : conductivity tensor in Torso
# M  : sigma_i + sigma_e in Heart 
#      sigma_t in Torso
# S/m
sigma_t = 0.8
sigma_i = 0.4
sigma_e = 0.8

def rho1(x):
    tensor = np.eye(tdim) * sigma_t
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho2(x):
    tensor = np.eye(tdim) * (sigma_i + sigma_e)
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho3(x):
    tensor = np.eye(tdim) * sigma_t / 5
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
def rho4(x):
    tensor = np.eye(tdim) * sigma_t * 3
    values = np.repeat(tensor, x.shape[1])
    return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])

V = functionspace(domain, ("DG", 0, (tdim, tdim)))
M = Function(V)
M.interpolate(rho1, cell_markers.find(1))
M.interpolate(rho2, cell_markers.find(2))
M.interpolate(rho3, cell_markers.find(3))
M.interpolate(rho4, cell_markers.find(4))

u = TestFunction(V1)
v = TrialFunction(V1)
dx1 = Measure("dx", domain = domain)
a_element = dot(grad(u), dot(M, grad(v))) * dx1
bilinear_form_a = form(a_element)
A = assemble_matrix(bilinear_form_a)
A.assemble()

A = petsc2array(A)

U, s, V = np.linalg.svd(A)
rank = np.sum(s > 1e-10)
print('last 3 sigular vale:', np.sort(s)[0:3])
print('Rank of the matrix:', rank)
print('matrix shape:', A.shape)

p = np.ones((A.shape[0],))
print('A*ones:', np.linalg.norm(A@p))