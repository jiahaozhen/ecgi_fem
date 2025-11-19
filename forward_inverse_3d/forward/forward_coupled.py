import sys

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI
from petsc4py import PETSc
import h5py

sys.path.append('.')
from utils.function_tools import extract_data_from_function

def forward_tmp(mesh_file, v_data, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, multi_flag=True, gdim=3):
    """
    Solves the forward problem for cardiac and torso electrostatic potentials using the finite element method (FEM).
    This function computes the transmembrane potential (TMP) in the body based on the provided ECG simulation data.

    Args:
        mesh_file (str): The path to the mesh file representing the body domain (in .msh format).
        v_data (ndarray): A 2D array of ventricle potential data, where each row corresponds to a time step.
        sigma_i (float, optional): The intracellular conductivity tensor in the heart (in S/m). Default is 0.2.
        sigma_e (float, optional): The extracellular conductivity tensor in the heart (in S/m). Default is 0.4.
        sigma_t (float, optional): The conductivity tensor in the torso (in S/m). Default is 0.2.
        multi_flag (bool, optional): Flag to control different conductivity values for different regions. Default is True.

    Returns:
        list: A list of computed body potentials (`u_data`) at each time step corresponding to the input `v_data`.
    """
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))
    V3 = functionspace(domain, ("DG", 0, (tdim, tdim)))
    u = Function(V1)
    v = Function(V2)

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
    
    M = Function(V3)
    M.interpolate(rho1, cell_markers.find(1))
    M.interpolate(rho2, cell_markers.find(2))
    if cell_markers.find(3).any():
        if multi_flag == True:
            M.interpolate(rho3, cell_markers.find(3))
        else:
            M.interpolate(rho1, cell_markers.find(3))
    if cell_markers.find(4).any():
        if multi_flag == True:
            M.interpolate(rho4, cell_markers.find(4))
        else:
            M.interpolate(rho1, cell_markers.find(4))
    Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_i))

    # A u = b
    # matrix A
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    dx1 = Measure("dx", domain = domain)
    a_element = dot(grad(v1), dot(M, grad(u1))) * dx1
    bilinear_form_a = form(a_element)
    A = assemble_matrix(bilinear_form_a)
    A.assemble()

    # b
    dx2 = Measure("dx", domain=subdomain_ventricle)
    b_element = -dot(grad(v1), dot(Mi, grad(v))) * dx2
    entity_map = {domain._cpp_object: ventricle_to_torso}
    linear_form_b = form(b_element, entity_maps=entity_map)
    b = create_vector(linear_form_b)

    solver = PETSc.KSP().create()
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)

    if v_data.ndim == 1:
        v_data = v_data.reshape(1,-1)
    total_num = len(v_data)
    u_data = []
    for i in range(total_num):
        v.x.array[:] = v_data[i]
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form_b)
        solver.solve(b, u.vector)
        u_data.append(u.x.array.copy())
    return np.array(u_data), V1

def compute_d_from_tmp(mesh_file, v_data, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, multi_flag=True):
    u_f_data, u_functionspace = forward_tmp(mesh_file, v_data, sigma_i=sigma_i, sigma_e=sigma_e, sigma_t=sigma_t, multi_flag=multi_flag)
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    points = np.array(geom['geom_thorax']['pts'])
    d_data = extract_data_from_function(u_f_data, u_functionspace, points)
    return d_data