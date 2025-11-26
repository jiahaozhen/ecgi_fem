'''
由于积分区域的不同， 相同基函数积分结果不同， 致矩阵A的非零模式不同
'''
from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import create_submesh
import numpy as np
from ufl import TestFunction, TrialFunction, dot, grad, Measure
from mpi4py import MPI

import sys
sys.path.append('.')
from utils.helper_function import petsc2array, fspace2mesh

def transfer_index(pts_1, pts_2):
    index = []
    for row in pts_1:
        # 计算当前 row 到 pts_2 所有点的欧式距离
        dist = np.linalg.norm(pts_2 - row, axis=1)
        # 找到距离最近的点索引
        idx = np.argmin(dist)
        index.append(idx)
    return index

def compare_A_matrix(mesh_file, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, gdim=3):
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))
    V3 = functionspace(domain, ("DG", 0, (tdim, tdim)))

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
    M.interpolate(rho3, cell_markers.find(3))
    M.interpolate(rho4, cell_markers.find(4))

    # matrix A
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    dx1 = Measure("dx", domain = domain)
    dx2 = Measure("dx", domain=subdomain_ventricle)
    a_element_1 = dot(grad(v1), dot(M, grad(u1))) * dx1
    bilinear_form_a_1 = form(a_element_1)
    # entity_map = {domain._cpp_object: ventricle_to_torso}
    # bilinear_form_a_1 = form(a_element_1, entity_maps=entity_map)
    A_1 = assemble_matrix(bilinear_form_a_1)
    A_1.assemble()

    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    Me = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_e))
    Mi = Constant(subdomain_ventricle, default_scalar_type(np.eye(tdim) * sigma_i))
    
    a_element_2 = dot(grad(v2), dot(Mi+Me, grad(u2))) * dx2
    bilinear_form_a_2 = form(a_element_2)
    A_2 = assemble_matrix(bilinear_form_a_2)
    A_2.assemble()

    index = transfer_index(V2.tabulate_dof_coordinates(), V1.tabulate_dof_coordinates())

    A_1_array = petsc2array(A_1)
    A_2_array = petsc2array(A_2)
    A_1_array = A_1_array[np.ix_(index, index)]

    subdomain_ventricle.topology.create_connectivity(tdim, 0)
    connectivity = subdomain_ventricle.topology.connectivity(tdim, 0)

    functionspace2mesh = fspace2mesh(V2)
    mesh2functionspace = np.argsort(functionspace2mesh)
    num_vertices = subdomain_ventricle.topology.index_map(0).size_local
    A_3 = np.zeros((num_vertices, num_vertices), dtype=int)

    for cell in range(connectivity.num_nodes):
        cell_vertices = connectivity.links(cell)
        for i in cell_vertices:
            for j in cell_vertices:
                A_3[mesh2functionspace[i], mesh2functionspace[j]] = 1
    
    # 非零位置索引
    A_1_nonzero = A_1_array != 0
    A_2_nonzero = A_2_array != 0
    A_3_nonzero = A_3 != 0

    # 相同非零模式？
    same_1_2 = np.array_equal(A_1_nonzero, A_2_nonzero)
    same_1_3 = np.array_equal(A_1_nonzero, A_3_nonzero)
    same_2_3 = np.array_equal(A_2_nonzero, A_3_nonzero)
    print("Non-zero pattern identical between A1 and A2:", same_1_2)
    print("Non-zero pattern identical between A1 and A3:", same_1_3)
    print("Non-zero pattern identical between A2 and A3:", same_2_3)

    # 可选：分别统计哪一方多出的非零位置
    only_in_A1 = np.argwhere(A_1_nonzero & ~A_2_nonzero)
    only_in_A2 = np.argwhere(~A_1_nonzero & A_2_nonzero)

    print(f"Nonzero only in A1 ({len(only_in_A1)}): {only_in_A1}")
    print(f"Nonzero only in A2 ({len(only_in_A2)}): {only_in_A2}")

    import pyvista
    from dolfinx.plot import vtk_mesh

    grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="white", opacity=0.5)
    for idx in only_in_A1:
        i, j = idx
        pt_i = V2.tabulate_dof_coordinates()[i]
        pt_j = V2.tabulate_dof_coordinates()[j]
        line = pyvista.Line(pt_i, pt_j)
        plotter.add_mesh(line, color="red", line_width=5)
    plotter.show()

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    compare_A_matrix(mesh_file)

    