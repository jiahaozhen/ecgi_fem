from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
import numpy as np
from mpi4py import MPI

import sys
sys.path.append('.')
from utils.function_tools import eval_function

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    gdim = 3
    sigma_t = 0.8
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim

    num_cells = domain.topology.index_map(tdim).size_local
    print("Cell indices:", np.arange(num_cells))

    V = functionspace(domain, ("DG", 0, (tdim, tdim)))
    ret = V.tabulate_dof_coordinates()

    def rho1(x):
        tensor = np.eye(tdim) * sigma_t
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    
    M = Function(V)
    M.interpolate(rho1, cell_markers.find(1))

    M_Val = eval_function(M, domain.geometry.x[0:10])
    print("Conductivity tensor M at first 10 mesh points:")
    print(M_Val)

    f_coords = V.tabulate_dof_coordinates()
    mesh_coords = domain.geometry.x

    print("f_coords shape:", f_coords.shape)
    print("mesh_coords shape:", mesh_coords.shape)

    dofs_per_cell = V.element.space_dimension

    # 获取单元节点索引
    cells = domain.topology.connectivity(domain.topology.dim, 0)
    cells = cells.array.reshape(-1, domain.geometry.dim + 1)  # 三角形每个单元有3个节点

    # 获取节点坐标
    node_coords = domain.geometry.x

    # 计算单元重心
    cell_centers = np.array([node_coords[cell].mean(axis=0) for cell in cells])
    
    print("Are DG0 DOF coordinates equal to cell centers?")
    print(np.allclose(f_coords, cell_centers))
    

