# 关于dolfinx中网格与函数的索引对齐

## 网格中点的顺序

假设网格是domain，网格中点的顺序为domain.geometry.x

## V = functionspace(domain, ("Lagrange", 1))中点的顺序

假设 f是定义的V上的函数，即f = Function(V)

此时可通过设置f.x.array设置点的顺序, 即 f.x.array[:] = new_val

但new_val中点的顺序与网格中点顺序不同

V中的点的顺序可通过V.tabulate_dof_coordinates()得到

## 索引对齐函数

utils.function_tools中有函数fspace2mesh，函数参数为functionspace（要求是("Lagrange", 1)）。输出的结果是函数空间中点到网格的映射。

例：假设返回值为fspace2mesh_map, fspace2mesh_map[0]象征着函数空间中序号为0的点（即及第一个点）在网格中的索引

通过mesh2functionspace = np.argsort(functionspace2mesh)可以的得到网格中点到函数空间的映射
