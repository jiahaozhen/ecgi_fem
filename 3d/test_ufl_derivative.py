from dolfinx import *
from mpi4py import MPI
import ufl
import numpy as np

# 定义网格和函数空间
mesh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.functionspace(mesh, ("CG", 1))
phi = fem.Function(V)
v = ufl.TestFunction(V)

phi.x.array[:] = np.random.randn(phi.x.array.shape[0])

# 参数设置
epsilon = 1e-3
delta_phi = ufl.exp(-(phi**2)/(epsilon**2)) / (epsilon * ufl.sqrt(ufl.pi))

# 定义泛函 F[phi]
F = delta_phi * ufl.sqrt(ufl.dot(ufl.grad(phi), ufl.grad(phi))) * ufl.dx

# 计算导数 dF/dphi
dF = ufl.derivative(F, phi, v)
form_dF = fem.form(dF)
J_p = fem.assemble_vector(form_dF)
print(J_p.array)
# # 输出结果形式
# print("泛函 F 的导数形式为:")
# print(dF)