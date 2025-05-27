import numpy as np
import time
import matplotlib.pyplot as plt
from main_ischemia import resting_ischemia_inversion
from main_ischemia_if_activation_known import ischemia_inversion
import sys
sys.path.append('.')
from utils.helper_function import compute_error_phi, eval_function

def find_lcurve_corner(rho, eta):
    """
    基于离散曲率识别 L-曲线拐点。
    输入:
      rho: array_like, 残差范数 ||Ax - b||，长度 N
      eta: array_like, 解范数 ||x||，长度 N
    返回:
      idx: int, 拐点所在索引
      kappa: float, 对应的最大曲率值
    """
    # 对数坐标
    # x = np.log(rho)   # log(残差)
    # y = np.log(eta)   # log(解范数)
    x = rho
    y = eta
    # 一阶和二阶导数（有限差分）
    dx  = np.gradient(x)    # 一阶导数
    dy  = np.gradient(y)
    d2x = np.gradient(dx)   # 二阶导数
    d2y = np.gradient(dy)
    # 计算曲率
    kappa = np.abs(dx * d2y - dy * d2x) / ( (dx**2 + dy**2) ** 1.5 )
    # 最大曲率位置即拐点
    idx = np.argmax(kappa)
    return idx, kappa[idx]

mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
d = np.load('3d/data/u_data_reaction_diffusion_ischemia_data_argument_20dB.npy')
v = np.load('3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy')
phi_1_exact = np.load("3d/data/phi_1_data_reaction_diffusion_ischemia.npy")
phi_2_exact = np.load("3d/data/phi_2_data_reaction_diffusion_ischemia.npy")

alpha_list = np.logspace(-6, 1, 20)
phi_1_list = []
rho_list = []
eta_list = []
cm_list = []
time_sequence = np.arange(0, 20, 1)
for alpha in alpha_list:
    time_start = time.time()
    phi_1, rho, eta = ischemia_inversion(mesh_file, d_data=d, v_data=v, time_sequence=time_sequence,
                                         phi_1_exact=phi_1_exact, phi_2_exact=phi_2_exact,
                                         alpha1=alpha)
    # phi_1, rho, eta = resting_ischemia_inversion(mesh_file, d_data=d, v_data=v, alpha1=alpha, transmural_flag=True)
    eta = eta / alpha
    phi_1_list.append(phi_1)
    rho_list.append(rho)
    eta_list.append(eta)
    cm_list.append(compute_error_phi(phi_1_exact[0], phi_1.x.array, phi_1.function_space))
    time_end = time.time()
    print(f"alpha: {alpha:.3e}, rho: {rho:.3e}, eta: {eta:.3e}, time: {time_end-time_start:.2f} s")  

rho = np.array(rho_list)
eta = np.array(eta_list)

print(cm_list)

idx_corner, kappa_max = find_lcurve_corner(rho, eta)
alpha_opt = alpha_list[idx_corner]
phi_1_opt = phi_1_list[idx_corner]
print(f"拐点索引: {idx_corner}, 最大曲率: {kappa_max:.3e}")

# 绘制 L-曲线
plt.figure(figsize=(8, 6))
plt.plot(rho, eta, label='L-curve')
plt.xlabel('log10(||Ax - b||)')
plt.ylabel('log10(||x||)')
plt.title('L-Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()