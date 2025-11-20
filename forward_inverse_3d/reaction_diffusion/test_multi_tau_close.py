'''
测试 不同tau_close对 APD 的影响
结论 tau_close 越大 APD 越长
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 固定参数
tau_in = 0.2
tau_out = 10
tau_open = 130
v_crit = 0.23
v_rest = 0
v_peak = 1

def J_stim(t):
    return 0.01 if 40 <= t <= 50 else 0.0

def odes(t, y, tau_close):
    v, h = y

    # 电流项
    J_in = (h * (v_peak - v) * (v - v_rest)**2) / tau_in
    J_out = -(v - v_rest) / tau_out

    dv_dt = J_in + J_out + J_stim(t)

    # 门控变量动力学
    n_gate = 0.1
    h_inf = 0.5 * (1 - math.tanh((v - v_crit) / n_gate))
    dh_dt = (1 / tau_close + (tau_close - tau_open) / tau_open / tau_close * h_inf) * (h_inf - h)

    return [dv_dt, dh_dt]

# 初始条件与时间设置
v0 = v_rest + 0.001
h0 = 1.0
t_span = (0, 600)
t_eval = np.linspace(t_span[0], t_span[1], 2500)

# 设定要测试的 tau_close 值
# tau_close_values = [50, 100, 150, 200, 300]
tau_close_values = [120, 150]

plt.figure(figsize=(8, 5))

for tau_close in tau_close_values:
    sol = solve_ivp(odes, t_span, [v0, h0], t_eval=t_eval, args=(tau_close,))
    t = sol.t
    v = sol.y[0]
    plt.plot(t, v, label=f"$\\tau_{{close}}={tau_close}$")

plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("v (Membrane potential)", fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
