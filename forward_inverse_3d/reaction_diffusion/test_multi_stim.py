'''
测试 多次刺激电流对膜电位的影响
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 固定参数
tau_in = 0.2
tau_out = 10
tau_close = 150
tau_open = 50  # 可调整
tau_open_list = [30, 50, 100]  # 可用于多组对比
v_crit = 0.23
v_rest = 0
v_peak = 1

# 多次刺激电流函数
def J_stim_multi(t):
    stim_times = [(40, 50), (150, 250), (300, 330)]
    for t_start, t_end in stim_times:
        if t_start <= t <= t_end:
            return 0.01
    return 0.0

def odes(t, y, tau_open):
    v, h = y
    J_in = (h * (v_peak - v) * (v - v_rest)**2) / tau_in
    J_out = -(v - v_rest) / tau_out
    dv_dt = J_in + J_out + J_stim_multi(t)
    n_gate = 0.1
    h_inf = 0.5 * (1 - math.tanh((v - v_crit) / n_gate))
    dh_dt = (1 / tau_close + (tau_close - tau_open) / tau_open / tau_close * h_inf) * (h_inf - h)
    return [dv_dt, dh_dt]

# 初始条件与时间设置
v0 = v_rest + 0.001
h0 = 1.0
t_span = (0, 600)
t_eval = np.linspace(*t_span, 3000)

plt.figure(figsize=(10, 6))
for tau_open in tau_open_list:
    sol = solve_ivp(lambda t, y: odes(t, y, tau_open), t_span, [v0, h0], t_eval=t_eval)
    plt.plot(sol.t, sol.y[0], label=f"tau_open={tau_open}")

plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v')
plt.title('Response to Multiple Stimuli')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
