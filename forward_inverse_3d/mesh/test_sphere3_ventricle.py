import matplotlib.pyplot as plt
import numpy as np


def create_truncated_ellipsoid(
    R_a, R_b, R_c, truncation_Z, offset_x=0, offset_y=0, offset_z=0
):
    """
    生成一个截断椭球体的坐标，沿着Z轴在指定的truncation_Z平面处截断。
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    X = R_a * np.outer(np.cos(u), np.sin(v)) + offset_x
    Y = R_b * np.outer(np.sin(u), np.sin(v)) + offset_y
    Z = R_c * np.outer(np.ones(np.size(u)), np.cos(v)) + offset_z

    mask = Z > truncation_Z

    X[mask] = np.nan
    Y[mask] = np.nan
    Z[mask] = np.nan

    return X, Y, Z


# --- 1. 定义几何体参数 ---
TRUNCATION_PLANE_Z = 1.0

# --- 关键修改：减小内部腔体半径 ---
REDUCTION_FACTOR = 0.9  # 新的腔体半径是旧的 90%
CAVITY_RA = 1.5 * REDUCTION_FACTOR  # 变为 1.35
CAVITY_RB = 1.4 * REDUCTION_FACTOR  # 变为 1.26
CAVITY_RC = 2.9 * REDUCTION_FACTOR  # 变为 2.61

# 偏移量需要根据新半径重新计算，确保腔体仍不相交
# 新的总直径为 2 * 1.35 = 2.7。我们设置中心间距为 2.8
OFFSET = 1.4  # 1.4 * 2 = 2.8
LV_OFFSET_X = -OFFSET
RV_OFFSET_X = OFFSET
Z_OFFSET_CAVITY = -1.0

# 心外膜参数 (保持上次调整后的包裹性参数)
Epi_Ra = 3.3
Epi_Rb = 1.5 * 1.2  # Y轴乘数设为1.2
Epi_Rc = 3.0 * 1.15  # Z轴乘数设为1.15
Epi_OFFSET_X = 0.0
Z_OFFSET_EPI = Z_OFFSET_CAVITY

# --- 2. 创建几何体 ---
LV_X, LV_Y, LV_Z = create_truncated_ellipsoid(
    CAVITY_RA,
    CAVITY_RB,
    CAVITY_RC,
    TRUNCATION_PLANE_Z,
    offset_x=LV_OFFSET_X,
    offset_z=Z_OFFSET_CAVITY,
)
RV_X, RV_Y, RV_Z = create_truncated_ellipsoid(
    CAVITY_RA,
    CAVITY_RB,
    CAVITY_RC,
    TRUNCATION_PLANE_Z,
    offset_x=RV_OFFSET_X,
    offset_z=Z_OFFSET_CAVITY,
)
Epi_X, Epi_Y, Epi_Z = create_truncated_ellipsoid(
    Epi_Ra,
    Epi_Rb,
    Epi_Rc,
    TRUNCATION_PLANE_Z,
    offset_x=Epi_OFFSET_X,
    offset_z=Z_OFFSET_EPI,
)


# --- 3. 绘图设置 ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 定义颜色和线条宽度
COLOR_LV = 'darkred'
COLOR_RV = 'deepskyblue'
COLOR_EPI = 'lightgray'
NEW_LINEWIDTH = 1.5
NEW_STRIDE = 5

# --- 4. 绘制心外膜 (Epicardium) ---
ax.plot_surface(
    Epi_X,
    Epi_Y,
    Epi_Z,
    color=COLOR_EPI,
    rstride=3,
    cstride=3,
    linewidth=0,
    edgecolor=None,
    antialiased=False,
    alpha=0.15,
    label='Epicardium',
)

# --- 5. 绘制心室腔 (LV/RV) ---
ax.plot_surface(
    LV_X,
    LV_Y,
    LV_Z,
    color=COLOR_LV,
    rstride=NEW_STRIDE,
    cstride=NEW_STRIDE,
    linewidth=NEW_LINEWIDTH,
    edgecolor='black',
    antialiased=False,
    alpha=1.0,
    label='Left Ventricle',
)

ax.plot_surface(
    RV_X,
    RV_Y,
    RV_Z,
    color=COLOR_RV,
    rstride=NEW_STRIDE,
    cstride=NEW_STRIDE,
    linewidth=NEW_LINEWIDTH,
    edgecolor='black',
    antialiased=False,
    alpha=1.0,
    label='Right Ventricle',
)


# --- 6. 完善视图 ---
ax.set_axis_off()

ax.view_init(elev=30, azim=270)
ax.dist = 10

ax.set_title("ventricle", color='black')

plt.show()
