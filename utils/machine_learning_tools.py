import numpy as np
import h5py
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def load_bsp_pts(path: str = 'forward_inverse_3d/data/geom_ecgsim.mat'):
    geom = h5py.File(path, 'r')
    points = np.array(geom['geom_thorax']['pts'])
    return points


def unwrap_surface(original_pts):
    """
    将环绕身体的 3D 电极点展开到 2D 平面
    返回 2D 坐标 (num_leads, 2)
    """
    X, Y, Z = original_pts[:,0], original_pts[:,1], original_pts[:,2]
    theta = np.arctan2(Y, X)  # 围绕中心旋转角度
    return np.column_stack([theta, Z])  # (num_leads, 2)


def project_bsp_on_surface(bsp_data, original_pts=load_bsp_pts(), length=50, width=50):
    num_timepoints, num_leads = bsp_data.shape

    xy_pts = unwrap_surface(original_pts)  # (num_leads, 2)

    # 构建规则网格
    x_min, x_max = xy_pts[:,0].min(), xy_pts[:,0].max()
    y_min, y_max = xy_pts[:,1].min(), xy_pts[:,1].max()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, length),
        np.linspace(y_min, y_max, width)
    )
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # 一次性构建两个插值器
    lin_interp   = LinearNDInterpolator(xy_pts, np.zeros(num_leads))
    near_interp  = NearestNDInterpolator(xy_pts, np.zeros(num_leads))

    surface_bsp = np.empty((num_timepoints, width, length), dtype=np.float64)

    for t in range(num_timepoints):
        voltages = bsp_data[t, :]

        # 更新插值器数值
        lin_interp.values[:]  = voltages.reshape(-1, 1)
        near_interp.values[:] = voltages
        # 线性插值
        grid_z = lin_interp(grid_pts).reshape(width, length)

        # 最近邻插值（完整）
        grid_z_near = near_interp(grid_pts).reshape(width, length)

        # 补洞，不会再 shape mismatch
        nan_mask = np.isnan(grid_z)
        if np.any(nan_mask):
            grid_z[nan_mask] = grid_z_near[nan_mask]

        surface_bsp[t] = grid_z

    return surface_bsp