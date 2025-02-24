import numpy as np

def calculate_ut_values(coords, source_point):
    """计算ut在给定坐标点的值"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - source_point[0]
    dy = y - source_point[1]
    dz = z - source_point[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    return 1/(4*np.pi*r)

def calculate_ue_values(coords, source_point):
    """计算ue在给定坐标点的值"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - source_point[0]
    dy = y - source_point[1]
    dz = z - source_point[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    return 1/(4*np.pi*r) + x**2 + y**2 + z**2 - 0.1**2

def calculate_ut_gradients(coords, source_point):
    """计算ut在给定坐标点的梯度"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - source_point[0]
    dy = y - source_point[1]
    dz = z - source_point[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r3 = r**3
    
    grad_x = -dx/(4*np.pi*r3)
    grad_y = -dy/(4*np.pi*r3)
    grad_z = -dz/(4*np.pi*r3)
    
    return np.vstack((grad_x, grad_y, grad_z))

def calculate_ue_gradients(coords, source_point):
    """计算ue在给定坐标点的梯度"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - source_point[0]
    dy = y - source_point[1]
    dz = z - source_point[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    r3 = r**3
    
    grad_x = -dx/(4*np.pi*r3) + 2*x
    grad_y = -dy/(4*np.pi*r3) + 2*y
    grad_z = -dz/(4*np.pi*r3) + 2*z
    
    return np.vstack((grad_x, grad_y, grad_z))

