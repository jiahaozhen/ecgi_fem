import numpy as np

# G_tau function
def G_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 1
    result[condition2] = 0
    result[condition3] = 0.5 * (1 + s[condition3] / tau + (1 / np.pi) * np.sin(np.pi * s[condition3] / tau))
    return result

# delta_tau function
def delta_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 0
    result[condition2] = 0
    result[condition3] = (1 / (2 * tau)) * (1 + np.cos(np.pi * s[condition3] / tau))
    return result

# delta^{'}_tau function
def delta_deri_tau(s, tau):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    condition1 = s > tau
    condition2 = s < -tau
    condition3 = ~(condition1 | condition2)
    result = np.zeros_like(s)
    result[condition1] = 0
    result[condition2] = 0
    result[condition3] = -(np.pi / (2 * tau**2)) * np.sin(np.pi * s[condition3] / tau)
    return result

def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s

def OuterBoundary1(x):
    return x[0]**2 + x[1]**2 > 81

def OuterBoundary2(x):
    return (x[0]-4)**2 + (x[1]-4)**2 > 1.8**2

def compare_CM(domain, phi_exact, phi_result):
    marker_exact = np.full(phi_exact.x.array.shape, 0)
    marker_exact[phi_exact.x.array < 0] = 1
    marker_result = np.full(phi_result.x.array.shape, 0)
    marker_result[phi_result.x.array < 0] = 1

    coordinates = domain.geometry.x
    coordinates_ischemic_exact = coordinates[np.where(marker_exact == 1)]
    coordinates_ischemic_result = coordinates[np.where(marker_result == 1)]

    cm1 = np.mean(coordinates_ischemic_exact, axis=0)
    cm2 = np.mean(coordinates_ischemic_result, axis=0)

    return np.linalg.norm(cm1-cm2)