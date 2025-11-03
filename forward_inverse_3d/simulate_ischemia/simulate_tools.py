'''
build simulate parameter
'''
import numpy as np
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import Mesh
from typing import Type, Callable

MARKER_EPI = 1
MARKER_MID = 0
MARKER_ENDO = -1

TAU_CLOSE_EPI = 145
TAU_CLOSE_MID = 150
TAU_CLOSE_ENDO = 155
TAU_CLOSE_SHIFT = 100

TAU_IN = 0.4
TAU_IN_ISCHEMIA = 0.3

D_VAL = 1e-1
D_VAL_ISCHEMIA = 5e-2
D_VAL_SCAR = 0

MI_ISCHEMIA_FACTOR = 0.9
MI_SCAR_FACTOR = 0.0
ME_ISCHEMIA_FACTOR = 0.6
ME_SCAR_FACTOR = 0.3

CAVITY_FACTOR = 3
LUNG_FACTOR = 0.2

def build_tau_close(marker_function: Function, condition: Type[Callable], ischemia=False, vary=False):
    functionspace = marker_function.function_space
    tau_close = Function(functionspace)
    if vary:
        marker_f = marker_function.x.array.round()
        tau_close.x.array[:] = np.where(marker_f == MARKER_EPI, 
                                        TAU_CLOSE_EPI, 
                                        np.where(marker_f == MARKER_MID, 
                                                TAU_CLOSE_MID, 
                                                TAU_CLOSE_ENDO))
    else:
        tau_close.x.array[:] = TAU_CLOSE_MID
    if ischemia:
        condition_instance = condition(TAU_CLOSE_SHIFT, 0)
        coords = functionspace.tabulate_dof_coordinates().T
        tau_close.x.array[:] += condition_instance(coords)
    return tau_close


def build_tau_in(f_space: functionspace, condition: Type[Callable], ischemia=False):
    tau_in = Function(f_space)
    if ischemia:
        tau_in.interpolate(condition(TAU_IN_ISCHEMIA, TAU_IN))
    else:
        tau_in.x.array[:] = TAU_IN
    return tau_in


def build_D(f_space: functionspace, condition: Type[Callable], scar=False, ischemia=False):
    D = Function(f_space)
    if scar:
        D.interpolate(condition(D_VAL_SCAR, D_VAL))
    elif ischemia:
        D.interpolate(condition(D_VAL_ISCHEMIA, D_VAL))
    else:
        D.x.array[:] = D_VAL
    return D


def build_Mi(domain: Mesh, condition: Type[Callable], sigma_i=0.4, scar=False, ischemia=False):
    tdim = domain.topology.dim
    f_space = functionspace(domain, ("DG", 0, (tdim, tdim)))
    coords = f_space.tabulate_dof_coordinates().T
    condition_instance = condition(True, False)
    mask = condition_instance(coords).astype(bool)
    Mi = Function(f_space)

    def rho_i(x):
        n_dofs = x.shape[1]
        tensor = np.eye(tdim) * sigma_i
        values = np.tile(tensor.reshape(-1,1), n_dofs)
        if ischemia:
            values[:, mask] = tensor.reshape(-1,1) * (sigma_i * MI_ISCHEMIA_FACTOR)
        if scar:
            values[:, mask] = tensor.reshape(-1,1) * (sigma_i * MI_SCAR_FACTOR)
        return values
    
    Mi.interpolate(rho_i)
    return Mi

def build_Me(domain: Mesh, condition: Type[Callable], sigma_e=0.8, scar=False, ischemia=False):
    tdim = domain.topology.dim
    f_space = functionspace(domain, ("DG", 0, (tdim, tdim)))
    coords = f_space.tabulate_dof_coordinates().T
    condition_instance = condition(True, False)
    mask = condition_instance(coords).astype(bool)
    Me = Function(f_space)

    def rho_e(x):
        n_dofs = x.shape[1]
        tensor = np.eye(tdim) * sigma_e
        values = np.tile(tensor.reshape(-1,1), n_dofs)
        if ischemia:
            values[:, mask] = tensor.reshape(-1,1) * (sigma_e * ME_ISCHEMIA_FACTOR)
        if scar:
            values[:, mask] = tensor.reshape(-1,1) * (sigma_e * ME_SCAR_FACTOR)
        return values
    
    Me.interpolate(rho_e)
    return Me

def build_M(domain: Mesh, cell_markers, condition: Type[Callable], multi_flag, sigma_i=0.4, sigma_e=0.8, sigma_t=0.8, scar=False, ischemia=False):
    tdim = domain.topology.dim
    f_space = functionspace(domain, ("DG", 0, (tdim, tdim)))
    M = Function(f_space)

    def rho1(x):
        tensor = np.eye(tdim) * sigma_t
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    def rho2(x):
        n_dofs = x.shape[1]
        tensor = np.eye(tdim) * sigma_e
        values = np.tile(tensor.reshape(-1,1), n_dofs)
        condition_instance = condition(True, False)
        mask = condition_instance(x).astype(bool)
        if ischemia:
            values[:, mask] = tensor.reshape(-1,1) * (sigma_e * ME_ISCHEMIA_FACTOR + sigma_i * MI_ISCHEMIA_FACTOR)
        if scar:
            values[:, mask] = tensor.reshape(-1,1) * (sigma_e * ME_SCAR_FACTOR + sigma_i * MI_SCAR_FACTOR)
        return values
    def rho3(x):
        tensor = np.eye(tdim) * sigma_t * LUNG_FACTOR
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    def rho4(x):
        tensor = np.eye(tdim) * sigma_t * CAVITY_FACTOR
        values = np.repeat(tensor, x.shape[1])
        return values.reshape(tensor.shape[0]*tensor.shape[1], x.shape[1])
    
    M.interpolate(rho1, cell_markers.find(1))
    M.interpolate(rho2, cell_markers.find(2))
    if cell_markers.find(3).any():
        if multi_flag == True:
            M.interpolate(rho3, cell_markers.find(3))
        else:
            M.interpolate(rho1, cell_markers.find(3))
    if cell_markers.find(4).any():
        if multi_flag == True:
            M.interpolate(rho4, cell_markers.find(4))
        else:
            M.interpolate(rho1, cell_markers.find(4))
    return M