import sys

from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import locate_entities_boundary, create_submesh
from dolfinx.plot import vtk_mesh
import h5py
import numpy as np
import scipy.interpolate
import pyvista

sys.path.append('.')
from utils.helper_function import eval_function

def ecgsim2fem(mesh_file, ischemic=False, t=39, plot_flag=False):
    """
    Converts ECGsim data to a finite element mesh for body and heart domains, interpolates potential data, 
    and optionally visualizes the results.

    Args:
        mesh_file (str): The path to the mesh file for the body domain.
        ischemic (bool, optional): Flag to indicate if ischemic ECG simulation data should be used. 
                                    If False, sinus rhythm data is used. Defaults to False.
        t (int, optional): The time index to extract specific ECG simulation data. Defaults to 39.

    Returns:
        None
    """
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

    V1 = functionspace(domain, ("Lagrange", 1))
    V2 = functionspace(subdomain, ("Lagrange", 1))
    u = Function(V1)

    geom_data_ecgsim = h5py.File('3d/data/geom_ecgsim.mat', 'r')
    v_pts_ecgsim = np.array(geom_data_ecgsim['geom_ventricle']['pts'])
    d_pts_ecgsim = np.array(geom_data_ecgsim['geom_thorax']['pts'])

    if ischemic:
        file_ecgsim = h5py.File('3d/data/ischemic_ecgsim.mat', 'r')
    else:
        file_ecgsim = h5py.File('3d/data/sinus_rhythm_ecgsim.mat', 'r')
    v_data_ecgsim = np.array(file_ecgsim['tmp'])
    d_data_ecgsim = np.array(file_ecgsim['surface_potential'])

    v_pts_fem = V2.tabulate_dof_coordinates()
    d_index = locate_entities_boundary(domain, tdim-3, lambda x: np.full(x.shape[1], True, dtype=bool))
    d_pts_fem = V1.tabulate_dof_coordinates()[d_index]

    v_data_fem = []
    d_data_fem = []
    data_num = v_data_ecgsim.shape[0]

    for i in range(data_num):
        # v
        if ischemic:
            # griddata
            v_fem_one = scipy.interpolate.griddata(v_pts_ecgsim, v_data_ecgsim[i], v_pts_fem, method='linear', fill_value=-90)
        else:
            # rbf
            rbf = scipy.interpolate.Rbf(v_pts_ecgsim[:,0], v_pts_ecgsim[:,1], v_pts_ecgsim[:,2], v_data_ecgsim[i])
            v_fem_one = rbf(v_pts_fem[:,0], v_pts_fem[:,1], v_pts_fem[:,2])
        v_data_fem.append(v_fem_one.copy())

        # d
        # rbf
        rbf = scipy.interpolate.Rbf(d_pts_ecgsim[:,0], d_pts_ecgsim[:,1], d_pts_ecgsim[:,2], d_data_ecgsim[i])
        d_fem_one = rbf(d_pts_fem[:,0], d_pts_fem[:,1], d_pts_fem[:,2])
        u.x.array[d_index] = d_fem_one
        d_data_fem.append(u.x.array[:].copy())

    np.save('3d/data/v_all.npy', v_data_fem)
    np.save('3d/data/d_all.npy', d_data_fem)
    if ischemic:
        np.save('3d/data/v.npy', v_data_fem[t])
        np.save('3d/data/d.npy', d_data_fem[t])
        v = Function(V2)
        v.x.array[:] = v_data_fem[t]
        if plot_flag:
            grid1 = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, tdim))
            grid1.point_data["v"] = eval_function(v, subdomain.geometry.x)
            grid1.set_active_scalars("v")
            grid2 = pyvista.UnstructuredGrid(*vtk_mesh(domain, tdim))
            grid2.point_data["u"] = d_data_fem[t]
            grid2.set_active_scalars("u")
            grid = (grid1, grid2)

            plotter = pyvista.Plotter(shape=(1,2))
            for i in range(2):
                plotter.subplot(0, i)
                plotter.add_mesh(grid[i], show_edges=True)
                plotter.view_xy()
                plotter.add_axes()
            plotter.show()


if __name__ == '__main__':
    file = '3d/data/mesh_multi_conduct_ecgsim.msh'
    ecgsim2fem(file, ischemic=True, plot_flag=True)