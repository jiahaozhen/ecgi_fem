import sys

from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from dolfinx.fem import functionspace, form, Constant, Function, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.mesh import create_submesh, locate_entities_boundary
from dolfinx.plot import vtk_mesh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import multiprocessing
import matplotlib.pyplot as plt

sys.path.append('.')
from utils.helper_function import eval_function

gdim = 3
if gdim == 2:
    mesh_file = '2d/data/heart_torso.msh'
    phi_1_result = np.load('2d/data/phi_1_result.npy')
    phi_2_result = np.load('2d/data/phi_2_result.npy')
    v_result = np.load('2d/data/v_result.npy')
    v_exact = np.load('2d/data/v_data_reaction_diffusion.npy')
else:
    mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
    phi_1_result = np.load('3d/data/phi_1_result.npy')
    phi_2_result = np.load('3d/data/phi_2_result.npy')
    v_result = np.load('3d/data/v_result.npy')
    v_exact = np.load('3d/data/v_data_reaction_diffusion.npy')

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim = gdim)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain_ventricle.topology.index_map(0).size_local

cc = []
for i in range(v_exact.shape[0]):
    # cc of v_exact and v_result
    cc.append(np.corrcoef(v_exact[i], v_result[i])[0, 1])
cc = np.array(cc)
plt.plot(cc)
plt.show()

V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

def plot_with_time(value, title):
    v_function = Function(V2)
    plotter = pyvista.Plotter(shape=(3, 7))
    for i in range(3):
        for j in range(7):
            plotter.subplot(i, j)
            grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain_ventricle, tdim))
            v_function.x.array[:] = value[i*70 + j*10]
            grid.point_data[title] = eval_function(v_function, subdomain_ventricle.geometry.x)
            grid.set_active_scalars(title)
            plotter.add_mesh(grid, show_edges=True)
            plotter.add_text(f"Time: {(i*70 + j*10)/5.0:.1f} ms", position='lower_right', font_size=9)
            plotter.view_xy()
            plotter.add_title(title, font_size=9)
    plotter.show()

p1 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_1_result < 0, 1, 0), 'ischemic'))
p2 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_2_result < 0, 1, 0), 'activation'))
p3 = multiprocessing.Process(target=plot_with_time, args=(v_result, 'v_result'))
p4 = multiprocessing.Process(target=plot_with_time, args=(v_exact, 'v_exact'))
p1.start()
p2.start()
p3.start()
p4.start()
p1.join()
p2.join()
p3.join()
p4.join()