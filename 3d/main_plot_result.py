import sys

from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.mesh import create_submesh
from dolfinx.plot import vtk_mesh
import numpy as np
from mpi4py import MPI
import pyvista
import multiprocessing
import matplotlib.pyplot as plt

sys.path.append('.')
from utils.helper_function import eval_function, compute_error_phi

gdim = 3
if gdim == 2:
    mesh_file = '2d/data/heart_torso.msh'
    phi_1_result = np.load('2d/data/phi_1_result.npy')
    phi_2_result = np.load('2d/data/phi_2_result.npy')
    v_result = np.load('2d/data/v_result.npy')
    v_exact = np.load('2d/data/v_data_reaction_diffusion.npy')
    phi_1_exact = np.load('2d/data/phi_1_exact_reaction_diffusion.npy')
    phi_2_exact = np.load('2d/data/phi_2_exact_reaction_diffusion.npy')
else:
    mesh_file = "3d/data/mesh_multi_conduct_ecgsim.msh"
    phi_1_result = np.load('3d/data/phi_1_result.npy')
    phi_2_result = np.load('3d/data/phi_2_result.npy')
    v_result = np.load('3d/data/v_result.npy')
    v_exact = np.load('3d/data/v_data_reaction_diffusion.npy')
    phi_1_exact = np.load('3d/data/phi_1_exact_reaction_diffusion.npy')
    phi_2_exact = np.load('3d/data/phi_2_exact_reaction_diffusion.npy')

# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain_ventricle.topology.index_map(0).size_local

V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

cc_v = []
for i in range(v_exact.shape[0]):
    # cc of v_exact and v_result
    cc_v.append(np.corrcoef(v_exact[i], v_result[i])[0, 1])
cc = np.array(cc_v)
print('cc of v_data and v_result:', np.mean(cc))

cc_phi_1 = []
marker_phi_1 = np.where(phi_1_exact < 0, 1, 0)
marker_phi_1_result = np.where(phi_1_result < 0, 1, 0)
for i in range(phi_1_result.shape[0]):
    cc_phi_1.append(np.corrcoef(marker_phi_1[i], marker_phi_1_result[i])[0, 1])
cc = np.array(cc_phi_1)

cc_phi_2 = []
marker_phi_2 = np.where(phi_2_exact < 0, 1, 0)
marker_phi_2_result = np.where(phi_2_result < 0, 1, 0)
for i in range(phi_2_exact.shape[0]):
    cc_phi_2.append(np.corrcoef(marker_phi_2[i], marker_phi_2_result[i])[0, 1])
cc = np.array(cc_phi_2)

cm_phi_1 = []
for i in range(phi_1_result.shape[0]):
    cm_phi_1.append(compute_error_phi(phi_1_exact[i], phi_1_result[i], V2))
cm_phi_1 = np.array(cm_phi_1)

cm_phi_2 = []
for i in range(phi_2_result.shape[0]):
    cm_phi_2.append(compute_error_phi(phi_2_exact[i], phi_2_result[i], V2))
cm_phi_2 = np.array(cm_phi_2)

def plot_seq(seq, title):
    plt.plot(np.linspace(0, 40, 201), seq)
    plt.xlabel('Time')
    plt.title(title)
    plt.show()

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
            plotter.add_text(f"Time: {i*14 + j*2:.1f} ms", position='lower_right', font_size=9)
            plotter.view_xy()
            plotter.add_title(title, font_size=9)
    plotter.show()

p1 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_1_result < 0, 1, 0), 'ischemia_result'))
p2 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_2_result < 0, 1, 0), 'activation_result'))
p3 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_1_exact < 0, 1, 0), 'ischemia_exact'))
p4 = multiprocessing.Process(target=plot_with_time, args=(np.where(phi_2_exact < 0, 1, 0), 'activation_exact'))
p5 = multiprocessing.Process(target=plot_with_time, args=(v_result, 'v_result'))
p6 = multiprocessing.Process(target=plot_with_time, args=(v_exact, 'v_exact'))
p7 = multiprocessing.Process(target=plot_seq, args=(cc_v, 'cc_v'))
p10 = multiprocessing.Process(target=plot_seq, args=(cm_phi_1, 'cm_phi_1'))
p11 = multiprocessing.Process(target=plot_seq, args=(cm_phi_2, 'cm_phi_2'))
p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p10.start()
p11.start()
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p10.join()
p11.join()