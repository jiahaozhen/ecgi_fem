import sys

from dolfinx.io import gmshio
from dolfinx.fem import functionspace, Function
from dolfinx.plot import vtk_mesh
from dolfinx.mesh import create_submesh
from mpi4py import MPI
import numpy as np
import pyvista
import multiprocessing

sys.path.append('.')
from utils.helper_function import eval_function, compute_phi_with_v_timebased

# mesh_file = '2d/data/heart_torso.msh'
mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
# mesh of Body
domain, cell_markers, facet_markers = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim = 3)
tdim = domain.topology.dim
# mesh of Heart
subdomain_ventricle, ventricle_to_torso, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
sub_node_num = subdomain_ventricle.topology.index_map(0).size_local

V2 = functionspace(subdomain_ventricle, ("Lagrange", 1))

v_exact_all_time = np.load('3d/data/v_data_reaction_diffusion.npy')
phi_1_exact = np.load('3d/data/phi_1_exact_reaction_diffusion.npy')
phi_2_exact = np.load('3d/data/phi_2_exact_reaction_diffusion.npy')

# phi_1_exact = np.load('2d/data/phi_1_exact_all_time.npy')
# phi_2_exact = np.load('2d/data/phi_2_exact_all_time.npy')
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

# phi_1_diff = np.linalg.norm(phi_1_result - phi_1_exact, axis=1)
# phi_2_diff = np.linalg.norm(phi_2_result - phi_2_exact, axis=1)
# p1 = multiprocessing.Process(target=plot_with_time, args=(phi_1_result, 'phi_1_result'))
p2 = multiprocessing.Process(target=plot_with_time, args=(phi_1_exact, 'phi_1_exact'))
# p3 = multiprocessing.Process(target=plot_with_time, args=(phi_2_result, 'phi_2_result'))
p4 = multiprocessing.Process(target=plot_with_time, args=(phi_2_exact, 'phi_2_exact'))
p5 = multiprocessing.Process(target=plot_with_time, args=(v_exact_all_time, 'v_exact'))
# p1.start()
p2.start()
# p3.start()
p4.start()
p5.start()
# p1.join()
p2.join()
# p3.join()
p4.join()
p5.join()