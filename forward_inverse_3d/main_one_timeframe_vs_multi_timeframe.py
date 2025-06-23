import sys

import numpy as np
import pyvista
import multiprocessing
from dolfinx.plot import vtk_mesh
from main_ischemia_if_activation_known import ischemia_inversion

sys.path.append('.')
from utils.helper_function import compute_error_phi, eval_function

mesh_file = '3d/data/mesh_multi_conduct_ecgsim.msh'
v_exact_data_file = '3d/data/v_data_reaction_diffusion_ischemia_data_argument.npy'
d_data_file = '3d/data/u_data_reaction_diffusion_ischemia_data_argument_20dB.npy'
phi_1_file = "3d/data/phi_1_data_reaction_diffusion_ischemia.npy"
phi_2_file = "3d/data/phi_2_data_reaction_diffusion_ischemia.npy"

v_exact = np.load(v_exact_data_file)
d_data = np.load(d_data_file)
phi_1_exact = np.load(phi_1_file)
phi_2_exact = np.load(phi_2_file)

phi_1_result_1 = ischemia_inversion(mesh_file, d_data=d_data, v_data=v_exact,
                                    time_sequence=np.arange(0, 1, 1),
                                    phi_1_exact=phi_1_exact, phi_2_exact=phi_2_exact,
                                    alpha1=1e-2, plot_flag=False)

phi_1_result_2 = ischemia_inversion(mesh_file, d_data=d_data, v_data=v_exact, 
                                    time_sequence=np.arange(1000, 1201, 10),
                                    phi_1_exact=phi_1_exact, phi_2_exact=phi_2_exact,
                                    alpha1=1e-2, plot_flag=False)
phi_1_exact_f = phi_1_result_1.copy()
phi_1_exact_f.x.array[:] = phi_1_exact[0]

cm1 = compute_error_phi(phi_1_exact[0], phi_1_result_1.x.array, phi_1_result_1.function_space)
cm2 = compute_error_phi(phi_1_exact[0], phi_1_result_2.x.array, phi_1_result_2.function_space)
print("cm1:", cm1)
print("cm2:", cm2)

# plot
def plot_f_on_subdomain(f, subdomain, title):
    grid = pyvista.UnstructuredGrid(*vtk_mesh(subdomain, 3))
    grid.point_data["f"] = eval_function(f, subdomain.geometry.x)
    grid.set_active_scalars("f")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_title(title)
    plotter.view_yz()
    plotter.add_axes()
    plotter.show()

subdomain = phi_1_result_1.function_space.mesh
phi_1_exact_f.x.array[:] = np.where(phi_1_exact[0] < 0, 1, 0)
phi_1_result_1.x.array[:] = np.where(phi_1_result_1.x.array < 0, 1, 0)
phi_1_result_2.x.array[:] = np.where(phi_1_result_2.x.array < 0, 1, 0)
p1 = multiprocessing.Process(target=plot_f_on_subdomain, args=(phi_1_exact_f, subdomain, 'phi_1_exact'))
p2 = multiprocessing.Process(target=plot_f_on_subdomain, args=(phi_1_result_1, subdomain, 'phi_1_result_1'))
p3 = multiprocessing.Process(target=plot_f_on_subdomain, args=(phi_1_result_2, subdomain, 'phi_1_result_2'))
p1.start()
p2.start()
p3.start()
p1.join()
p2.join()
p3.join()