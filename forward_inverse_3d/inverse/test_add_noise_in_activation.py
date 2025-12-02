import numpy as np
import multiprocessing
from utils.transmembrane_potential_tools import get_activation_time_from_v
from utils.visualize_tools import plot_val_on_mesh
from utils.signal_processing_tools import add_noise_based_on_snr

v_file = 'forward_inverse_3d/data/inverse/v_data_ischemia.npy'
v_exact = np.load(v_file)

activation_time_exact = get_activation_time_from_v(v_exact)
activation_time_noise = add_noise_based_on_snr(activation_time_exact, snr=20)

print("Correlation between exact and noisy activation times:", 
      np.corrcoef(activation_time_exact, activation_time_noise)[0, 1])

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'

p1 = multiprocessing.Process(target=plot_val_on_mesh, kwargs={'mesh_file': mesh_file,
                                                              'val': activation_time_exact,
                                                              'title': "Exact Activation Time",
                                                              'target_cell': 2,
                                                              'f_val_flag': True})
p2 = multiprocessing.Process(target=plot_val_on_mesh, kwargs={'mesh_file': mesh_file,
                                                              'val': activation_time_noise,
                                                              'title': "Noisy Activation Time",
                                                              'target_cell': 2,
                                                              'f_val_flag': True})
p1.start()
p2.start()
p1.join()
p2.join()