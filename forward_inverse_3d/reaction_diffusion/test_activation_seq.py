import numpy as np
import h5py
from forward_inverse_3d.reaction_diffusion.simulate_reaction_diffusion import compute_v_based_on_reaction_diffusion
from utils.helper_function import get_activation_time_from_v
from utils.function_tools import extract_data_from_function
from utils.visualize_tools import plot_triangle_mesh
from utils.simulate_tools import get_activation_dict

def get_function_space(mesh_file, gdim=3):
    '''
    获取心脏跨膜电压的函数空间
    '''
    from dolfinx.io import gmshio
    from dolfinx.mesh import create_submesh
    from dolfinx.fem import functionspace
    from mpi4py import MPI
    # mesh of Body
    domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
    tdim = domain.topology.dim
    # mesh of Heart
    subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))
    V = functionspace(subdomain_ventricle, ("Lagrange", 1))
    return V

if __name__ == "__main__":
    # compute tmp
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
    T = 500
    step_per_timeframe = 8

    activation_dict = get_activation_dict(mesh_file, threshold=40)

    v_data, _, _ = compute_v_based_on_reaction_diffusion(mesh_file, 
                                                         T=T, 
                                                         step_per_timeframe=step_per_timeframe,
                                                         tau_in_val=0.4,
                                                         activation_dict_origin=activation_dict)
    
    # get activation from tmp (257)
    activation_f = get_activation_time_from_v(v_data) / step_per_timeframe
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    ventricle_pts = np.array(geom['geom_ventricle']['pts'])
    activation_1 = extract_data_from_function(activation_f, get_function_space(mesh_file), ventricle_pts).reshape(-1)

    # get activation from ecgsim
    
    activation_times = h5py.File(r'forward_inverse_3d/data/activation_times_ecgsim.mat', 'r')
    activation_2 = np.array(activation_times['dep']).reshape(-1)

    # plot
    ventricle_fac = np.array(geom['geom_ventricle']['fac'])-1

    import multiprocessing

    p1 = multiprocessing.Process(target=plot_triangle_mesh, args=(ventricle_pts, ventricle_fac), kwargs={'point_values': activation_1, 'title':'pde'})
    p2 = multiprocessing.Process(target=plot_triangle_mesh, args=(ventricle_pts, ventricle_fac), kwargs={'point_values': activation_2, 'title':'ecgsim'})
    p1.start()
    p2.start()

    p1.join()
    p2.join()