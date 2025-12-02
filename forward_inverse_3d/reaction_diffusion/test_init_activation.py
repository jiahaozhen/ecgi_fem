from utils.visualize_tools import plot_scatter_on_mesh
from utils.simulate_tools import get_activation_dict

if __name__ == "__main__":
    mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'

    activation_dict = get_activation_dict(mesh_file, mode='IVS', threshold=60)

    plot_scatter_on_mesh(mesh_file, scatter_pts=list(activation_dict.values()))