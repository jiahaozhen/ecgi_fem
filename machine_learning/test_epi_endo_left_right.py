from dolfinx.io import gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI
from utils.ventricular_segmentation_tools import distinguish_left_right_endo_epi
from utils.visualize_tools import plot_val_on_domain

mesh_file = r'machine_learning/data/mesh/mesh_multi_conduct_ecgsim.msh'
gdim = 3
domain, cell_markers, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=gdim)
tdim = domain.topology.dim

subdomain_ventricle, _, _, _ = create_submesh(domain, tdim, cell_markers.find(2))

marker = distinguish_left_right_endo_epi(mesh_file, gdim=gdim)

print("Epi:", len(marker[marker == 1]), 
      "Mid:", len(marker[marker == 0]), 
      "Endo:", len(marker[(marker == -1) | (marker == -2)]))

plot_val_on_domain(subdomain_ventricle, 
                   marker, 
                   name="Epi-Endo Marker", 
                   tdim=tdim, 
                   title="Epi-Endo Marker (1: Epi, 0: Mid, -1: Left Endo, -2: Right Endo)")