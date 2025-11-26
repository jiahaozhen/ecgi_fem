# 检验内外膜划分是否正确
from utils.ventricular_segmentation_tools import distinguish_epi_endo
from utils.visualize_tools import plot_val_on_mesh

mesh_file = r'forward_inverse_3d/data/mesh_multi_conduct_ecgsim.msh'
gdim = 3

epi_endo_marker = distinguish_epi_endo(mesh_file, gdim=gdim)

print("Epi:", len(epi_endo_marker[epi_endo_marker == 1]), 
      "Mid:", len(epi_endo_marker[epi_endo_marker == 0]), 
      "Endo:", len(epi_endo_marker[epi_endo_marker == -1]))

plot_val_on_mesh(mesh_file, 
                 epi_endo_marker, 
                 target_cell=2,
                 name="Epi-Endo Marker", 
                 title="Epi-Endo Marker (1: Epi, 0: Mid, -1: Endo)")