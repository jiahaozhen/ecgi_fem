from forward_inverse_3d.mesh.create_mesh_ecgsim_multi_conduct import create_mesh

if __name__ == '__main__':
    source_file = 'forward_inverse_3d/data/geom_ecgsim.mat'
    target_file = 'forward_inverse_3d/data/mesh/mesh_multi_conduct_lc_{}_lc_ratio_{}.msh'
    for lc in [20, 30, 40, 50, 60, 70, 80]:
        for lc_ratio in [4]:
            create_mesh(source_file, target_file.format(lc, lc_ratio), lc, multi_flag=True, lc_ratio=lc_ratio)