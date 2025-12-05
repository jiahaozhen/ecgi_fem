from forward_inverse_3d.mesh.create_mesh_ecgsim_multi_conduct import create_mesh

if __name__ == "__main__":
    lc = 40
    case_name = ["normal_male", "normal_male2", "normal_young_male"]
    case_index = 0  # Change this index to select different cases
    source_file = r'forward_inverse_3d/data/raw_data/geom_{}.mat'.format(
        case_name[case_index]
    )
    target_file = r'forward_inverse_3d/data/mesh/mesh_{}.msh'.format(
        case_name[case_index]
    )
    create_mesh(source_file, target_file, lc)
