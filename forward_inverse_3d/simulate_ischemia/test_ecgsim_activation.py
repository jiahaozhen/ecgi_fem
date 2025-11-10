import numpy as np
import h5py

if __name__ == '__main__':
    geom = h5py.File(r'forward_inverse_3d/data/geom_ecgsim.mat', 'r')
    ventricle_pts = np.array(geom['geom_ventricle']['pts'])

    activation_times = h5py.File(r'forward_inverse_3d/data/activation_times_ecgsim.mat', 'r')
    activation = np.array(activation_times['dep']).reshape(-1)

    assert ventricle_pts.shape[0] == activation.shape[0], \
        "Number of ventricle points does not match number of activation times"
    
    assert np.unique(activation).shape[0] == activation.shape[0]

    # create dict time -> coords
    activation_dict = {}
    for i in range(ventricle_pts.shape[0]):
        time = activation[i]
        coord = ventricle_pts[i]
        activation_dict[time] = coord