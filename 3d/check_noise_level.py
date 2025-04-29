import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('.')
from utils.helper_function import add_noise_based_on_snr, check_noise_level_snr

d_data = np.load('3d/data/u_data_reaction_diffusion_normal_data_argument.npy')
# noise_data_10dB = add_noise_based_on_snr(d_data, snr=10)
# noise_data_20dB = add_noise_based_on_snr(d_data, snr=20)
# noise_data_30dB = add_noise_based_on_snr(d_data, snr=30)

# np.save('3d/data/u_data_reaction_diffusion_normal_data_argument_10dB.npy', noise_data_10dB)
# np.save('3d/data/u_data_reaction_diffusion_normal_data_argument_20dB.npy', noise_data_20dB)
# np.save('3d/data/u_data_reaction_diffusion_normal_data_argument_30dB.npy', noise_data_30dB)
noise = add_noise_based_on_snr(d_data, snr=20)
noise_level = check_noise_level_snr(d_data, noise-d_data)
print(f'Noise level SNR:{noise_level} dB')

index = 10
plt.figure()
plt.plot(d_data[:, index], label='Original data')
plt.plot(noise[:, index], label='Noise data')
plt.title(f'Noise level SNR: {noise_level} dB')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()
plt.show()