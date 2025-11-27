'''
检查电压数据噪声水平
'''
import numpy as np
import matplotlib.pyplot as plt
from utils.helper_function import add_noise_based_on_snr, check_noise_level_snr

d_data = np.load('forward_inverse_3d/data/inverse/u_data_reaction_diffusion_normal.npy')

noise = add_noise_based_on_snr(d_data, snr=30)
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