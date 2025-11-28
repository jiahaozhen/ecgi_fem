import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def transfer_bsp_to_standard12lead(bsp_data: np.ndarray, lead_index: np.ndarray):
    standard12Lead = np.zeros((bsp_data.shape[0], 12))
    # I = VL - VR
    standard12Lead[:,0] = bsp_data[:,lead_index[7]] - bsp_data[:,lead_index[6]]
    # II = VF - VR
    standard12Lead[:,1] = bsp_data[:,lead_index[8]] - bsp_data[:,lead_index[6]]
    # III = VF - VL
    standard12Lead[:,2] = bsp_data[:,lead_index[8]] - bsp_data[:,lead_index[7]]
    # Vi = Vi - (VR + VL + VF) / 3
    standard12Lead[:, 3:9] = bsp_data[:, lead_index[0:6]] - np.mean(bsp_data[:, lead_index[6:9]], axis=1, keepdims=True)
    # aVR = VR - (VL + VF) / 2
    standard12Lead[:, 9] = bsp_data[:, lead_index[6]] - np.mean(bsp_data[:, lead_index[7:9]], axis=1)
    # aVL = VL - (VR + VF) / 2
    standard12Lead[:, 10] = bsp_data[:, lead_index[7]] - np.mean(bsp_data[:, [lead_index[6], lead_index[8]]], axis=1)
    # aVF = VF - (VR + VL) / 2
    standard12Lead[:, 11] = bsp_data[:, lead_index[8]] - np.mean(bsp_data[:, lead_index[6:8]], axis=1)
    
    return standard12Lead


def transfer_bsp_to_standard300lead(bsp_data: np.ndarray, lead_index: np.ndarray = [0, 1, 65]):
    bsp_data = np.asarray(bsp_data, dtype=float)
    bsp_data = bsp_data - np.mean(bsp_data[:, lead_index], axis=1, keepdims=True)
    return bsp_data


def smooth_ecg_gaussian(ecg_matrix, sigma=2.0):
    """
    对 N×12 ECG 信号矩阵进行逐导联高斯平滑。
    sigma: 平滑宽度（越大越平滑）
    """
    ecg_matrix = np.asarray(ecg_matrix, dtype=float)
    if ecg_matrix.ndim != 2:
        raise ValueError(f"输入应为二维矩阵 (N, 12)，当前形状 {ecg_matrix.shape}")
    
    N, leads = ecg_matrix.shape
    smoothed = np.zeros_like(ecg_matrix)

    for i in range(leads):
        sig = np.nan_to_num(ecg_matrix[:, i])
        smoothed[:, i] = gaussian_filter1d(sig, sigma=sigma)
    
    return smoothed


def smooth_ecg_mean(ecg_matrix, window_size=50):
    """
    对 N×12 ECG 信号矩阵进行滑动平均平滑。
    window_size: 滑动窗口长度（奇数）
    """
    ecg_matrix = np.asarray(ecg_matrix, dtype=float)
    if ecg_matrix.ndim != 2:
        raise ValueError(f"输入应为二维矩阵 (N, 12)，当前形状 {ecg_matrix.shape}")

    N, leads = ecg_matrix.shape
    smoothed = np.zeros_like(ecg_matrix)

    kernel = np.ones(window_size) / window_size

    for i in range(leads):
        sig = np.nan_to_num(ecg_matrix[:, i])
        smoothed[:, i] = np.convolve(sig, kernel, mode='same')

    return smoothed


def smooth_ecg_savgol(ecg_matrix, window_length=11, polyorder=3):
    """
    对 N×12 ECG 信号矩阵进行 Savitzky–Golay 平滑。
    window_length: 窗口长度（必须为奇数）
    polyorder: 局部多项式阶数
    """
    ecg_matrix = np.asarray(ecg_matrix, dtype=float)
    if ecg_matrix.ndim != 2:
        raise ValueError(f"输入应为二维矩阵 (N, 12)，当前形状 {ecg_matrix.shape}")

    N, leads = ecg_matrix.shape
    smoothed = np.zeros_like(ecg_matrix)

    # 确保窗口长度有效
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= N:
        window_length = N - 1 if N % 2 == 0 else N

    for i in range(leads):
        sig = np.nan_to_num(ecg_matrix[:, i])
        smoothed[:, i] = savgol_filter(sig, window_length=window_length, polyorder=polyorder)

    return smoothed


def add_noise_based_on_snr(data: np.ndarray, snr: float) -> np.ndarray:
    """
    Add noise to the data based on the specified SNR (Signal-to-Noise Ratio).
    
    Parameters:
    - data: The original data to which noise will be added.
    - snr: The desired SNR in decibels (dB).
    
    Returns:
    - Noisy data with the specified SNR.
    """
    signal_power = np.mean(data**2)
    noise_power = signal_power / (10**(snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    noisy_data = data + noise
    return noisy_data


def check_noise_level_snr(data: np.ndarray, noise: np.ndarray) -> float:
    """
    Check the SNR (Signal-to-Noise Ratio) of the data.
    
    Parameters:
    - data: The original data.
    - noise: The noise added to the data.
    
    Returns:
    - SNR in decibels (dB).
    """
    signal_power = np.mean(data**2)
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr