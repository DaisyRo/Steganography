import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim, mean_squared_error as sk_mse

def compute_metrics(orig, stego):
    a = orig.astype(np.float32)
    b = stego.astype(np.float32)
    mse_v = float(sk_mse(a, b))
    psnr_v = float(sk_psnr(a, b, data_range=255.0))
    ssim_v = float(sk_ssim(a, b, data_range=255.0))
    return {'MSE': mse_v, 'PSNR': psnr_v, 'SSIM': ssim_v}
