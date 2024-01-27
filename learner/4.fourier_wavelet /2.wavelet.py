import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt

img_path = './data/1.png'
img = cv2.imread(img_path, 0)
print(img.shape)

# Haar小波变换
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

# 选择不同数量的最大系数
K_values = range(1, 9605, 10)
psnr_values = []

for K in K_values:
    # 重构图像
    cA_sorted = np.sort(np.abs(cA.flatten()))
    K = min(K, len(cA_sorted))  # 确保K不超过数组大小
    threshold = cA_sorted[-K]
    cA_filtered = cA * (np.abs(cA) >= threshold)
    img_reconstructed = pywt.idwt2((cA_filtered, (cH, cV, cD)), 'haar')

    # 计算PSNR
    mse = np.mean((img - img_reconstructed) ** 2)
    psnr = 10 * np.log10(255.0**2 / mse)
    psnr_values.append(psnr)

# 显示重构图像
plt.figure()
plt.plot(K_values, psnr_values)
plt.xlabel('K')
plt.ylabel('PSNR')
plt.savefig('./data/psnr_2py.png')




