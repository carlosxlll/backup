import numpy as np
import matplotlib.pyplot as plt
import cv2

def ReconstructedImage(img, fshift, K):
    '''
    :param img: 原图像  
    :param fshift: 傅里叶变换后的频谱
    :param K: 选择的系数个数
    :return: 重构后的图像
    '''
    indices = np.unravel_index(np.argsort(-np.abs(fshift).ravel()), fshift.shape)
    mask = np.zeros_like(fshift)
    mask[indices[0][:K], indices[1][:K]] = 1

    # 重构图像
    fshift_filtered = fshift * mask
    f_filtered = np.fft.ifftshift(fshift_filtered)
    img_reconstructed = np.fft.ifft2(f_filtered).real

    # 计算PSNR
    mse = np.mean((img - img_reconstructed) ** 2)
    psnr = 10 * np.log10(255.0**2 / mse)
    print('PSNR = ', psnr, 'dB')

    return img_reconstructed, psnr

def pltshow(img, img_reconstructed, magnitude_spectrum, phase_spectrum):
    # 可视化
    fig = plt.figure()

    # 显示原图像
    plt.subplot(221),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # 显示重构图像
    plt.subplot(222),plt.imshow(img_reconstructed, cmap = 'gray')
    plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])
    # 显示幅度谱
    plt.subplot(223),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # 显示相位谱
    plt.subplot(224),plt.imshow(phase_spectrum, cmap = 'gray')
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
    # 保存figure到文件
    fig.savefig('./data/output_1py.png')

# 重构图像的PSNR随K的变化
def pltPsnrAboutK(img, fshift, num_K):
    '''
    :param img: 原图像
    :param fshift: 傅里叶变换后的频谱
    :param num_K: K的最大值
    '''
    K_list = range(1, num_K, 10)
    psnr_list = []
    for K in K_list:
        _, psnr = ReconstructedImage(img, fshift, K)
        psnr_list.append(psnr)

    # 可视化
    fig = plt.figure()
    plt.plot(K_list, psnr_list)
    plt.xlabel('K')
    plt.ylabel('PSNR')
    plt.title('PSNR About K')
    fig.savefig('./data/psnr_1py.png')

if __name__ == '__main__':
    img_path = './data/1.png'
    img = cv2.imread(img_path, 0)
    # 二维傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # 计算频谱图
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # 计算相位谱
    phase_spectrum = np.angle(fshift)

    # 重构图像
    img_reconstructed, psnr = ReconstructedImage(img, fshift, 1000)
    # 可视化
    pltshow(img, img_reconstructed, magnitude_spectrum, phase_spectrum)
    # 重构图像的PSNR随K的变化
    pltPsnrAboutK(img, fshift, 38000)


