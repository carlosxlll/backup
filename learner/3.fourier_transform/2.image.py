import numpy as np
import cv2
from matplotlib import pyplot as plt
import string

def image_enhancement(image_path, filter_size=30):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2
    # 低通滤波器
    mask_low = np.zeros((rows, cols), np.bool_)
    mask_low[crow-filter_size:crow+filter_size, ccol-filter_size:ccol+filter_size] = 1
    fshift_low = fshift * mask_low  
     
    
    # 高通滤波器
    mask_high = np.ones((rows, cols), np.uint8)
    mask_high[crow-filter_size:crow+filter_size, ccol-filter_size:ccol+filter_size] = 0
    fshift_high = fshift * mask_high


    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    f_ishift_low = np.fft.ifftshift(fshift_low)
    img_back_low = np.fft.ifft2(f_ishift_low)
    img_back_low = np.abs(img_back_low) 

    f_ishift_high = np.fft.ifftshift(fshift_high)
    img_back_high = np.fft.ifft2(f_ishift_high)
    img_back_high = np.abs(img_back_high)


    # 绘制原始图像和增强后的图像
    plt.subplot(241),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(242),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image Back'), plt.xticks([]), plt.yticks([])
    plt.subplot(243),plt.imshow(img_back_low, cmap = 'gray')
    plt.title('Image Back Low'), plt.xticks([]), plt.yticks([])
    plt.subplot(244),plt.imshow(img_back_high, cmap = 'gray')
    plt.title('Image Back High'), plt.xticks([]), plt.yticks([])

    # 绘制频谱图
    plt.subplot(246),plt.imshow(np.log(np.abs(fshift)+1), cmap = 'gray')
    plt.title('Fourier Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(247),plt.imshow(np.log(np.abs(fshift_low)+1), cmap = 'gray')
    plt.title('Fourier Spectrum Low'), plt.xticks([]), plt.yticks([])
    plt.subplot(248),plt.imshow(np.log(np.abs(fshift_high)+1), cmap = 'gray')
    plt.title('Fourier Spectrum High'), plt.xticks([]), plt.yticks([])



    plt.show()






if __name__ == '__main__':
    image_enhancement('./data/4.png', filter_size=30)
