import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft

def create_data(N:int):
    '''
    生成数据
    '''
    t = np.linspace(0, 1, N, endpoint=False)
    signal = np.sin(2*np.pi*20*t) + np.sin(2*np.pi*40*t)
    # 添加随机噪声
    noise = np.random.normal(0, 0.5, signal.shape)
    signal_noise = signal + noise

    return t, signal, signal_noise


def observe_freq(freq:list,freq_observe:list[list]):
    '''
    根据保留频率区间生成对应的二值列表

    Params:
    freq: 生成的频率列表 
    freq_observe: 要保留的频率列表 [(a1,b1), (a2,b2),..., (an,bn)]

    Returns:
    mask: 二值列表，其中在指定区间内的元素为True，其他元素为False


    '''
    # 创建一个全为False的数组
    mask = np.zeros_like(freq, dtype=bool)

    # 对每个区间进行处理
    for a, b in freq_observe:
        mask |= (freq >= a) & (freq <= b)
        mask |= (freq >=-b) & (freq <= -a)
    return mask

def visualize(t, signal, signal_noise, signal_filtered, freq, F):
        # 绘制原始信号，加噪声信号，傅里叶恢复的信号
    plt.figure(figsize=(12, 9))

    plt.subplot(3, 2, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.subplot(3, 2, 2)
    plt.plot(t, signal_noise)
    plt.title('Signal with Noise')
    plt.subplot(3, 2, 3)
    plt.plot(t, signal_filtered)
    plt.title('Filtered Signal')

    # 绘制频率图
    # 绘制幅度
    plt.subplot(3, 2, 4)
    plt.plot(freq[:50], abs(F)[:50], label='Amplitude')
    plt.title('Frequency Domain')

    plt.show()


if __name__ == '__main__':
    t, signal, signal_noise = create_data(1000)

    # 傅里叶变换
    F = fft.fft(signal_noise)
    freq = fft.fftfreq(t.shape[-1])
    # 滤掉噪声
    F_filtered = F.copy()
    print(freq)
    mask = observe_freq(freq, [(0.018, 0.022),(0.038,0.042)])
    print(mask)
    F_filtered[~mask] = 0
    # 恢复信号
    signal_filtered = fft.ifft(F_filtered)

    visualize(t, signal, signal_noise, signal_filtered, freq, F)


