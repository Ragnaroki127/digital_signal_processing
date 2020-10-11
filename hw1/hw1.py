import numpy as np
import matplotlib.pyplot as plt

def signal_with_noise(sig_intensity, noise_intensity):
    x = np.arange(0, 100, 0.1)
    signal = sig_intensity * np.sin(x)
    noise = noise_intensity * np.random.normal(0, 1, 1000)
    return x, signal + noise

def variant_frequency_signal(alpha):
    x = np.arange(0, 10, 0.01)
    frequency = 2 * alpha * x
    signal = np.sin(alpha * x**2)
    return x, frequency, signal



if __name__ == "__main__":
    #信号与噪声强度比为10:1
    x1, sig_noise = signal_with_noise(100, 10)
    #变频信号为sin(t^2)
    x2, frequency, signal2 = variant_frequency_signal(1)

    plt.figure(1)
    plt.plot(x1, sig_noise, 'b-')
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.legend(labels=['SNR=20'], loc='best')

    plt.figure(2)
    plt.plot(x2, frequency, '-')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.legend(labels=['Frequency w.r.t time'], loc='best')

    plt.figure(3)
    plt.plot(x2, signal2, '-')
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.legend(labels=['Signal intensity'], loc='upper right')

    plt.show()
