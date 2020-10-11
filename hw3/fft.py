import numpy as np
import matplotlib.pyplot as plt


def W_Nk(N, k, inverse=False):
    if not inverse:
        return np.exp(-1j * 2 * np.pi / N * np.arange(0, k, 1))
    else:
        return np.exp(1j * 2 * np.pi / N * np.arange(0, k, 1))

def fft_custom(x_n, N):
    assert N & (N - 1) == 0
    if N == 2:
        return np.array([x_n[0] + x_n[1], x_n[0] - x_n[1]])
    else:
        A_n = x_n[0:N:2]
        B_n = x_n[1:N:2]
        A_k = fft_custom(A_n, int(N / 2))
        B_k = fft_custom(B_n , int(N / 2))
        W = W_Nk(N, int(N / 2))
        X1_k = A_k + W * B_k
        X2_k = A_k - W * B_k
        X_k = np.append(X1_k, X2_k, axis=0)
        return X_k

def ifft_custom(X_k, N):
    assert N & (N - 1) == 0
    if N == 2:
        return 0.5 * np.array([X_k[0] + X_k[1], X_k[0] - X_k[1]])
    else:
        A_k = X_k[0:N:2]
        B_k = X_k[1:N:2]
        A_n = ifft_custom(A_k, int(N / 2))
        B_n = ifft_custom(B_k, int(N / 2))
        W = W_Nk(N, int(N / 2), inverse=True)
        x1_n = 0.5 * (A_n + W * B_n)
        x2_n = 0.5 * (A_n - W * B_n)
        x_n = np.append(x1_n, x2_n, axis=0)
        return x_n

def self_correlate(x_n, N):
    X_k = fft_custom(x_n, N)
    X_p = 1 / N * np.abs(X_k)**2
    corr = np.real(ifft_custom(X_p, N))
    return corr


if __name__ == "__main__":
    N = 2 ** 14
    x = np.arange(0, 2 * np.pi, 2 * np.pi / N)
    x_n = np.sin(2 * x) + np.cos(5 * x)

    X_k1 = fft_custom(x_n, N)

    X_k2 = np.fft.fft(x_n)

    x_n_inverse = np.real(ifft_custom(X_k1, N))

    plt.figure(1)
    plt.plot(x, np.abs(X_k1), 'b-')
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.title('Frequency domain using custom FFT function')

    plt.figure(2)
    plt.plot(x, np.abs(X_k2), 'b-')
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.title('Frequency domain using np.fft.fft')

    plt.figure(3)
    plt.plot(x, x_n_inverse, 'r-')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title('Time domain using custom IFFT function')

    plt.figure(4)
    plt.plot(x, x_n, 'y-')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title('Original Signal')

    plt.figure(5)
    plt.plot(x, self_correlate(x_n, N), 'g-')
    plt.xlabel('m')
    plt.ylabel('amplitude')
    plt.title('Self-correlation function')
    plt.show()
