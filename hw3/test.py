from fft import *


if __name__ == "__main__":
    N = 1024
    x = np.arange(0, 2 * np.pi, 2 * np.pi / N)
    x_n = np.sin(20 * x) + np.cos(50 * x) + np.sin(80 * x) + np.cos(100 * x)

    N = 1024
    M = 128
    overlap = 64
    n = int((N - overlap) / (M - overlap))
    P_x = np.zeros(N)
    for i in range(n):
        mask = np.zeros(N)
        mask[i * (M - overlap):i * (M - overlap) + M] = 1
        x_n_slice = x_n * mask
        P_x_slice = np.abs(np.fft.fft(x_n_slice))**2
        P_x += P_x_slice
    P_x /= M * n

    plt.plot(x, P_x, 'b-')
    plt.show()