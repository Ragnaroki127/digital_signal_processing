import numpy as np
import cmath
import matplotlib.pyplot as plt

def generate_WN(N):
    a = np.arange(0, N, 1)
    W_coef = a[np.newaxis].T @ a[np.newaxis]

    W_N = np.exp(-2 * np.pi / N * 1j) ** W_coef
    return W_N

if __name__=='__main__':
    '''
    N = 100
    a = np.arange(0, N, 1)
    W_coef = a[np.newaxis].T @ a[np.newaxis]

    W_N = np.exp(-2 * np.pi / N * 1j) ** W_coef
    '''
    N = 100
    W_N = generate_WN(N)

    x_n = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / N))[np.newaxis].T
    X_k = W_N @ x_n

    x_array = np.arange(0, 2 * np.pi, 2 * np.pi / N)

    plt.figure(1)
    plt.bar(x_array, x_n.T.squeeze(), width=0.01)
    plt.xlabel('time')
    plt.ylabel('amplitude')

    plt.figure(2)
    plt.bar(x_array , np.abs(X_k).T.squeeze(), width=0.01)
    plt.xlabel('frequency')
    plt.ylabel('amplitude')

    plt.figure(3)
    fft_result = np.fft.fft(x_n.T.squeeze())
    plt.bar(x_array, np.abs(fft_result), width=0.01)
    plt.xlabel('frequency')
    plt.ylabel('amplitude')

    plt.figure(4)
    plt.plot(x_array , np.angle(X_k).T.squeeze(), 'b-')
    plt.xlabel('frequency')
    plt.ylabel('angle')

    plt.figure(5)
    fft_result = np.fft.fft(x_n.T.squeeze())
    plt.plot(x_array , np.angle(fft_result), 'r-')
    plt.xlabel('frequency')
    plt.ylabel('angle')

        plt.show()