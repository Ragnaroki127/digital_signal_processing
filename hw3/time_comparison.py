import time
from fft import *

if __name__ == "__main__":
    N_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    time_custom = []
    time_np = []
    for N in N_list:
        x = np.arange(0, 2, 2 / N)
        x_n = np.cos(50 * 2 * np.pi * x)

        start_time = time.time()
        X_k = fft_custom(x_n, N)
        end_time = time.time()
        time_custom.append(end_time - start_time)

        start_time = time.time()
        X_k = np.fft.fft(x_n, N)
        end_time = time.time()
        time_np.append(end_time - start_time)
    
    plt.figure(1)
    p1, = plt.plot(N_list, time_custom, 'b-')
    p2, = plt.plot(N_list, time_np, 'r-')
    plt.title('Time consumption')
    plt.xlabel('N')
    plt.ylabel('time/s')
    plt.legend([p1, p2], ['Custom fft function', 'Numpy fft function'], loc='upper left')
    plt.show()