import scipy.io as sio
import scipy.interpolate as sinter
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided

#以下程序提供了.mat文件读取的功能
########################################
mat_contents = sio.loadmat('sound4')
sig = mat_contents['sig']
fs = mat_contents['fs']
sig = sig[:,0]
fs = np.asscalar(fs)
sigLen = sig.size
t = np.arange(0,sigLen)/fs
########################################

print(sigLen)

sig_pad = np.append(sig[:10000], np.zeros(38000))
sig_freq_abs = np.abs(np.fft.fft(sig_pad))
sig_freq = np.append(sig_freq_abs[:10000], np.zeros(380000))
h = [1, -1]
sig_freq_conv = np.convolve(sig_freq_abs, h, mode='same')
for i in range(sigLen - 1):
    if sig_freq_conv * sig_freq_conv


plt.figure(1)
plt.plot(t * fs, sig_freq_abs, 'b-')
plt.scatter(fs * t[sig_index], sig_freq_abs[sig_index], '*')
plt.show()


