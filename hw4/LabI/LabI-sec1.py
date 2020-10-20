import matplotlib.pyplot as plt
import numpy as np

#ground truth spec
#用100s的信号来近似代表真实的谱图
sigL1 = 100000
fs = 1000
nfft1 = sigL1

f1 = 100
f2 = 100.1
f3 = 20.4

t1 = np.arange(0,sigL1)/fs
#the point number nfft//2 reprents fs/2
half_fs1 = nfft1//2
ff1 = fs*np.arange(0,half_fs1+1)/(2*half_fs1)

sig1 = 0.5*np.sin(2*np.pi*f1*t1)+0.8*np.sin(2*np.pi*f2*t1)+0.7*np.sin(2*np.pi*f3*t1)+0.1*np.random.randn(sigL1)
s1 = np.fft.fft(sig1,nfft1)/sigL1
p1 = 10*np.log10(np.real(s1)**2+np.imag(s1)**2)
plt.figure(figsize=(12,4))
plt.plot(ff1,p1[:half_fs1+1],linewidth=2,label="simulated ground truth")
plt.legend()

#看一下20Hz附近
seg1 = np.arange(2000,2100)
plt.figure(figsize=(12,4))
plt.plot(ff1[seg1],p1[seg1],linewidth=2,label="simulated ground truth")
plt.legend()
#看一下100Hz附近
seg2 = np.arange(9950,10050)
plt.figure(figsize=(12,4))
plt.plot(ff1[seg2],p1[seg2],linewidth=2,label="simulated ground truth")
plt.legend()

################################################################################
#信号长度1000点，fs=1000，fft也是1000点，即1s完整信号
sigL2 = 1000
nfft2 = sigL2
t2 = np.arange(0,sigL2)/fs
half_fs2 = nfft2//2
ff2 = fs*np.arange(0,half_fs2+1)/(2*half_fs2)
sig2 = 0.5*np.sin(2*np.pi*f1*t2)+0.8*np.sin(2*np.pi*f2*t2)+0.7*np.sin(2*np.pi*f3*t2)+0.1*np.random.randn(sigL2)
s2 = np.fft.fft(sig2,nfft2)/sigL2
p2 = 10*np.log10(np.real(s2)**2+np.imag(s2)**2)

plt.figure(figsize=(12,4))
plt.plot(ff2,p2[:half_fs2+1],linewidth=2,label="1s,fs=1kHz")
plt.legend()

#看一下100Hz附近
plt.figure(figsize=(12,4))
plt.plot(ff1[9800:10200],p1[9800:10200],linewidth=2,label="simulated ground truth")
plt.plot(ff2[98:103],p2[98:103],'ro',label="1s,fs=1kHz")
plt.legend()
#看一下20Hz附近
plt.figure(figsize=(12,4))
plt.plot(ff1[1800:2100],p1[1800:2100],linewidth=2,label="simulated ground truth")
plt.plot(ff2[18:22],p2[18:22],'ro',label="1s,fs=1kHz")
plt.legend()

################################################################################
#信号长度1000点，fs=1000，fft是100000点，即补零到100s
s3 = np.fft.fft(sig2,nfft1)/sigL2
p3 = 10*np.log10(np.real(s3)**2+np.imag(s3)**2)

plt.figure(figsize=(12,4))
plt.plot(ff1,p3[:half_fs1+1],linewidth=2,label="1s,fs=1kHz,1e5 nfft")
plt.legend()
#看一下20Hz附近
plt.figure(figsize=(12,4))
plt.plot(ff1[seg1],p1[seg1],linewidth=2,label="simulated ground truth")
plt.plot(ff1[seg1],p3[seg1],'ro',label="1s,fs=1kHz,1e5 nfft")
plt.legend()
#看一下100Hz附近
plt.figure(figsize=(12,4))
plt.plot(ff1[seg2],p1[seg2],linewidth=2,label="simulated ground truth")
plt.plot(ff1[seg2],p3[seg2],'ro',label="1s,fs=1kHz,1e5 nfft")
plt.legend()

plt.show()