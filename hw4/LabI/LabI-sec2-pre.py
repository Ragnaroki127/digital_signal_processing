import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

#读取实验数据并且转换成一维数组
sig = np.load("LabIsig.npy")
sig = sig.squeeze()

#数据N点
N = 4000
#采样率2000Hz
fs = 2000
#时间标签
t = np.arange(0,N)*(1/fs)


#绘制信号
plt.figure(figsize=(10,5))
plt.plot(t,sig,linewidth=2,label="signal")

#坐标标注和调整
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.ylim(-0.5,0.5)
#显示图例
plt.legend()

#归一化到1
sig = sig/np.max(sig)

#信号分段
#计算需要多少段
segL = 20
overlap = 10
delta = segL-overlap
segNum = np.int32(np.ceil((N-overlap)/delta));
#扩展信号
padNum = segNum*delta+overlap-N
if padNum==0:
    sigEx = sig
elif padNum>0:
    sigEx = np.hstack((sig,np.zeros(padNum)))    

#分段标签
segIdx = np.arange(0,segNum)*delta
#生成分段矩阵
segMat = as_strided(sigEx,shape=(segNum,segL),strides=(sigEx.strides[0]*delta,sigEx.strides[0]))

E1 = 1 / segL * np.sum(segMat**2, axis=1)
E2 = np.max(np.abs(segMat), axis=1)

plt.figure(2)
plt.plot(np.arange(segMat.shape[0]), E1.T.squeeze(), 'b-')
plt.plot(np.arange(segMat.shape[0]), E2.T.squeeze(), 'r-')
plt.title('Signal Energy')

h1 = np.array([1, 0, -1])
h2 = np.array([-1, 0, 1])
E1_conv_back = np.convolve(E1, h1, mode='same')
E1_conv_for = np.convolve(E1, h2, mode='same')

conv_sig = E1_conv_back * E1_conv_for

plt.figure(3)
plt.plot(np.arange(segNum), E1_conv_for, 'r')
plt.plot(np.arange(segNum), E1_conv_back, 'g')
plt.plot(np.arange(segNum), conv_sig, 'b-')
plt.show()





