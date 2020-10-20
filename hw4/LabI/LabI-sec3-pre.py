# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:22:25 2020

@author: qx-HW
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
#时频原子参数
N = 1000
fs = 1000
alpha = 100
t = np.arange(0,N)*(1/fs)
t0 = 0.4
f0 = 0
f1 = 100
f = (f1-f0)*t+f0

sig = ((alpha/np.pi)**0.25)*np.exp(-0.5*alpha*(t-t0)*(t-t0))*np.sin(2*np.pi*f*t)

#fft点数以及频率轴
nfft = 200
half_fs = nfft//2
ff = fs*np.arange(0,half_fs+1)/(2*half_fs)

segL = ff.shape[0]
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

pMat = np.abs(np.fft.fft(segMat, axis=1))
plt.contourf(t[segIdx], ff, pMat[:, 0:half_fs + 1].T, 50)

plt.show()
