# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

plt.figure(num="第三题")

plt.title("第一、二问", fontproperties='SimHei')
sum = np.float32(0)
max = np.power(2, 24)
x = np.arange(0, max, 1)
y = np.zeros(max, dtype=np.float32, order='C')
i = 0
for i in range(1, max):
    y[i] += y[i-1] + np.float32(1/i)
    if y[i] == y[i-1]:
        print ("float32", i, y[i])
        break
max = i+1
plt.plot(x[0:max], y[0:max])

sum = np.float64(0)
y2 = np.zeros(max, dtype=np.float64, order='C')

start_time = time.time()
for i in range(1, max):
    y2[i] += y2[i-1] + np.float64(1/i)
end_time = time.time()
print ('time cost',end_time-start_time,'s')

plt.plot(x[0:max], y2[0:max])
print ("float64", max-1, y2[max-1])
e = np.abs((y2[max-1] - y[max-1]) / y[max-1])
print (e)

plt.show()
