# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.figure(num="第一题")

x = np.logspace(-16, 0, 1000)
x2 = np.logspace(-16, 0, 40)
y = [i/2+2*10**(-16)/i for i in x]
y_real = [np.abs((np.sin(1+i)-np.sin(1))/i-np.cos(1)) for i in x2]
y1 = [i/2 for i in x]
y2 = [2*10**(-16)/i for i in x]

plt.title("例1.4", fontproperties='SimHei')
plt.xlabel("步长h", fontproperties='SimHei', fontsize=14)
plt.ylabel("误差", fontproperties='SimHei', fontsize=14)

plt.loglog(x, y, lw=2, basex=10, basey=10)
plt.loglog(x2, y_real, lw=1, basex=10, basey=10)
plt.loglog(x, y1, basex=10, basey=10, color = 'red', linewidth = 1.0, linestyle = '--')
plt.loglog(x, y2, basex=10, basey=10, color = 'green', linewidth = 1.0, linestyle = '--')
plt.show()

"""
f(x) = sin(x)
f'(x) = cos(x)
f''(x) = -sin(x)

epsilon_real = |(sin(1+h) - sin(1) - h*cos(1)) / h|
=> epsilon_real = |(sin(1+h) - sin(1))/h - cos(1)|
"""
