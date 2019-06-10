# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.optimize import fsolve

eps = 1.0E-10

def fzerotx(F, ab, *args):
    a = ab[0]
    b = ab[1]
    fa = F(a, *args)
    fb = F(b, *args)
    if np.sign(fa) == np.sign(fb):
        print ("给定区间内可能无根！")
        return
    c = a
    fc = fa
    d = b - c
    e = d
    while fb != 0:
        if np.sign(fa) == np.sign(fb):              #调整a、b的值，使函数F(x)在它们之间改变正负号
            a = c
            fa = fc
            d = b - c
            e = d
        if abs(fa) < abs(fb):                       #交换a、b的值，因为b总是标记最优解
            c = b
            fc = fb
            b = a
            fb = fa
            a = c
            fa = fc
        m = 0.5 * (a-b)
        tol = 2.0 * eps * max(abs(b), 1.0)
        if abs(m) <= tol or fb == 0.0:              #如果收敛就退出
            break
        if (abs(e) < tol) or (abs(fc) <= abs(fb)):  #二分法
            d = e = m
        else:
            s = fb / fc
            if (a == c):                            #割线法
                p = 2.0 * m * s
                q = 1.0 - s
            else:                                   #逆二次插值法
                q = fc / fa
                r = fb / fa
                p = s * (2.0 * m * q * (q-r) - (b-c) * (r-1.0))
                q = (q-1.0) * (r-1.0) * (s-1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if (2.0 * p < 3.0 * m * q - abs(tol * q)) and (p < abs(0.5 * e * q)):
                e = d                               #判断逆二次插值/割线法的结果是否可接受
                d = p / q
            else:
                d = e = m
        c = b                                       #准备下一步迭代，并计算F函数值
        fc = fb
        if abs(d) > tol:
            b += d
        else:
            b -= np.sign(b-a) * tol
        fb = F(b, *args)
    return (b, fb)


if __name__ == '__main__':

    list = [0]
    ansX = [0] * 10
    ansY = [0] * 10
    f = j0(0)
    num = 0
    for i in range(1, 50):
        if np.sign(j0(i)) != np.sign(f):            #符号相反说明有零点
            list.append(i)
            f = j0(i)
            ansX[num], ansY[num] = fzerotx(j0, (list[num], i))
            print ("x%d = %.10f\t fx = %.10E" % (num, ansX[num], ansY[num]))
            num += 1
            if num >= 10:
                break

    plt.figure(num="第三题")
    plt.title("第三题", fontproperties='SimHei')

    x = np.arange(0, 32, 0.01)                      #画出J0(x)的图像
    y = j0(x)
    plt.plot(x, y)
    plt.plot(ansX, ansY, 'ro')                      #标出零点的位置
    plt.show()
