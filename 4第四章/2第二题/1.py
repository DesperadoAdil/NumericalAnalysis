# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

n = 100
h = np.float64(1/n)
a = 0.5

def create_A(eps):
    A = np.zeros([n-1,n-1],dtype=np.float64)
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                A[i-1][j-1] = -1 * (2*eps + h)
            elif i == j-1:
                A[i-1][j-1] = eps + h
            elif i == j+1:
                A[i-1][j-1] = eps
    return A


def format(f, n=4):
    if round(f)==f:
        m = len(str(f))-1-n
        if f/(10**m) ==0.0:
            return f
        else:
            return float(int(f)/(10**m)*(10**m))
    return round(f, n - len(str(int(f)))) if len(str(f))>n+1 else f


def get_e(x_k, x_k_1):
    x = np.max(np.fabs(np.array(x_k) - np.array(x_k_1)))
    return x


def Jacobi(A, B, x, e, dim):
    k_e = 555555
    k = 0
    e_list = list()
    while k_e > e and k < 1000:
        y = list()
        for i in range(dim):
            y.append(x[i])
        for i in range(dim):
            first = 0
            second = 0
            for j in range(0, i):
                first = first + A[i][j] * y[j]
            for j in range(i+1, dim):
                second = second + A[i][j] * y[j]
            x[i] = (B[i] - first - second) / A[i][i]
        k_e = get_e(x, y)
        k = k + 1
        e_list.append(k_e)
    return e_list, x


def GS(A, B, x, e, dim):
    k_e = 555555
    k = 0
    e_list = list()
    while k_e > e and k < 1000:
        y = list()
        for i in range(dim):
            y.append(x[i])
        for i in range(dim):
            first = 0
            second = 0
            for j in range(0, i):
                first = first + A[i][j] * x[j]
            for j in range(i+1, dim):
                second = second + A[i][j] * x[j]
            x[i] = (B[i] - first - second) / A[i][i]
        k_e = get_e(x, y)
        k = k + 1
        e_list.append(k_e)
    return e_list, x


def SOR(A, B, w, x, e, dim):
    k_e = 555555
    k = 0
    e_list = list()
    while k_e > e:
        y = list()
        for i in range(dim):
            y.append(x[i])
        for i in range(dim):
            first = 0
            second = 0
            for j in range(0, i):
                first = first + A[i][j] * x[j]
            for j in range(i+1, dim):
                second = second + A[i][j] * x[j]
            x[i] = (1.0 - w) * x[i] + w * (B[i] - first - second) / A[i][i]
        k_e = get_e(x, y)
        k = k + 1
        e_list.append(k_e)
    return e_list, x, k


def work(eps):
    A = create_A(eps)                                                   #创建A矩阵
    b = [a*(h**2) for x in range(n-1)]                                  #b = [ah**2] * n-1
    e = 1e-4
    x_0 = [np.float64(0.0) for x in range(n-1)]                         #x(0) = [0] * n-1

    A_I = np.mat(A).I                                                   #求A的逆矩阵
    x_r = A_I.dot(b)                                                    #求解b的精确值
    print ("Exact:")
    print (x_r)                                                         #输出精确值
    print ()

    e_r, x_p = Jacobi(A, b, x_0, e, n-1)                                #Jacobi迭代法计算
    r = x_r - x_p                                                       #计算误差
    print ("Jacobi:")
    print ("|r| = %.20f" % (np.max(abs(r))))                            #误差的无穷范数
    print ("x = ", np.mat([np.float64("%.3E" % x) for x in x_p]))       #输出结果，保留4位有效数字

    print ()

    e_r, x_p = GS(A, b, x_0, e, n-1)                                    #GS迭代法计算
    r = x_r - x_p
    print ("GS:")
    print ("|r| = %.20f" % (np.max(abs(r))))
    print ("x = ", np.mat([np.float64("%.3E" % x) for x in x_p]))
    print ()

    e_r, x_p, k = SOR(A, b, 1.25, x_0, e, n-1)                          #SOR迭代法计算
    r = x_r - x_p
    print ("SOR:")
    print ("|r| = %.20f" % (np.max(abs(r))))
    print ("x = ", np.mat([np.float64("%.3E" % x) for x in x_p]))


if __name__ == '__main__':
    work(1.0)
    #work(0.1)
    #work(0.01)
    #work(0.0001)
