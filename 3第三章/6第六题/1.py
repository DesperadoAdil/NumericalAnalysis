# -*- coding: UTF-8 -*-
import numpy as np

def create_H(dim):
    H = np.ones([dim,dim],dtype=np.float64)
    for i in range(1, dim+1):
        for j in range(1, dim+1):
            H[i-1][j-1] = 1.0 / np.float64(i+j-1)
    return H


def Cholesky_L(H, dim):
    for j in range(dim):
        for k in range(j):
            H[j][j] = H[j][j] - (H[j][k]**2)
        H[j][j] = np.sqrt(H[j][j])
        for i in range(j+1, dim):
            for k in range(j):
                H[i][j] = H[i][j] - H[i][k] * H[j][k]
            H[i][j] = H[i][j] / H[j][j]
    L = np.zeros([dim, dim], dtype=np.float64)
    for i in range(dim):
        for j in range(i+1):
            L[i][j] = H[i][j]

    for i in range(dim):
        for j in range(i+1, dim):
            H[i][j] = H[j][i]
    return H, L


def aft_generation(L, b, dim):
    x = np.zeros([dim], dtype=np.float64)
    for i in range(dim):
        k = i
        i = dim-1-i
        x[i] = b[i]
        for j in range(k):
            j = dim-1-j
            x[i] = x[i] - L[i][j] * x[j]
        x[i] = x[i] / L[i][i]
    return x


def pre_generation(L, b, dim):
    x = np.zeros([dim], dtype=np.float64)
    for i in range(dim):
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - L[i][j]*x[j]
        x[i] = x[i] / L[i][i]
    return x


def figure_x(A, b, dim):
    H, L = Cholesky_L(np.copy(A), dim)
    L_t = np.transpose(np.copy(L))
    y = pre_generation(np.copy(L), b, dim)
    x = aft_generation(np.copy(L_t), y, dim)
    return x


def work(dim, eps):
    H = create_H(dim)                                                           #创建希尔伯特矩阵H
    s_1 = np.max(abs(np.sum(H, 1)))                                              #无穷范数
    H_I = np.mat(H).I                                                           #H的逆矩阵
    s_2 = np.max(abs(np.sum(H_I, 1)))                                            #逆矩阵的无穷范数

    x_f = np.ones([dim], dtype=np.float64)                                      #创建x向量
    b = H.dot(x_f)                                                              #b = H*x
    b_eps = b + eps                                                             #b加上扰动
    x_L = figure_x(H, b_eps, dim)                                               #求解方程Hx=b ==>x^
    b_L = H.dot(x_L)                                                            #b^ = H*x^

    r_b = b - b_L                                                               #计算残差r = b - Hx^
    r_x = x_f - x_L                                                             #计算误差x^-x
    print ("Dim = %d\teps_b = %.10f" % (dim, eps))
    print ("cond(H) = %d\t|H| = %d\t|H(-1)| = %d" % (s_1*s_2, s_1, s_2))
    print ("|r_b| = %.20f"% (np.max(abs(r_b))))
    print ("|r_x| = %.20f"% (np.max(abs(r_x))))
    print ()
    return (np.max(abs(r_b)), np.max(abs(r_x)), s_1*s_2)


if __name__ == '__main__':
    work(10, 0.0)
    work(10, 1e-7)
    work(8, 0.0)
    work(12, 0.0)
