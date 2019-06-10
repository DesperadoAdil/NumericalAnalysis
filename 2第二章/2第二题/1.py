# -*- coding: UTF-8 -*-
import numpy as np
from scipy.optimize import fsolve

def f1(x):
    return x**3 - x - 1.0

def f1_(x):
    return 3 * x**2 - 1.0

def f2(x):
    return -1 * x**3 + 5.0 * x

def f2_(x):
    return -3 * x**2 + 5.0


def newton(f, f_, x0, eps1, eps2, lamd, damp=True):
    print ('Damp:') if damp else print ('No damp:')
    k = 0
    xk = x0
    xk_pre = x0
    while abs(f(xk)) > eps1 or abs(xk - xk_pre) > eps2:
        s = f(xk) / f_(xk)
        xk_pre = xk
        xk = xk_pre - s
        if damp:
            i = 0
            lamdk = lamd
            while abs(f(xk)) > abs(f(xk_pre)):
                xk = xk_pre - lamdk * s
                lamdk /= 2.0
                i += 1
        k += 1
        if damp:
            print ("\tlamd%d: %.10f\tx%d: %.10f" % (k, lamdk, k, xk))
        else:
            print ("\tx%d: %.10f" % (k, xk))
    print ("\tx = %.10f\tf(x) = %E" % (xk, f(xk)))
    return (k, xk)


if __name__ == '__main__':
    lamd0 = input('lambda0:')
    lamd0 = np.float64(lamd0) if lamd0 != '' else 0.2
    eps = input('eps:')
    eps = np.float64(eps) if eps != '' else 0.001

    x0 = 0.6
    print ('(1)x^3-x-1=0')
    k, xk = newton(f1, f1_, np.array(x0, dtype=np.float64), eps, eps, lamd0)
    k, xk = newton(f1, f1_, np.array(x0, dtype=np.float64), eps, eps, lamd0, damp=False)
    result = fsolve(f1, x0)
    print ("fsolve:\tx = %.10f\tf(x) = %E" % (result, f1(result)))

    x0 = 1.35
    print ('(2)-x^3+5x=0')
    k, xk = newton(f2, f2_, np.array(x0, dtype=np.float64), eps, eps, lamd0)
    k, xk = newton(f2, f2_, np.array(x0, dtype=np.float64), eps, eps, lamd0, damp=False)
    result = fsolve(f2, x0)
    print ("fsolve:\tx = %.10f\tf(x) = %E" % (result, f2(result)))
