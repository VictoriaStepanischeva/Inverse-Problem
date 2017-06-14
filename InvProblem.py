import numpy as np
import scipy
from scipy.integrate import quad
from scipy import special
import matplotlib.pyplot as plt
import math
import sys

def ga(x):
    if (x <= 6):
        return 1 / np.sqrt(4 * np.pi) * sum(map(lambda k: 1 / math.factorial(k) * (k + 1.0 / 2.0) * special.zeta(k + 3.0 / 2.0) * (-x**2 / 4)**k, range(0, 41)))
    else:
        return -np.sqrt(2 * np.pi) * np.exp(-np.sqrt(np.pi) * x) * np.sin(np.sqrt(np.pi) * x + np.pi/4.0)

def g(x, t):
    return 1 / (np.sqrt(t)) * ga(x / np.sqrt(t))

def integrand1(p):
    return lambda s, x, t: np.sqrt(t / np.pi) * p(s) * (np.exp(-(x - s) * (x - s) / (4 * t)) - np.exp(-(x + s) * (x + s) / (4 * t)))

def integrand2(p):
    return lambda s, x, t: 1.0 / 2.0 * p(s) * (abs(x + s) * (1 - special.erf(abs(x + s) / np.sqrt(4 * t))) - abs(x - s) * (1 - special.erf(abs(x - s) / np.sqrt(4 * t))))

def integrandp(I1, I2, T):
    return lambda y, x, t: 1 / T * (I1(y, t) + I2(y, t)) * (g(y - x, t) - g(y + x, t))

def quadint(integrand):
    return lambda x, t: quad(integrand, 0, np.inf, args = (x, t), epsabs = 1.49e-7, epsrel = 1.49e-7, limit = 10000)[0]

def derivative(I1, I2, h):
    return lambda x, t: (I1(x + h, t) + I2(x + h, t) - 2*(I1(x, t) + I2(x, t)) + I1(x - h, t) + I2(x - h, t)) / (h**2)

def vosstanovit(p):
    T = 1.0
    h = 0.01
    I1 = quadint(integrand1(p))
    I2 = quadint(integrand2(p))
    I3 = quadint(integrandp(I1, I2, T))
    der = derivative(I1, I2, h)
    vecI1 = np.vectorize(I1)
    vecI2 = np.vectorize(I2)

    # X range definition
    X0 = np.arange(0.0, 5.0, 0.1)
    X1 = np.arange(0.0, 5.0, 0.2)

    # Source источник plot
    PB =[ p(X1[i]) for i in range(len(X1)) ]
    plt.axis([0, 7, 0, 3])
    plt.xlabel('x', fontsize = 18)
    plt.ylabel('p(x)')
    plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
    plt.plot(X1, PB, color = 'black', linewidth = 10.0)
    plt.show()

    U = vecI1(X0, 1.0) + vecI2(X0, 1.0)
    U1 = vecI1(X0, 5.0) + vecI2(X0, 5.0)
    U2 = vecI1(X0, 10.0) + vecI2(X0, 10.0)
    U3 = vecI1(X0, 15.0) + vecI2(X0, 15.0)

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
    plt.plot(X0, U, linewidth = 10.0)
    plt.plot(X0, U1, linewidth = 10.0)
    plt.plot(X0, U2, linewidth = 10.0)
    plt.plot(X0, U3, linewidth = 10.0)
    plt.show()

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
    plt.plot(X0, U, color = 'darkgreen',linewidth = 10.0)
    plt.show()

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
    plt.plot(X0, U2, color = 'red',linewidth = 10.0)
    plt.show()

    UP = [ der(X0[i], T) for i in range(len(X0)) ]
    plt.plot(X0, UP)
    plt.show()

    P = [ I3(X0[x], T) for x in range(len(X0)) ]
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.plot(X0, np.array(P) - np.array(UP))
    plt.show()

    plt.axis([0,9,0,6])
    plt.tick_params(axis = 'both', which = 'major', labelsize = 16)
    plt.plot(X0, np.array(P) - np.array(UP), color ='orange', linewidth=10.0, label ='reconstructed')
    plt.plot(X1, PB, 'ro', color ='black', linewidth=8.0, label='original')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    method = sys.argv[1]
    if method == 'method1':
        vosstanovit(lambda s: 1.0/2.0 * (abs(s-2) + 2 - s))
    elif method == 'method2':
        vosstanovit(lambda s: 5 * np.exp(-2 * np.abs(s - 2)) - np.exp(-2 * np.abs(s - 5)))
    elif method == 'method3':
        vosstanovit(lambda s: 5 * np.exp(-2 * np.abs(s - 2)))
    elif method == 'method4':
        vosstanovit(lambda s: 2*np.exp(-2*(s-2)**2))
    else:
        print("Unknown method")
