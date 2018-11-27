'''
Código adaptado de 
https://gluon.mxnet.io/chapter06_optimization/optimization-intro.html
'''

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * np.cos(np.pi * x)

x = np.arange(-1.0, 2.0, 0.1)
fig = plt.figure()
subplt = fig.add_subplot(111)
subplt.annotate('mínima local', xy=(-0.3, -0.2), xytext=(-0.8, -1.0),
            arrowprops=dict(facecolor='black', shrink=0.05))
subplt.annotate('mínima global', xy=(1.1, -0.9), xytext=(0.7, 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot(x, f(x))
plt.show()


x = np.arange(-2.0, 2.0, 0.1)
fig = plt.figure()
subplt = fig.add_subplot(111)
subplt.annotate('ponto de sela', xy=(0, -0.2), xytext=(-0.4, -5.0),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot(x, x**3)
plt.show()

