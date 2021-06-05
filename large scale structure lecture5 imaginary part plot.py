#虚部的图像，不一定对，在这里我取的k=n=1，默认用的一类贝塞尔函数作的图

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

chi = np.linspace(0.,20,10000)
def Ylm(l):
    return special.sph_harm(0,l,0,0)
y1 = 4*np.pi*((1/(4*np.pi)))**0.5*1j**0*special.j0(chi)*Ylm(0)
plt.plot(chi,np.abs(y1),c='r',label='$J_0(x)$')
y2 = 4*np.pi*((3/4/np.pi))**0.5*1j**1*special.j1(chi)*Ylm(1)
plt.plot(chi,np.abs(y2),c='g',label='$J_1(x)$',linestyle="--")
y3 = 4*np.pi*((5/4/np.pi))**0.5*1j**2*special.jv(2, chi)*Ylm(2)
plt.plot(chi,np.abs(y3),c='b',label='$J_2(x)$',linestyle="-.")
plt.legend()
plt.xlabel('$\chi$')

plt.savefig('imaginary part plot')
