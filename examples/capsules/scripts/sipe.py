import numpy as np
from pylab import *



def eta(epsilon, f, s, kappa_max, N_kappa):

    kappa_x, kappa_y = np.meshgrid(np.linspace(-kappa_max,kappa_max,N_kappa), np.linspace(-kappa_max,kappa_max,N_kappa))
    kappa = np.stack([kappa_x, kappa_y], axis=0)

    F = sqrt(s**2.0+1.0)-s
    G = 0.5*(sqrt(s**2.0+4.0)+s)-sqrt(s**2.0+1.0)
    R = (epsilon-1.0)/(epsilon+1.0)

    gammat = (epsilon-1.0)/(4.0*pi*(1.0+0.5*(1.0-f)*(epsilon-1.0)*(F-R*G)))

    t = 2.0/(1.0+sqrt(epsilon))

    UnMk = sqrt((1.0+0*1j)-kappa[0]**2-kappa[1]**2)
    EpsMk = sqrt(epsilon-kappa[0]**2-kappa[1]**2)

    hss = (2.0*1j)/(UnMk+EpsMk)
    hkk = (2.0*1j)*sqrt((epsilon-kappa[0]**2-kappa[1]**2)*((1+0*1j)-kappa[0]**2-kappa[1]**2))/(epsilon*UnMk+EpsMk)

    v = (hss*(kappa[1]**2/(kappa[0]**2+kappa[1]**2))+hkk*(kappa[0]**2/(kappa[0]**2+kappa[1]**2)))*gammat*norm(t)**2
    eta = 2.0*pi*sqrt((v+conjugate(v))**2)
    return abs(eta)