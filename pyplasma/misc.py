#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c

from . import drude as dru


# electron-molecule collision rate
def g_en(material):
	return material.cross_section*(material.density-material.rho)*(2.0*abs(material.Ekin)/material.m_CB)**0.5

# hole-molecule collision rate
def g_hn(material):
	return material.cross_section*(material.density-material.rho)*(2.0*abs(material.Ekin_h)/material.m_VB)**0.5

# electron-electron collision rate
def g_ee(material):
	return 4.*c.pi*c.epsilon_0/c.e**2.*(6./material.m_CB)**.5*(2.*material.Ekin/3.)**1.5


def Ekin_max(material,laser,E,s="eh"):

	if s == "eh":
		m = material.m_red
	elif s == "e":
		m = material.m_CB
	elif s == "h":
		m = material.m_VB

	Ep = c.e**2.0*laser.E**2.0/(4.0*material.m_red*(material.damping**2.0+laser.omega**2.0))
	Ec = (1.0+material.m_red/material.m_VB)*(material.bandgap+Ep)

	laser.E = E.max()

	return (-1.5*Ec/(np.log(dru.ibh(material,laser,s=s)*c.hbar*laser.omega\
		/(2.*Ec*material.cross_section*material.density)*(m*c.pi/(3.*Ec))**.5))).max()


if __name__ == '__main__':
	pass