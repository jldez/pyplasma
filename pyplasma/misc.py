#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c

from . import drude as dru


def g_en(material):
	""" electron-molecule collision rate """
	return material.cross_section*(material.density-material.rho)*(2.0*abs(material.Ekin)/material.m_CB/c.m_e)**0.5

def g_hn(material):
	""" hole-molecule collision rate """
	return material.cross_section*(material.density-material.rho)*(2.0*abs(material.Ekin_h)/material.m_VB/c.m_e)**0.5

def g_ee(material):
	""" electron-electron collision rate """
	return 4.*c.pi*c.epsilon_0/c.e**2.*(6./material.m_CB/c.m_e)**.5*(2.*material.Ekin/3.)**1.5


def Ekin_max(material,laser,E,s="eh"):
	""" 
	Upper bound estimation function for the mean kinetic energy of the charge carriers. 
	
		Arguments:
			material (Material object): The material in which the plasma formation
				takes place.

			laser (Laser object): The laser that causes the plasma formation.

			E (numpy array): Electric field of the laser during the simulation.

			s (str): Indicates if the mean kinetic energy is calculated for
				both electrons and holes (s="eh"), the electrons only (s="e")
				or the holes only (s="h"). Default is s="eh".

		Returns:
			Upper bound of the mean kinetic energy in Joules (float).
	"""

	if s == "eh":
		m = material.m_red
	elif s == "e":
		m = material.m_CB
	elif s == "h":
		m = material.m_VB
	m *= c.m_e

	Ep = c.e**2.0*laser.E**2.0/(4.0*material.m_red*c.m_e*(material.damping**2.0+laser.omega**2.0))
	Ec = (1.0+material.m_red/material.m_VB)*(material.bandgap+Ep)

	laser.E = E.max()

	return (-1.5*Ec/(np.log(dru.ibh(material,laser,s=s)*c.hbar*laser.omega\
		/(2.*Ec*material.cross_section*material.density)*(m*c.pi/(3.*Ec))**.5))).max()


if __name__ == '__main__':
	pass