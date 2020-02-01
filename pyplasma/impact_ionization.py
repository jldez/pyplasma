#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
from scipy.special import erfc

from . import drude as dru
from . import misc as mis


def ii_rate(material,laser,dt):
	"""
	Calculate impact ionization rate according to the rate equation model chosen for the material.

		Arguments:
			material (Material object): The material in which the plasma formation
				takes place.

			laser (Laser object): The laser that causes the plasma formation.

			dt (float): The time step of the simulation.

		Returns:
			ii_rate (float): The impact ionization rate in 1/(sm^3). 
			Divide by material's density to obtain the rate in 1/s.

	"""


	if material.rate_equation.lower() in ["single","sre"]:
		return sre(material,laser)
	if material.rate_equation.lower() in ["multiple","mre"]:
		return mre(material,laser,dt)
	if material.rate_equation.lower() in ["delayed","dre"]:
		return dre(material,laser,dt)


def sre(material,laser):
	""" Single rate equation model """

	intensity = c.c*material.index*c.epsilon_0*laser.E**2./2.
	saturation = (material.density-material.rho)/material.density
	return material.alpha_sre*material.rho*intensity*saturation


def mre(material,laser,dt):
	""" Multiple rate equation model """

	if len(material.rho_k) == 0:
		Ep = c.e**2.0*laser.E0**2.0/(4.0*material.m_red*c.m_e*(material.damping**2.0+laser.omega**2.0))
		Ec = (1.0+material.m_red/material.m_VB)*(material.bandgap+Ep)
		material.critical_energy = Ec
		material.k = int(np.ceil(Ec/(c.hbar*laser.omega)))
		material.rho_k = np.zeros((material.k+1))
		material.rho_hk = np.zeros((material.k+1))

	material.Ekin, material.Ekin_h = 0, 0
	for ik in range(material.k+1)[1:]:
		if material.rho > 0:
			material.Ekin += ik*c.hbar*laser.omega*material.rho_k[ik]/material.rho
			material.Ekin_h += ik*c.hbar*laser.omega*material.rho_hk[ik]/material.rho

	ibh, ibh_h = dru.ibh(material,laser,s="e"), dru.ibh(material,laser,s="h")
	coll_freq_en, coll_freq_hn = mis.g_en(material), mis.g_hn(material)

	rho_k_copy = material.rho_k.copy()
	rho_hk_copy = material.rho_hk.copy()

	# Electrons
	material.rho_k[0] += dt*(material.rate_fi - ibh*rho_k_copy[0] + 2.*coll_freq_en*rho_k_copy[-1] \
		+ coll_freq_hn*rho_hk_copy[-1] - material.recombination_rate*rho_k_copy[0])
	for ik in range(material.k)[1:]:
		material.rho_k[ik] += dt*(ibh*(rho_k_copy[ik-1]-rho_k_copy[ik]) \
			- material.recombination_rate*rho_k_copy[ik])
	material.rho_k[-1] += dt*(ibh*rho_k_copy[-2] \
		- coll_freq_en*rho_k_copy[-1] - material.recombination_rate*rho_k_copy[-1])

	# Holes
	material.rho_hk[0] += dt*(material.rate_fi - ibh_h*rho_hk_copy[0] + 2.*coll_freq_hn*rho_hk_copy[-1] \
		+ coll_freq_en*rho_k_copy[-1] - material.recombination_rate*rho_hk_copy[0])
	for ik in range(material.k)[1:]:
		material.rho_hk[ik] += dt*(ibh_h*(rho_hk_copy[ik-1]-rho_hk_copy[ik]) \
			- material.recombination_rate*rho_hk_copy[ik])
	material.rho_hk[-1] += dt*(ibh_h*rho_hk_copy[-2] \
		- coll_freq_hn*rho_hk_copy[-1] - material.recombination_rate*rho_hk_copy[-1])

	return coll_freq_en*material.rho_k[material.k] + coll_freq_hn*material.rho_hk[material.k]


def dre(material,laser,dt):
	""" Delayed rate equation model """

	ibh = dru.ibh(material,laser,s="e")
	ibh_h = dru.ibh(material,laser,s="h")
	coll_freq_en = mis.g_en(material)
	coll_freq_hn = mis.g_hn(material)
	Ep = c.e**2.0*laser.E**2.0/(4.0*material.m_red*c.m_e*(material.damping**2.0+laser.omega**2.0))
	Ec = (1.0+material.m_red/material.m_VB)*(material.bandgap+Ep)
	material.critical_energy = Ec
	re, rh = r(Ec,material.Ekin), r(Ec,material.Ekin_h)
	xi_e, xi_h = xi(re), xi(rh)
	material.xi, material.xi_h = xi_e, xi_h

	temp, temp2 = material.Ekin, material.Ekin_h
	material.Ekin += dt*(ibh*c.hbar*laser.omega - coll_freq_en*xi_e*Ec - \
		material.Ekin*(material.rate_fi/(material.rho+1e-10) + coll_freq_en*xi_e + coll_freq_hn*xi_h)) 
	material.Ekin_h += dt*(ibh_h*c.hbar*laser.omega - coll_freq_hn*xi_h*Ec - \
		material.Ekin_h*(material.rate_fi/(material.rho+1e-10) + coll_freq_en*xi_e + coll_freq_hn*xi_h)) 

	return material.rho*(coll_freq_en*xi_e + coll_freq_hn*xi_h)


def r(Ec,Ekin):
	return abs(3.*Ec/(2.*Ekin+1e-100))**.5
def xi1(r):
	return erfc(r)
def xi2(r):
	return 2.*r/c.pi**.5*np.exp(-r**2.)
def xi(r):
	return xi1(r) + xi2(r)



if __name__ == '__main__':
	pass