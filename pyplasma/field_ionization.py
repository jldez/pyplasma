#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
from scipy.special import ellipk,ellipe,dawsn
import tqdm

from . import laser as las


# Keldysh ionisation
def fi_rate(material, laser, tol=1e-3):
	"""
	Calculate the field ionization rate according to Keldysh's formula.

		Arguments:
			material (Material object): The material in which the plasma formation
				takes place.

			laser (Laser object): The laser that causes the plasma formation.

		Returns:
			fi_rate (float): The field ionization rate in 1/(sm^3). 
			Divide by material's density to obtain the rate in 1/s.
	"""

	E = abs(laser.E)
	if (E<1e3):
		return 0.0

	# if fi_table != None and laser.E <= fi_table[0,-1]:
	# 	return interp(E, fi_table[0,:], fi_table[1,:])

	def CEI1(a):
		return ellipk(a)
	def CEI2(a):
		return ellipe(a)

	#Dawson integral
	def DI(a):
		return dawsn(a)

	#Keldysh parameter
	def g(E):
		return laser.omega*(material.m_red*material.bandgap)**0.5/(c.e*E)

	#dummy parameters
	def g1(E):
		return g(E)**2.0/(1.0+g(E)**2.0)
	def g2(E):
		return 1.0/(1.0+g(E)**2.0)
	def g3(E):
		return 2.0/c.pi*material.bandgap/(c.hbar*laser.omega)*CEI2(g2(E))/(g1(E))**0.5

	#infinite sum
	def Q(E):
		sol,err,n = 0.0,1e10,0
		c1=(CEI1(g1(E))-CEI2(g1(E)))/CEI2(g2(E))
		c2=2.0*CEI1(g2(E))*CEI2(g2(E))
		while (err > tol):
			term = np.exp(-n*c.pi*c1)*DI(c.pi*((np.floor(g3(E)+1.0)-g3(E)+n)/c2)**0.5)
			sol = sol + term
			n = n+1
			err = term/sol
		return sol*(c.pi/(2.0*CEI1(g2(E))))**0.5

	return 4.0*laser.omega/(9.0*c.pi)*(material.m_red*laser.omega/(c.hbar*g1(E)**0.5))**1.5*Q(E) \
		*np.exp(-c.pi*np.floor(g3(E)+1)*(CEI1(g1(E))-CEI2(g1(E)))/CEI2(g2(E)))




def fi_table(material, laser, N=1e3, tol=1e-3, output=None, progress_bar=True):

	Es = np.logspace(3,np.log10(5*laser.E0),N)
	table = []

	if progress_bar:
		Es = tqdm.tqdm(Es)

	for E in Es:
		fake_laser = las.Fake_Laser(E=E, omega=laser.omega)
		rate = fi_rate(material, fake_laser, tol=tol)
		table.append([E,rate])
	table = np.array(table)

	if output != None:
		np.save(output+'.npy', table)
	return table



if __name__ == '__main__':
	pass