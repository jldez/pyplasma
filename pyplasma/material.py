#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
from scipy.interpolate import *
import cmath

from . import drude
from . import field_ionization as fi
from . import impact_ionization as ii



class Material(object):

	"""
	Object used to define the medium to be irradiated by a laser. 

	Arguments:
		name (str): The name of the material, only used as an id.

		rate_equation (str): The rate equation model to use when calculating plasma 
			formation rate. Choose "single" or "sre" to use single rate equation. Choose 
			"multiple" or "mre" for multiple rate equations. Choose "delayed" or "dre" for 
			the delayed rate equations. Default is "delayed".

		index (float): Linear refractive index. Default is 1.

		bandgap (float): The bandgap between the conduction and the valence bands in Joules.

		m_CB (float): The effective mass of the electrons in the conduction band in units of
			their free space mass. Default is 1.

		m_VB (float): The effective mass of the holes in the valence band in units of the
			free space electron mass. Default is 1.

		density (float): The density of atoms of molecules in 1/m^3.

		cross_section (float): The cross section for the calculation of the collision rate 
			between charge carriers and neutral atoms or molecules in 1/m^2.

		damping (float): The damping rate of the plasma in 1/s. Default is 0.

		recombination_rate (float): The recombination rate of the charge carriers in 1/s.
			Default is 0.
			
		alpha_sre (float): The impact rate coefficient used if the rate equation is SRE in
			m^2/J. Default is 0.
		
	"""

	def __init__(self, name="", rate_equation="delayed", index=1, bandgap=0, m_CB=1, m_VB=1, \
				 density=0, cross_section=0, damping=0, recombination_rate=0, alpha_sre=0, \
				 chi2=0, chi3=0, resonance=0):
		super(Material, self).__init__()
		self.name = name
		self.rate_equation = rate_equation
		self.index = index
		self.chi2 = chi2
		self.chi3 = chi3
		self.resonance = resonance
		self.bandgap = bandgap
		self.m_CB = m_CB*c.m_e
		self.m_VB = m_VB*c.m_e
		if m_CB !=0 and m_VB !=0 :
			self.m_red = (self.m_CB**-1.+self.m_VB**-1.)**-1.
		else: 
			self.m_red = self.m_CB
		self.density = density
		self.cross_section = cross_section
		self.damping = damping
		self.recombination_rate = recombination_rate

		if self.rate_equation.lower() in ["single","sre"]:
			self.alpha_sre = alpha_sre
		if self.rate_equation.lower() in ["multiple","mre"]:
			self.rho_k = []
			self.rho_hk = []
			self.Ec = 0
			self.k = 0
		if self.rate_equation.lower() in ["delayed","dre"]:
			self.xi = 0	
			self.xi_h = 0		

		self.rho = 0
		self.Ekin = 0
		self.Ekin_h = 0
		self.electric_current = 0


	def __copy__(self):
		return Material(name=self.name, rate_equation=self.rate_equation, \
						index=self.index, bandgap=self.bandgap, m_CB=self.m_CB/c.m_e, \
						m_VB=self.m_VB/c.m_e, density=self.density, \
						cross_section=self.cross_section, damping=self.damping, \
						recombination_rate=self.recombination_rate, alpha_sre=self.alpha_sre)


	def plasma_freq(self):
		return abs(c.e**2*self.rho/(c.epsilon_0*self.m_red))**.5

	def update_electric_current(self, laser, dt):
		self.electric_current = self.electric_current*(1.0-dt*self.damping) \
							  + dt*c.epsilon_0*self.plasma_freq()*laser.E

	def update_rho(self, laser, dt):
		if self.rate_equation != 'None':
			try:
				# Much faster to do linear interpolation even if log interpolation should be done instead.
				self.rho_fi = np.interp(np.abs(laser.E), self.fi_table[:,0], self.fi_table[:,1])*(self.density-self.rho)/self.density
			except:
				self.rho_fi = fi.fi_rate(self,laser)*(self.density-self.rho)/self.density
			self.rho_ii = ii.ii_rate(self,laser,dt)
			self.rho_re = self.recombination_rate*self.rho
			self.rho += dt*(self.rho_fi + self.rho_ii - self.rho_re)

	def Drude_index(self, laser):
		self.drude_index = cmath.sqrt(self.index**2. - \
			self.plasma_freq()**2./(laser.omega**2.+1j*laser.omega*self.damping))
		return self.drude_index

	def Reflectivity(self, laser):
		n = self.Drude_index(laser)
		self.reflectivity = abs((n-1.)/(n+1.))**2.
		return self.reflectivity

	def add_fi_table(self, fi_table):
		self.fi_table = fi_table



def log_interp(zz, xx, yy):
	logz = np.log10(zz)
	logx = np.log10(xx)
	logy = np.log10(yy)
	return np.power(10.0, np.interp(logz, logx, logy))



if __name__ == '__main__':
	pass