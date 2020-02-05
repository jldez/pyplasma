"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
import cmath
import copy

from . import field_ionization as fi
from . import impact_ionization as ii

from .misc import *
from .backend import backend as bd



class Material():
	"""
	Object used to define the material parameters. 

	Arguments:

		index (float): Linear refractive index. Default is 1.

		resonance (float): Resonance frequency [Hz] for the Lorentz model. Ignored for 0D simulations.

		chi2 (float): Second order susceptibility [m/V]. Ignored for 0D simulations.

		chi3 (float): Third order susceptibility [m^2/V^2]. Ignored for 0D simulations.


		drude_params (dict): Contains the relevant parameters for Drude model. The keys are:

			damping (float): Plasma damping rate [Hz].

			m_CB (float): Effective mass of the electrons in the conduction band (relative to free mass). Default is 1.

			m_VB (float): Effective mass of the holes in the valence band (relative to free mass). Default is 1.

			rho (float): Plasma density [1/m^3]. Default is 0.


		ionization_params (dict): Contains the relevant parameters for ionization. The keys are:

			rate_equation (str): The rate equation model to use when calculating plasma 
				formation rate. Choose "single" or "sre" to use single rate equation. Choose 
				"multiple" or "mre" for multiple rate equations. Choose "delayed" or "dre" for 
				the delayed rate equations. Choose 'fi' for field ionization only. Default is "none".

			bandgap (float): The bandgap between the conduction and the valence bands in Joules.

			density (float): The density of atoms or molecules in 1/m^3.

			cross_section (float): The cross section for the calculation of the collision rate 
				between charge carriers and neutral atoms or molecules in 1/m^2. 
				In use if rate equation is MRE or DRE.

			recombination_rate (float): The recombination rate [Hz] of the charge carriers. Default is 0.
				
			alpha_sre (float): The impact rate coefficient [m^2/J] used if the rate equation is SRE.

			fi_damping_correction (bool): To use or not a correction factor to Keldsh field ionization
				that accounts for plasma damping. Default is False.
		
	"""


	def __init__(self, index, resonance=0, chi2=0, chi3=0, drude_params:dict={}, ionization_params:dict={}):

		self.index = index
		self.resonance = resonance
		self.chi2 = chi2
		self.chi3 = chi3
		self.rho = 0

		self.trackables = []

		if drude_params == {}:
			self.drude = False
		else:
			self.drude = True
			self.damping = drude_params['damping']
			self.m_CB = 1.0
			self.m_VB = 1.0
			if 'm_CB' in drude_params:
				self.m_CB = drude_params['m_CB']
			if 'm_VB' in drude_params:
				self.m_VB = drude_params['m_VB']
			if self.m_CB !=0 and self.m_VB !=0 :
				self.m_red = (self.m_CB**-1.+self.m_VB**-1.)**-1.
			else: 
				self.m_red = self.m_CB
			if 'rho' in drude_params:
				self.rho = drude_params['rho']
			
		if ionization_params == {}:
			# error if D > 0 and not drude
			self.rate_equation = 'none'
		else:
			self.bandgap = ionization_params['bandgap']
			self.density = ionization_params['density']

			self.recombination_rate = 0
			if 'recombination_rate' in ionization_params:
				self.recombination_rate = ionization_params['recombination_rate']

			self.fi_damping_correction = False
			if 'fi_damping_correction' in ionization_params:
				self.fi_damping_correction = ionization_params['fi_damping_correction']

			if ionization_params['rate_equation'].lower() in ['fi']:
				self.rate_equation = 'fi'

			if ionization_params['rate_equation'].lower() in ['sre','single']:
				self.rate_equation = 'sre'
				self.alpha_sre = ionization_params['alpha_sre']

			if ionization_params['rate_equation'].lower() in ['mre','multiple']:
				self.rate_equation = 'mre'
				self.cross_section = ionization_params['cross_section']
				self.trackables += ['el_heating_rate','hl_heating_rate','coll_freq_en','coll_freq_hn','critical_energy']

			if ionization_params['rate_equation'].lower() in ['dre','delayed']:
				self.rate_equation = 'dre'
				self.cross_section = ionization_params['cross_section']
				self.trackables += ['el_heating_rate','hl_heating_rate','coll_freq_en','coll_freq_hn','critical_energy','r_e','r_h','xi_e','xi_h']

			self.mask = 1


	def place_in_domain(self, domain, boundaries):

		self.domain = domain
		self.boundaries = boundaries
		self.mask = bd.ones(self.domain.grid)

		if self.domain.D > 0:

			try: # xmin
				ind, rem = self.parse_index(boundaries['xmin']/self.domain.dx)
				# if material boundary is in pml region, fill all pmls with material
				if ind <= int(self.domain.pml_width/self.domain.dx):
					self.boundaries['xmin'] = self.domain.x.min()
				else:
					self.mask[:ind] = 0
					self.mask[ind] -= rem
			except: self.boundaries['xmin'] = self.domain.x.min()

			try: # xmax
				ind, rem = self.parse_index(boundaries['xmax']/self.domain.dx)
				# if material boundary is in pml region, fill all pmls with material
				if ind >= self.domain.Nx - int(self.domain.pml_width/self.domain.dx): 
					self.boundaries['xmax'] = self.domain.x.max()
				else:
					self.mask[ind+1:] = 0
					self.mask[ind] -= 1-rem
			except: self.boundaries['xmax'] = self.domain.x.max()

			try: # ymin
				ind, rem = self.parse_index(boundaries['ymin']/self.domain.dy)
				self.mask[:,:ind] = 0
				self.mask[:,ind] -= rem
			except: self.boundaries['ymin'] = self.domain.y.min()

			try: # ymax
				ind, rem = self.parse_index(boundaries['ymax']/self.domain.dy)
				self.mask[:,ind+1:] = 0
				self.mask[:,ind] -= 1-rem
			except: self.boundaries['ymax'] = self.domain.y.max()

			try: # zmin
				ind, rem = self.parse_index(boundaries['zmin']/self.domain.dz)
				self.mask[:,:,:ind] = 0
				self.mask[:,:,ind] -= rem
			except: self.boundaries['zmin'] = self.domain.z.min()

			try: # zmax
				ind, rem = self.parse_index(boundaries['zmax']/self.domain.dz)
				self.mask[:,:,ind+1:] = 0
				self.mask[:,:,ind] -= 1-rem
			except: self.boundaries['zmax'] = self.domain.z.max()


		if self.rate_equation == 'mre':

			self.k = self.get_number_mre_levels()
			self.rho_k = bd.zeros(self.k+1) # +1 for the zeroth level
			self.rho_hk = bd.zeros(self.k+1)
			# Place initial population in lower level
			self.rho_k[0] = self.rho
			self.rho_hk[0] = self.rho

			if self.domain.D > 0:
				self.rho_k = self.rho_k[:,None,None,None]*self.mask
				self.rho_hk = self.rho_hk[:,None,None,None]*self.mask

		if self.rate_equation in ['mre','dre']:

			self.Ekin = bd.zeros(self.domain.grid)
			self.Ekin_h = bd.zeros(self.domain.grid)


		self.rho *= self.mask

			



	def make_fi_table(self, laser):
		self.fi_table = fi.fi_table(self, laser)


	
	def field_ionization(self, E):

		# # TODO : optionnal use of a fit instead
		# # FIXME : extremely memory expensive
		# # Nearest interpolation
		# diff_squared = (self.fi_table[:,0][None,None,None,:] - E[...,None])**2
		# ind = bd.argmin(diff_squared, -1)
		# self.fi_rate = self.fi_table[ind, 1]*self.density

		E_flat = bd.flatten(E)
		self.fi_rate = np.interp(bd.numpy(E_flat), bd.numpy(self.fi_table[:,0]), bd.numpy(self.fi_table[:,1]))
		self.fi_rate = bd.reshape(bd.array(self.fi_rate), E.shape)*self.density

		# Saturation
		self.fi_rate *= self.mask*(self.density-self.rho)/self.density

		# Plasma formation
		self.rho += self.domain.dt*self.fi_rate

		# Energy loss of the field = Energy gain in plasma
		self.domain.fields['Jfi'] = self.bandgap*self.fi_rate[...,None]*self.domain.fields['E']/(E[...,None]+1.0)**2.0



	def impact_ionization(self, E):
		tracks = [observer.target for observer in self.domain.observers if observer.target in self.trackables]
		self.ii_rate = ii.ii_rate(E, self, self.domain.laser, self.domain.dt, tracks)
		self.rho += self.domain.dt*self.ii_rate



	def get_number_mre_levels(self):
		# critical_energy = (1.0+self.m_red/self.m_VB) * (self.bandgap + ponderomotive_energy(self.domain.laser.E0, self, self.domain.laser))
		return int(get_critical_energy(self.domain.laser.E0, self, self.domain.laser)/(c.hbar*self.domain.laser.omega) + 1)


	@staticmethod
	def parse_index(float_index):
		return int(float_index), float_index%1


	@property
	def plasma_freq(self):
		return (c.e**2*self.rho/(c.epsilon_0*self.m_red*c.m_e))**.5

	@property
	def _Drude_index(self):
		# torch doesn't support complex numbers yet, so we have to do a detour in numpy
		wp = bd.numpy(self.plasma_freq)
		self.drude_index = np.zeros(wp.shape, dtype=complex)
		self.drude_index = np.sqrt(self.index**2 - wp**2 / (self.domain.laser.omega**2 + 1j*self.domain.laser.omega*self.damping))
		return self.drude_index

	@property
	def Reflectivity(self):
		n = self._Drude_index
		self.reflectivity = np.abs((n-1.)/(n+1.))**2
		return bd.array(self.reflectivity)













# class Material(object):

# 	"""
# 	Object used to define the medium to be irradiated by a laser. 

# 	Arguments:
# 		name (str): The name of the material, only used as an id.

# 		rate_equation (str): The rate equation model to use when calculating plasma 
# 			formation rate. Choose "single" or "sre" to use single rate equation. Choose 
# 			"multiple" or "mre" for multiple rate equations. Choose "delayed" or "dre" for 
# 			the delayed rate equations. Default is "delayed".

# 		index (float): Linear refractive index. Default is 1.

# 		bandgap (float): The bandgap between the conduction and the valence bands in Joules.

# 		m_CB (float): The effective mass of the electrons in the conduction band in units of
# 			their free space mass. Default is 1.

# 		m_VB (float): The effective mass of the holes in the valence band in units of the
# 			free space electron mass. Default is 1.

# 		density (float): The density of atoms of molecules in 1/m^3.

# 		cross_section (float): The cross section for the calculation of the collision rate 
# 			between charge carriers and neutral atoms or molecules in 1/m^2.

# 		damping (float): The damping rate of the plasma in 1/s. Default is 0.

# 		recombination_rate (float): The recombination rate of the charge carriers in 1/s.
# 			Default is 0.
			
# 		alpha_sre (float): The impact rate coefficient used if the rate equation is SRE in
# 			m^2/J. Default is 0.
		
# 	"""

# 	def __init__(self, name="", rate_equation="delayed", index=1, bandgap=0, m_CB=1, m_VB=1, \
# 				 density=0, cross_section=0, damping=0, recombination_rate=0, alpha_sre=0, \
# 				 chi2=0, chi3=0, resonance=0, fi_damping_correction=False):
# 		super(Material, self).__init__()
# 		self.name = name
# 		self.rate_equation = rate_equation
# 		self.index = index
# 		self.chi2 = chi2
# 		self.chi3 = chi3
# 		self.resonance = resonance
# 		self.bandgap = bandgap
# 		self.m_CB = m_CB
# 		self.m_VB = m_VB
# 		if m_CB !=0 and m_VB !=0 :
# 			self.m_red = (self.m_CB**-1.+self.m_VB**-1.)**-1.
# 		else: 
# 			self.m_red = self.m_CB
# 		self.density = density
# 		self.cross_section = cross_section
# 		self.damping = damping
# 		self.recombination_rate = recombination_rate
# 		self.fi_damping_correction = fi_damping_correction

# 		if self.rate_equation.lower() in ["single","sre"]:
# 			self.alpha_sre = alpha_sre
# 		if self.rate_equation.lower() in ["multiple","mre"]:
# 			self.rho_k = []
# 			self.rho_hk = []
# 			self.Ec = 0
# 			self.k = 0
# 		if self.rate_equation.lower() in ["delayed","dre"]:
# 			self.xi = 0	
# 			self.xi_h = 0		

# 		self.rho = 0
# 		self.rho_fi = 0
# 		self.rho_ii = 0
# 		self.Ekin = 0
# 		self.Ekin_h = 0
# 		self.electric_current = 0
# 		self.critical_energy = 0


# 	def __copy__(self):
# 		return Material(name=self.name, rate_equation=self.rate_equation, \
# 						index=self.index, bandgap=self.bandgap, m_CB=self.m_CB, \
# 						m_VB=self.m_VB, density=self.density, \
# 						cross_section=self.cross_section, damping=self.damping, \
# 						recombination_rate=self.recombination_rate, alpha_sre=self.alpha_sre)


# 	def plasma_freq(self):
# 		return abs(c.e**2*self.rho/(c.epsilon_0*self.m_red*c.m_e))**.5

# 	def update_electric_current(self, laser, dt):
# 		self.electric_current = self.electric_current*(1.0-dt*self.damping) \
# 							  + dt*c.epsilon_0*self.plasma_freq()*laser.E

# 	def update_rho(self, laser, dt):
# 		if self.rate_equation.lower() != 'none':
# 			try:
# 				# Much faster to do linear interpolation even if log interpolation should be done instead.
# 				self.rate_fi = np.interp(np.abs(laser.E), self.fi_table[:,0], self.fi_table[:,1])*(self.density-self.rho)/self.density
# 			except:
# 				self.rate_fi = fi.fi_rate(self,laser)*(self.density-self.rho)/self.density
# 			self.rate_ii = ii.ii_rate(self,laser,dt)
# 			self.rate_re = self.recombination_rate*self.rho
# 			self.rho_fi += dt*(self.rate_fi - self.recombination_rate*self.rho_fi)
# 			self.rho_ii += dt*(self.rate_ii - self.recombination_rate*self.rho_ii)
# 			self.rho = self.rho_fi + self.rho_ii

# 	def Drude_index(self, laser):
# 		self.drude_index = cmath.sqrt(self.index**2. - \
# 			self.plasma_freq()**2./(laser.omega**2.+1j*laser.omega*self.damping))
# 		return self.drude_index

# 	def Reflectivity(self, laser):
# 		n = self.Drude_index(laser)
# 		self.reflectivity = abs((n-1.)/(n+1.))**2.
# 		return self.reflectivity

# 	def add_fi_table(self, fi_table):
# 		self.fi_table = fi_table



# def log_interp(zz, xx, yy):
# 	logz = np.log10(zz)
# 	logx = np.log10(xx)
# 	logy = np.log10(yy)
# 	return np.power(10.0, np.interp(logz, logx, logy))



if __name__ == '__main__':
	pass