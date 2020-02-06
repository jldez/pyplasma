"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c



def get_el_heating_rate(E, material, laser):
	A = c.e**2*material.damping*E**2/(2*c.hbar*laser.omega*(material.damping**2+laser.omega**2))
	return A/(material.m_CB*c.m_e)

def get_hl_heating_rate(E, material, laser):
	A = c.e**2*material.damping*E**2/(2*c.hbar*laser.omega*(material.damping**2+laser.omega**2))
	return A/(material.m_VB*c.m_e)

def ponderomotive_energy(E, material, laser):
	return c.e**2*E**2 / (4*material.m_red*c.m_e * (material.damping**2 + laser.omega**2))

def get_critical_energy(E, material, laser):
	return (1+material.m_red/material.m_VB) * (material.bandgap + ponderomotive_energy(E, material, laser))

def ee_coll_freq(Ekin, material):
	""" electron-electron collision rate """
	return 4*c.pi*c.epsilon_0/c.e**2*(6/material.m_CB/c.m_e)**0.5*(2*Ekin/3)**1.5

def el_Ekin_max(E, material, laser):
	""" Upper bound estimation function for the mean kinetic energy of the electrons. 
	
		Arguments:

			E: Electric field of the laser during the simulation.

			material (Material object): The material in which the plasma formation
				takes place.

			laser (Laser object): The laser that causes the plasma formation.


		Returns:
			Upper bound of the mean kinetic energy in Joules (float).
	"""

	Ec = get_critical_energy(0, material, laser) # For some reason, we have to use Ec at rest?
	el_heating_rate = get_el_heating_rate(E, material, laser)

	return (-1.5*Ec/(np.log(el_heating_rate*c.hbar*laser.omega\
		/(2.*Ec*material.cross_section*material.density)*(material.m_CB*c.m_e*c.pi/(3.*Ec))**.5))).max()

