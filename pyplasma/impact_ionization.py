"""

"""
import copy
import scipy.constants as c

from .misc import *
from .backend import backend as bd




def ii_rate(E, material, laser, dt, tracks=[]):
	"""
	Calculate impact ionization rate according to the rate equation model chosen for the material.

		Arguments:
			E (float) : Amplitude of the electric field.

			material (Material object): The material in which the plasma formation
				takes place.

			laser (Laser object): The laser that causes the plasma formation.

			dt (float): The time step of the simulation.

			tracks (list): The names of variables or a arrays to add to the material's attributes.

		Returns:
			ii_rate (float): The impact ionization rate in 1/(sm^3). 
			Divide by material's density to obtain the rate in 1/s.

	"""


	if material.rate_equation.lower() in ["single","sre"]:
		return sre(E, material, laser)
	if material.rate_equation.lower() in ["multiple","mre"]:
		return mre(E, material, laser, dt, tracks)
	if material.rate_equation.lower() in ["delayed","dre"]:
		return dre(E, material, laser, dt, tracks)


def sre(E, material, laser):
	""" Single rate equation model """

	intensity = c.c*material.index*c.epsilon_0*E**2./2.
	ii_rate = material.alpha_sre*material.rho*intensity

	# Saturation FIXME :should be done simultaneously for both FI and II
	ii_rate *= (material.density-material.rho)/material.density

	return ii_rate



def mre(E, material, laser, dt, tracks=[]):
	""" Multiple rate equation model """

	material.Ekin *= 0
	material.Ekin_h *= 0
	for ik in range(material.k+1)[1:]:
		material.Ekin += ik*c.hbar*laser.omega*material.rho_k[ik]/(material.rho + 1e-16)
		material.Ekin_h += ik*c.hbar*laser.omega*material.rho_hk[ik]/(material.rho + 1e-16)
	
	A = c.e**2*material.damping*E**2/(2*c.hbar*laser.omega*(material.damping**2+laser.omega**2))
	el_heating_rate = A/(material.m_CB*c.m_e)
	hl_heating_rate = A/(material.m_VB*c.m_e)

	B = material.cross_section*(material.density-material.rho)
	coll_freq_en = B*(2*material.Ekin/material.m_CB/c.m_e)**0.5
	coll_freq_hn = B*(2*material.Ekin_h/material.m_VB/c.m_e)**0.5

	rho_k_copy = copy.deepcopy(material.rho_k)
	rho_hk_copy = copy.deepcopy(material.rho_hk)

	# Electron cascade
	material.rho_k[0] += dt*(material.fi_rate - el_heating_rate*rho_k_copy[0] + 2*coll_freq_en*rho_k_copy[-1] \
		+ coll_freq_hn*rho_hk_copy[-1] - material.recombination_rate*rho_k_copy[0])
	for ik in range(material.k)[1:]:
		material.rho_k[ik] += dt*(el_heating_rate*(rho_k_copy[ik-1]-rho_k_copy[ik]) \
			- material.recombination_rate*rho_k_copy[ik])
	material.rho_k[-1] += dt*(el_heating_rate*rho_k_copy[-2] \
		- coll_freq_en*rho_k_copy[-1] - material.recombination_rate*rho_k_copy[-1])

	# Holes
	material.rho_hk[0] += dt*(material.fi_rate - hl_heating_rate*rho_hk_copy[0] + 2.*coll_freq_hn*rho_hk_copy[-1] \
		+ coll_freq_en*rho_k_copy[-1] - material.recombination_rate*rho_hk_copy[0])
	for ik in range(material.k)[1:]:
		material.rho_hk[ik] += dt*(hl_heating_rate*(rho_hk_copy[ik-1]-rho_hk_copy[ik]) \
			- material.recombination_rate*rho_hk_copy[ik])
	material.rho_hk[-1] += dt*(hl_heating_rate*rho_hk_copy[-2] \
		- coll_freq_hn*rho_hk_copy[-1] - material.recombination_rate*rho_hk_copy[-1])

	for track in tracks:
		track_array = copy.deepcopy(eval(track))
		setattr(material, track, track_array)

	return coll_freq_en*material.rho_k[material.k] + coll_freq_hn*material.rho_hk[material.k]


def dre(E, material, laser, dt, tracks=[]):
	""" Delayed rate equation model """

	el_heating_rate = get_el_heating_rate(E, material, laser)
	hl_heating_rate = get_hl_heating_rate(E, material, laser)

	B = material.cross_section*(material.density-material.rho)
	coll_freq_en = B*bd.abs(2*material.Ekin/material.m_CB/c.m_e)**0.5
	coll_freq_hn = B*bd.abs(2*material.Ekin_h/material.m_VB/c.m_e)**0.5

	critical_energy = get_critical_energy(E, material, laser)

	r_e = bd.abs(3*critical_energy/(2*material.Ekin+1e-100))**0.5
	xi_e = xi(r_e)
	r_h = bd.abs(3*critical_energy/(2*material.Ekin_h+1e-100))**0.5
	xi_h = xi(r_h)

	material.Ekin += dt*(el_heating_rate*c.hbar*laser.omega - coll_freq_en*xi_e*critical_energy - \
		material.Ekin*(material.fi_rate/(material.rho+1e-10) + coll_freq_en*xi_e + coll_freq_hn*xi_h)) 
	material.Ekin_h += dt*(hl_heating_rate*c.hbar*laser.omega - coll_freq_hn*xi_h*critical_energy - \
		material.Ekin_h*(material.fi_rate/(material.rho+1e-10) + coll_freq_en*xi_e + coll_freq_hn*xi_h)) 

	for track in tracks:
		track_array = copy.deepcopy(eval(track))
		setattr(material, track, track_array)

	return material.rho*(coll_freq_en*xi_e + coll_freq_hn*xi_h)


def xi1(r):
	return bd.erfc(r)
def xi2(r):
	return 2*r/c.pi**0.5*bd.exp(-r**2)
def xi(r):
	return xi1(r) + xi2(r)
