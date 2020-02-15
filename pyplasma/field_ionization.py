"""

"""
import numpy as np
import scipy.constants as c
from scipy.special import ellipk, ellipe, dawsn
import tqdm

from . import laser as las
from .backend import backend as bd



# Keldysh ionisation
def fi_rate(E, material, laser, tol=1e-3):
	"""
	Calculate the field ionization rate according to Keldysh's formula.

		Arguments:
			E (float) : Amplitude of the electric field.

			material (Material object): The material in which the plasma formation
				takes place.

			laser (Laser object): The laser that causes the plasma formation.

		Returns:
			fi_rate (float): The field ionization rate in 1/s. 
	"""

	E = abs(E)

	if (E<1e3) or material.rate_equation == 'none':
		return 0.0

	def CEI1(a):
		return ellipk(a)
	def CEI2(a):
		return ellipe(a)

	#Dawson integral
	def DI(a):
		return dawsn(a)

	#Keldysh parameter
	def g(E):
		return laser.omega*(material.m_red*c.m_e*material.bandgap)**0.5/(c.e*E)

	#dummy parameters
	def g1(E):
		return g(E)**2.0/(1.0+g(E)**2.0)
	def g2(E):
		return 1.0/(1.0+g(E)**2.0)
	def g3(E):
		stark_factor = 2.0/c.pi*CEI2(g2(E))/(g1(E))**0.5
		if not material.fi_damping_correction:
			return stark_factor*material.bandgap/(c.hbar*laser.omega)
		return material.bandgap/(c.hbar*laser.omega)*(stark_factor - 1)*laser.omega**2/(material.damping**2+laser.omega**2) \
				+ material.bandgap/(c.hbar*laser.omega)

	#infinite sum
	def Q(E):
		sol,err,n = 0.0,1e10,0
		c1=(CEI1(g1(E))-CEI2(g1(E)))/CEI2(g2(E))
		c2=2.0*CEI1(g2(E))*CEI2(g2(E))
		while (err > tol):
			term = np.exp(-n*c.pi*c1)*DI(c.pi*((np.floor(g3(E)+1.0)-g3(E)+n)/c2)**0.5)
			sol = sol + term
			n = n+1
			err = term/(sol+1e-16)
		return sol*(c.pi/(2.0*CEI1(g2(E))))**0.5

	try:
		return 4.0*laser.omega/(9.0*c.pi)*(material.m_red*c.m_e*laser.omega/(c.hbar*g1(E)**0.5))**1.5*Q(E) \
			*np.exp(-c.pi*np.floor(g3(E)+1)*(CEI1(g1(E))-CEI2(g1(E)))/CEI2(g2(E))) /material.density
	except:
		return 0.0



def fi_table(material, laser, N=1000, tol=1e-4):

	Es = np.logspace(3,np.log10(3*laser.E0),int(N))
	table = bd.zeros((N,2))

	for i, E in enumerate(Es):
		rate = fi_rate(E, material, laser, tol=tol)
		table[i,0] = E
		table[i,1] = rate

	return table



if __name__ == '__main__':
	pass