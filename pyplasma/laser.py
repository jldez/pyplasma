"""

"""
import numpy as np
import scipy.constants as c



class Laser():

	"""
	Defines the laser parameters.

	Arguments:

		wavelength (float): The wavelength in meters. Have to be specified if omega is not.

		omega (float): The angular frequency in rad/s. Have to be specified if wavelength is not.

		phase (bool): If True, the laser oscillates as normal. If False, oscillations are turned off and
					  only the amplitude enveloppe is accounted for. Default is True.

		pulse_duration (float): The full width at half maximum (FWHM) in seconds of the gaussian
								enveloppe of the pulse. Default is infinite, for a continuous laser.

		fluence (float): The fluence is the total energy per surface area of the laser pulse in J/m^2.
		                 It makes sense only if the pulse_duration is finite. Have to be specified if
						 E0 is not.

		E0 (float): The amplitude of the electric field in V/m. Have to be specified if fluence is not.

		t0 (float): Time shift in seconds. Default is 0. 


	Note: Use laser.E(t) to get the electric field in V/m at time t in seconds.

	"""

	# FIXME: phase is confusing. Should be named something else to represent that it activates the oscillations.
	#        Then, phase should add a phase.
	# TODO: add an exponential ramp to 'turn on' the electric field less abruptly in simulations.

	def __init__(self, wavelength=0, omega=0, phase=True, pulse_duration=np.inf, fluence=0, E0=0, t0=0):

		self.wavelength = wavelength
		self.omega = omega
		if omega == 0 and wavelength != 0:
			self.omega = 2.*c.pi*c.c/wavelength
		elif wavelength == 0 and omega != 0:
			self.wavelength = 2.*c.pi*c.c/omega
		else:
			raise ValueError('You have to specify either wavelength or omega.')

		self.phase = phase
		self.pulse_duration = pulse_duration
		self.remove_reflected_part = False
		self.ramp = False

		self.E0 = E0
		self.fluence = fluence
		if fluence == 0:
			self.fluence = c.c*c.epsilon_0*E0**2*pulse_duration/2.*(c.pi/np.log(2.))**.5
			if phase:
				self.fluence *= np.sqrt(2.)
		elif E0 == 0:
			self.E0 = (2.*fluence/(c.c*c.epsilon_0*pulse_duration)*(np.log(2.)/c.pi)**.5)**.5
			if phase:
				self.E0 /= np.sqrt(2.)

		self.t0 = t0
		

	def E(self, t):
		""" Electric field at time t """

		E = self.E0

		# Gaussian pulse shape
		E *= np.exp(-2.*np.log(2.)*((t-self.t0)/self.pulse_duration)**2.)

		# Instantaneous phase
		if self.phase:
			E *= np.cos(self.omega*(t-self.t0))

		# Transmition
		if self.remove_reflected_part and self.domain.D == 0:
			material = self.domain.materials[0]
			E *= ((1.-material.Reflectivity)/material._Drude_index.real)**0.5

		return E

	
	@property
	def index_in_domain(self):
		""" The index at which to apply the electric field source. Propagation only along x axis supported. """
		return int((self.position - self.domain.x.min())/self.domain.dx)

