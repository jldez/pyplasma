#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c



class Laser(object):
	"""
	Define the laser properties and state.

	Arguments:
		wavelength (float): The wavelength of the laser in m.

		omega (float): The angular frequency of the laser in 1/s. Either the wavelength
			or omega can be specified and the other will be automatically calculated.

		phase (bool): If True, the instantaneous electric field is multiplied by the
			function cos(omega*time) to account for sub-cycle effects. If False,
			only the envelope (cycle-averaged field) of the laser is accounted for.
			Default is False.

		pulse_duration (float): The pulse duration in seconds is the FWHM of the 
			gaussian laser pulse. Default is np.inf, which means that the laser is
			continuous.

		fluence (float): The fluence of the laser pulse in J/m^2.

		E0 (float): The amplitude of the electric field in V/m. Either the fluence
			or E0 has to be specified and the other is automatically calculated.

		t0 (float): The time at which the laser is initialized in seconds. Default is 0.

		transmit (bool): If True, the amplitude of the electric field is multiplied
			by its transmisibility in a material. In that case, when calling the method
			update_Electric_field(), the material has to be an argument. Default is False.
	"""

	def __init__(self, wavelength=0, omega=0, phase=False, pulse_duration=np.inf, \
				 fluence=0, E0=0, t0=0, transmit=False, pos=0):
		super(Laser, self).__init__()

		self.wavelength = wavelength
		self.omega = omega
		if omega == 0 and wavelength != 0:
			self.omega = 2.*c.pi*c.c/wavelength
		elif wavelength == 0 and omega != 0:
			self.wavelength = 2.*c.pi*c.c/omega
		else:
			print("Error. You have to specify wavelength or omega.")

		self.phase = phase
		self.pulse_duration = pulse_duration
		self.transmit = transmit
		self.pos = pos

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

		self.time = t0
		self.update_Electric_field()


	def update_Electric_field(self, material=0):
		"""
		Calculate the instantaneous electric field value at the current time.

		Arguments:
			material (Material object): If Laser.transmit = True, the material
				has to be specified in the arguments. 
		"""

		self.E = self.E0
		# Gaussian pulse shape
		self.E *= np.exp(-2.*np.log(2.)*(self.time/self.pulse_duration)**2.)
		# Instantaneous phase
		if self.phase:
			self.E *= np.cos(self.omega*self.time)
		# Transmition
		if self.transmit and material != 0:
			self.E *= ((1.-material.Reflectivity(self))/material.index)**0.5


	def time_step(self, dt):
		self.time += dt
		self.update_Electric_field()




class Fake_Laser(object):
	def __init__(self, E, omega, E0=0):
		self.E = E
		self.omega = omega
		self.E0 = E0


if __name__ == '__main__':
	pass