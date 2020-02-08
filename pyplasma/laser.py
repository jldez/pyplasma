"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c



class Laser():

	# FIXME: phase should be True by default
	# TODO: Documentation

	def __init__(self, wavelength=0, omega=0, phase=False, pulse_duration=np.inf, fluence=0, E0=0, t0=0):

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

		E = self.E0

		# Gaussian pulse shape
		E *= np.exp(-2.*np.log(2.)*((t-self.t0)/self.pulse_duration)**2.)

		# Instantaneous phase
		if self.phase:
			E *= np.cos(self.omega*(t-self.t0))

		# Transmition
		if self.remove_reflected_part and self.domain.D == 0:
			material = self.domain.materials[0]
			E *= ((1.-material.Reflectivity)/material.index)**0.5

		return E

	
	@property
	def index_in_domain(self):
		return int((self.position - self.domain.x.min())/self.domain.dx)

