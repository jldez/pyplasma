#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
import copy


class Domain(object):

	def __init__(self, length, Nx, x0=0, Laser=None, \
				 materials=[{'x_min':0,'x_max':-1,'material':None}], pml_width=0):

		super(Domain, self).__init__()
		self.length = length
		self.Nx = Nx
		self.x0 = x0
		self.Laser = Laser
		self.materials = materials
		self.pml_width = pml_width

		self.construct_domain()

		self.fields = {}
		for field in ['E','H','P','Jb','Jf','Jfi']:
			self.fields[field] = np.zeros(len(self.x))


	def construct_domain(self):
		self.x = np.linspace(self.x0, self.length, self.Nx)
		self.dx = np.abs(self.x[1]-self.x[0])

		self.add_materials()
		self.add_pml()
		self.las_ind = self.get_laser_position_index()
			

	def add_materials(self):
		self.medium = []
		for m in self.materials:
			for i in range(self.Nx):
				if self.x[i] >= m['x_min'] and self.x[i] <= m['x_max']:
					self.medium.append(copy.deepcopy(m['material']))
				else:
					self.medium.append(None)

	def get_laser_position_index(self):
		return np.argmin(np.abs(self.Laser.pos-self.x))


	def add_pml(self):
		# Extend domain
		pml_min = np.arange(self.x0, self.x0-self.pml_width-self.dx, -self.dx)[1:]
		pml_max = np.arange(self.x[-1], self.x[-1]+self.pml_width+self.dx, self.dx)[1:]
		self.nb_pml = len(pml_min)
		self.x = np.hstack([pml_min[::-1],self.x,pml_max])

		# Get refractive index at edges
		if not self.medium[0]:
			n0_pml_min = 1.0
		else:
			n0_pml_min = self.medium[0].index
		if not self.medium[-1]:
			n0_pml_max = 1.0
		else:
			n0_pml_max = self.medium[-1].index

		# Add None material in pml
		self.medium = np.hstack([[None for i in range(self.nb_pml)],self.medium,[None for i in range(self.nb_pml)]])


		m = 3 #grading order for sigma
		self.sigma_pml = np.zeros(len(self.x))

		if self.nb_pml > 0:
			# conductivity in first pml
			sigma_max = (m+1)*8.0/(2*(c.mu_0/(c.epsilon_0*n0_pml_min**2.0))**0.5*self.nb_pml*self.dx)
			self.sigma_pml[0:self.nb_pml] = np.linspace(1,0,self.nb_pml)**m*sigma_max
			# conductivity in second pml
			sigma_max = (m+1)*5.0/(2*(c.mu_0/(c.epsilon_0*n0_pml_max**2.0))**0.5*self.nb_pml*self.dx)
			self.sigma_pml[self.Nx+self.nb_pml:self.Nx+2*self.nb_pml] = np.linspace(0,1,self.nb_pml)**m*sigma_max


	def remove_pml(self):
		if self.nb_pml > 0:
			self.x = self.x[self.nb_pml:-self.nb_pml]
			self.medium = self.medium[self.nb_pml:-self.nb_pml]
			for field in self.fields:
				self.fields[field] = self.fields[field][self.nb_pml:-self.nb_pml]


	def get_rho(self):
		rho = []
		for i in range(len(self.x)):
			try:
				rho.append(self.medium[i].rho)
			except:
				rho.append(0.)
		return np.array(rho)

	def get_rho_fi(self):
		rho_fi = []
		for i in range(len(self.x)):
			try:
				rho_fi.append(self.medium[i].rho_fi)
			except:
				rho_fi.append(0.)
		return np.array(rho_fi)

	def get_resonance(self):
		resonance = []
		for i in range(len(self.x)):
			try:
				resonance.append(self.medium[i].resonance)
			except:
				resonance.append(np.inf)
		return np.array(resonance)

	def get_chis(self):
		chis = []
		for i in range(len(self.x)):
			try:
				chi1 = self.medium[i].index**2 -1.
				chis.append([chi1,self.medium[i].chi2,self.medium[i].chi3])
			except:
				chis.append([0.,0.,0.])
		chis = np.array(chis)
		if self.nb_pml > 0:
			chis[:self.nb_pml,0] = chis[self.nb_pml,0]
			chis[-self.nb_pml:,0] = chis[-self.nb_pml-1,0]
		return chis

	def get_damping(self):
		damping = []
		for i in range(len(self.x)):
			try:
				damping.append(self.medium[i].damping)
			except:
				damping.append(0.)
		return np.array(damping)

	def get_m_red(self):
		m_red = []
		for i in range(len(self.x)):
			try:
				m_red.append(self.medium[i].m_red)
			except:
				m_red.append(1.)
		return np.array(m_red)

	def get_bandgap(self):
		bandgap = []
		for i in range(len(self.x)):
			try:
				bandgap.append(self.medium[i].bandgap)
			except:
				bandgap.append(0.)
		return np.array(bandgap)
