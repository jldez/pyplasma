#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Depreciated
"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
import copy

from . import laser as las
from . import drude as dru
from . import run as run


class Domain(object):

	def __init__(self, length, Nx, x0=0, Laser=None, \
				 materials=[{'x_min':0,'x_max':-1,'material':None}], pml_width=0):

		# TODO : x0 is confusing. Remove?

		super(Domain, self).__init__()
		self.length = length
		self.Nx = Nx
		self.x0 = x0
		self.Laser = Laser
		self.materials = materials
		self.pml_width = pml_width
		self.rho_ik_avalable = True # until proven wrong
		self.k_mre = 0

		self.construct_domain()

		self.fields = {}
		for field in ['E','H','P','Jb','Jf','Jfi']:
			self.fields[field] = np.zeros(len(self.x))

	
	def propagate(self, time, output = ["rho","electric_field"], out_step=1, \
			      remove_pml=True, accelerate_fi=True, source_mode='TFSF', progress_bar=True):
		return run.propagate(time, self, output, out_step, remove_pml, accelerate_fi, source_mode, progress_bar)


	def construct_domain(self):
		self.x = np.linspace(self.x0, self.length, self.Nx)
		self.dx = np.abs(self.x[1]-self.x[0])

		self.add_materials()
		self.add_pml()
		self.las_ind = self.get_laser_position_index()
			

	def add_materials(self):
		self.medium = []
		for i in range(self.Nx):
			self.medium.append(None)
			for m in self.materials:
				if self.x[i] >= m['x_min'] and self.x[i] <= m['x_max']:
					self.medium[-1]=copy.deepcopy(m['material'])
					

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
		self.medium = np.hstack([[self.medium[0] for i in range(self.nb_pml)],self.medium,[self.medium[-1] for i in range(self.nb_pml)]])


		m = 2 #grading order for sigma
		self.sigma_pml = np.zeros(len(self.x))

		if self.nb_pml > 0:
			# conductivity in first pml
			sigma_max = (m+1)*9.0/(2*(c.mu_0/(c.epsilon_0*n0_pml_min**2.0))**0.5*self.nb_pml*self.dx)
			self.sigma_pml[0:self.nb_pml] = np.linspace(1,0,self.nb_pml)**m*sigma_max
			# conductivity in second pml
			sigma_max = (m+1)*9.0/(2*(c.mu_0/(c.epsilon_0*n0_pml_max**2.0))**0.5*self.nb_pml*self.dx)
			self.sigma_pml[self.Nx+self.nb_pml:self.Nx+2*self.nb_pml] = np.linspace(0,1,self.nb_pml)**m*sigma_max


	def remove_pml(self):
		if self.nb_pml > 0:
			self.x = self.x[self.nb_pml:-self.nb_pml]
			self.medium = self.medium[self.nb_pml:-self.nb_pml]
			for field in self.fields:
				self.fields[field] = self.fields[field][self.nb_pml:-self.nb_pml]

	def __len__(self):
		return len(self.x)

	@property
	def rho(self):
		rho = np.zeros(len(self))
		for i in range(len(self)):
			try: rho[i] = self.medium[i].rho
			except: pass
		return rho

	@property
	def rho_fi(self):
		rho_fi = np.zeros(len(self))
		for i in range(len(self)):
			try: rho_fi[i] = self.medium[i].rho_fi
			except: pass
		return rho_fi

	@property
	def rho_ii(self):
		rho_ii = np.zeros(len(self))
		for i in range(len(self)):
			try: rho_ii[i] = self.medium[i].rho_ii
			except: pass
		return rho_ii

	@property
	def rho_k(self):
		if self.rho_ik_avalable and self.k_mre == 0:
			for i in range(len(self)):
				try: self.k_mre = self.medium[i].k
				except: pass
			if self.k_mre == 0:
				self.rho_ik_avalable = False

		if self.rho_ik_avalable and self.k_mre > 0:
			rho_k = np.zeros((len(self), self.k_mre))
			for i in range(len(self)):
				for ik in range(self.k_mre):
					try:
						rho_k[i,ik] = self.medium[i].rho_k[ik]
					except: pass
			return rho_k
		else:
			return np.zeros(len(self))

	@property
	def rate_fi(self):
		rate_fi = np.zeros(len(self))
		for i in range(len(self)):
			try: rate_fi[i] = self.medium[i].rate_fi
			except: pass
		return rate_fi

	@property
	def rate_ii(self):
		rate_ii = np.zeros(len(self))
		for i in range(len(self)):
			try: rate_ii[i] = self.medium[i].rate_ii
			except: pass
		return rate_ii

	@property
	def resonance(self):
		resonance = np.full(len(self), np.inf)
		for i in range(len(self)):
			try: resonance[i] = self.medium[i].resonance
			except: pass
		return resonance

	@property
	def chis(self):
		chis = np.zeros((len(self),3))
		for i in range(len(self)):
			try:
				chi1 = self.medium[i].index**2 -1.
				chis[i,:] = [chi1,self.medium[i].chi2,self.medium[i].chi3]
			except: pass
		return chis

	@property
	def damping(self):
		damping = np.zeros(len(self))
		for i in range(len(self)):
			try: damping[i] = self.medium[i].damping
			except: pass
		return damping

	@property
	def m_red(self):
		m_red = np.ones(len(self))
		for i in range(len(self)):
			try: m_red[i] = self.medium[i].m_red
			except: pass
		return m_red

	@property
	def bandgap(self):
		bandgap = np.zeros(len(self))
		for i in range(len(self)):
			try: bandgap[i] = self.medium[i].bandgap
			except: pass
		return bandgap

	def get_ponderomotive_energy(self, E):
		ponderomotive_energy = []
		for i in range(len(self.x)):
			fake_laser = las.Fake_Laser(E[i],self.Laser.omega,self.Laser.E0)
			try:
				ponderomotive_energy.append(dru.Ep(self.medium[i], fake_laser))
			except:
				ponderomotive_energy.append(0.)
		return np.array(ponderomotive_energy)

	def get_ibh(self, E):
		ibh = []
		for i in range(len(self.x)):
			fake_laser = las.Fake_Laser(E[i],self.Laser.omega,self.Laser.E0)
			try:
				ibh.append(dru.ibh(self.medium[i], fake_laser))
			except:
				ibh.append(0.)
		return np.array(ibh)

	@property
	def kinetic_energy(self):
		kinetic_energy = np.zeros(len(self))
		for i in range(len(self)):
			try:
				kinetic_energy[i] = self.medium[i].Ekin
			except: pass
		return kinetic_energy

	@property
	def critical_energy(self):
		critical_energy = np.zeros(len(self))
		for i in range(len(self)):
			try:
				critical_energy[i] = self.medium[i].critical_energy
			except: pass
		return critical_energy

