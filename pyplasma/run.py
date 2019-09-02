#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
import tqdm
import copy

from . import misc as mis
from . import drude as dru



def run(Time, Material, Laser, output = ["rho","electric_field"], progress_bar=True):
	"""
	This function run an entire simulation of plasma formation in a 
		dielectric irradiated by a laser.

		Arguments:
			Time (numpy array): 1D array containing all discrete times for the simulation.
				Exemple: Time=np.linspace(start,end,N) where N is the number of time steps.

			Material (Material object): The material in which the plasma formation
				takes place.

			Laser (Laser object): The laser that causes the plasma formation.

			output (list): A list of strings. The strings indicate what data to output.
				-"rho": plasma density
				-"rho_fi": field ionization rate
				-"rho_ii": impact ionization rate
				-"xi": If the impact ionization model is DRE, xi is the fraction
					of the electrons E_kin > E_c.
				-"xi_h": Same as "xi", for holes.
				-"Ekin": Mean kinetic energy of the electrons.
				-"Ekin_h": Mean kinetic energy of the holes.
				-"collision_freq_en": Electron-molecule collision frequency.
				-"collision_freq_hn": Hole-molecule collision frequency.
				-"collision_freq_ee": Electron-electron collision frequency.
				-"ibh": Inverse Bremmstrahlung heating rate.
				-"electric_field": Electric field of the laser.
				-"Reflectivity": Reflectivity of the material.
				Default is ["rho","electric_field"].

			progress_bar (bool): If True, a progress bar will be displayed 
				when running the simulation. Default is True.

		Returns:
			(dict): A dictionnary that contains all requested data. 
				The keys correspond to the strings in the output argument.
	"""

	out_data = {}
	for obj in output:
		out_data[obj] = []

	dt = abs(Time[1]-Time[0])

	if progress_bar:
		Time = tqdm.tqdm(Time)

	for t in Time:
		Laser.time = t
		Laser.update_Electric_field(Material)
		Material.update_rho(Laser,dt)

		if "rho" in output:
			out_data["rho"].append(Material.rho)
		if "rho_fi" in output:
			out_data["rho_fi"].append(dt*Material.rho_fi)
		if "rho_ii" in output:
			out_data["rho_ii"].append(dt*Material.rho_ii)
		if "xi" in output and Material.rate_equation.lower() in ["delayed","dre"]:
			out_data["xi"].append(Material.xi)
		if "xi_h" in output and Material.rate_equation.lower() in ["delayed","dre"]:
			out_data["xi_h"].append(Material.xi_h)
		if "Ekin" in output:
			out_data["Ekin"].append(Material.Ekin)
		if "Ekin_h" in output:
			out_data["Ekin_h"].append(Material.Ekin_h)
		if "collision_freq_en" in output:
			out_data["collision_freq_en"].append(mis.g_en(Material))
		if "collision_freq_hn" in output:
			out_data["collision_freq_hn"].append(mis.g_hn(Material))
		if "collision_freq_ee" in output:
			out_data["collision_freq_ee"].append(mis.g_ee(Material))
		if "ibh" in output:
			out_data["ibh"].append(dru.ibh(Material,Laser,s="eh"))
		if "electric_field" in output:
			out_data["electric_field"].append(Laser.E)
		if "Reflectivity" in output:
			out_data["Reflectivity"].append(Material.Reflectivity(Laser))

	for obj in output:
		out_data[obj] = np.array(out_data[obj])

	return out_data




def propagate(Time, Domain, output = ["rho","electric_field"], remove_pml=True, progress_bar=True):

	# Todo : generate keldysh rate table at beginning abd then use it.

	out_data = {}
	for obj in output:
		out_data[obj] = []

	dt = abs(Time[1]-Time[0])
	# print(dt/Domain.dx*c.c)

	if progress_bar:
		Time = tqdm.tqdm(Time)


	w0, chi1, chi2, chi3 = [],[],[],[]
	for i in range(len(Domain.x)):
		try:
			w0.append(2*c.pi*c.c/Domain.medium[i].resonance)
			chi1.append(Domain.medium[i].index**2 -1.)
			chi2.append(Domain.medium[i].chi2)
			chi3.append(Domain.medium[i].chi3)
		except:
			w0.append(0.)
			chi1.append(0.)
			chi2.append(0.)
			chi3.append(0.)

	for t in Time:
	
		# Update ionization state
		for i in range(len(Domain.x)):
			try:
				fake_laser = Fake_Laser(Domain.fields['E'][i],Domain.lasers[0]['laser'].omega,Domain.lasers[0]['laser'].E0)
				Domain.medium[i].update_rho(fake_laser, dt)
			except:
				continue
			

		# Update bounded currents
		P = c.epsilon_0*(chi1*Domain.fields['E']+chi2*Domain.fields['E']**2+chi3*Domain.fields['E']**3)
		Domain.fields['Jb'] = Domain.fields['Jb'] + dt*np.array(w0)**2*(P - Domain.fields['P'])
		Domain.fields['P'] += dt*Domain.fields['Jb']

	


		# Propagate
		A = (c.epsilon_0 - dt*Domain.sigma_pml/2)/(c.epsilon_0 + dt/2*Domain.sigma_pml)
		B = dt/Domain.dx/(c.epsilon_0 + dt*Domain.sigma_pml/2)*c.epsilon_0/c.mu_0
		C = c.mu_0/c.epsilon_0*B

		Domain.fields['H'][:-1] = A[:-1]*Domain.fields['H'][:-1]\
								 -B[:-1]*(Domain.fields['E'][1:]-Domain.fields['E'][:-1])
		Domain.fields['E'][1:-1] = A[1:-1]*Domain.fields['E'][1:-1]\
								  -C[1:-1]*(Domain.fields['H'][1:-1]-Domain.fields['H'][:-2])\
								  -dt*(Domain.fields['Jb'][1:-1])/c.epsilon_0





		# Update laser sources
		for las in Domain.lasers:
			las['laser'].time = t
			ind_las = np.argmin(np.abs(las['x']-Domain.x))
			las['laser'].update_Electric_field(Domain.medium[ind_las])
			Domain.fields['E'][ind_las] += las['laser'].E #todo : make sure to add source on top of E

		# Output data
		if 'rho' in output:
			rho = []
			for i in range(len(Domain.x)):
				try:
					rho.append(Domain.medium[i].rho)
				except:
					rho.append(0.)
			out_data["rho"].append(rho)
		if "electric_field" in output:
			out_data["electric_field"].append(copy.copy(Domain.fields['E']))
		if "bounded_current" in output:
			out_data["bounded_current"].append(copy.copy(Domain.fields['Jb']))

	if remove_pml:
		Domain.remove_pml()
	for obj in out_data:
		out_data[obj] = np.array(out_data[obj])
		if remove_pml:
			if Domain.nb_pml > 0:
				try:
					out_data[obj] = out_data[obj][:,Domain.nb_pml:-Domain.nb_pml]
				except:
					pass

	return out_data


class Fake_Laser(object):
	def __init__(self, E, omega, E0=0):
		self.E = E
		self.omega = omega
		self.E0 = E0