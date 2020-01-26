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
from . import laser as las
from . import field_ionization as fi



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


	To do : - rho_fi should output the total ionization from fi
			- add fi_rate to replace current rho_fi
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






def propagate(Time, Domain, output = ["rho","electric_field"], out_step=1, \
			  remove_pml=True, accelerate_fi=True, source_mode='TFSF', progress_bar=True):

	# TFSF implementation might not be perfect. See references:
	# Section 3.0 of : https://studylib.net/doc/8392930/6.-total-field---scattered-field-fdtd-implementation-in-m...
	# Section 3.10 of : https://www.eecs.wsu.edu/~schneidj/ufdtd/chap3.pdf

	out_data = {}
	for obj in output:
		out_data[obj] = []

	dt = abs(Time[1]-Time[0])
	chis = Domain.get_chis()
	print('Stability condition : dt = {:f} * dt_max'.format(dt/Domain.dx*c.c*(chis[:,0].max()+1)**0.5))

	if progress_bar:
		Time = tqdm.tqdm(Time)

	resonance = Domain.get_resonance()
	damping = Domain.get_damping()
	m_red = Domain.get_m_red()
	bandgap = Domain.get_bandgap()

	if accelerate_fi:
		for m in range(len(Domain.materials)):
			if Domain.materials[m]['material'] != None:
				fi_table = fi.fi_table(Domain.materials[m]['material'], Domain.Laser, \
								N=5e3, tol=1e-3, output=None, progress_bar=False)
			for i in range(len(Domain.x)):
				if Domain.x[i]>=Domain.materials[m]['x_min'] and Domain.x[i]<=Domain.materials[m]['x_max']:
					if Domain.medium[i] != None:
						Domain.medium[i].add_fi_table(fi_table)


	n=0
	for t in Time:

		# Update laser source
		Domain.Laser.time = t
		Domain.Laser.update_Electric_field(Domain.medium[Domain.las_ind])

		# Update ionization state
		for i in range(len(Domain.x)):
			try:
				fake_laser = las.Fake_Laser(Domain.fields['E'][i],Domain.Laser.omega,Domain.Laser.E0)
				Domain.medium[i].update_rho(fake_laser, dt)
			except:
				pass
		rho = Domain.get_rho()
		rho_fi = Domain.get_rho_fi()

		# Update mpi interband currents
		Domain.fields['Jfi'] = bandgap*rho_fi*Domain.fields['E']/(Domain.fields['E']+1.0)**2.0

		# Update bounded currents
		w0 = 2*c.pi*c.c/resonance
		P = c.epsilon_0*(chis[:,0]*Domain.fields['E']+chis[:,1]*Domain.fields['E']**2+chis[:,2]*Domain.fields['E']**3)
		Domain.fields['Jb'] = Domain.fields['Jb'] + dt*w0**2*(P - Domain.fields['P'])
		Domain.fields['P'] += dt*Domain.fields['Jb']

		# Update free currents
		G = damping*dt/2
		omega_p = abs(c.e**2*rho/(c.epsilon_0*m_red))**.5
		Domain.fields['Jf'] = Domain.fields['Jf']*(1-G)/(1+G) + dt*c.epsilon_0*omega_p**2.0*Domain.fields['E']/(1+G)

		

		# Coefficients for FDTD scheme
		A = (c.epsilon_0 - dt*Domain.sigma_pml/2)/(c.epsilon_0 + dt/2*Domain.sigma_pml)
		B = dt/Domain.dx/(c.epsilon_0 + dt*Domain.sigma_pml/2)*c.epsilon_0/c.mu_0
		C = c.mu_0/c.epsilon_0*B

		# Update magnetic field
		Domain.fields['H'][:-1] = A[:-1]*Domain.fields['H'][:-1]\
								 -B[:-1]*(Domain.fields['E'][1:]-Domain.fields['E'][:-1])

		# Magnetic field correction for TFSF laser source
		if source_mode == 'TFSF':
			Domain.fields['H'][Domain.las_ind] += dt/(c.mu_0*Domain.dx)*Domain.Laser.E

		# Update electric field
		Domain.fields['E'][1:-1] = A[1:-1]*Domain.fields['E'][1:-1]\
								  -C[1:-1]*(Domain.fields['H'][1:-1]-Domain.fields['H'][:-2])\
								  -dt*(Domain.fields['Jb'][1:-1])/c.epsilon_0\
								  -dt*(Domain.fields['Jf'][1:-1])/c.epsilon_0\
								  -dt*(Domain.fields['Jfi'][1:-1])/c.epsilon_0

		# Electric field correction for TFSF laser source
		if source_mode == 'TFSF':
			Domain.fields['E'][Domain.las_ind] += dt/(c.epsilon_0*Domain.dx)*Domain.Laser.E/(120*c.pi)

		# Hard source update
		if source_mode == 'hard':
			Domain.fields['E'][Domain.las_ind] = Domain.Laser.E

		if n%out_step == 0:
			# Output data
			if 'rho' in output:
				out_data["rho"].append(rho)
			if "electric_field" in output:
				out_data["electric_field"].append(copy.copy(Domain.fields['E']))
			if "magnetic_flux" in output:
				out_data["magnetic_flux"].append(copy.copy(Domain.fields['H']))
			if "bounded_current" in output:
				out_data["bounded_current"].append(copy.copy(Domain.fields['Jb']))
			if "free_current" in output:
				out_data["free_current"].append(copy.copy(Domain.fields['Jf']))
			if "fi_current" in output:
				out_data["fi_current"].append(copy.copy(Domain.fields['Jfi']))
			if 'ponderomotive_energy' in output:
				out_data["ponderomotive_energy"].append(copy.copy(Domain.get_ponderomotive_energy(Domain.fields['E'])))
			if 'ibh' in output:
				out_data["ibh"].append(copy.copy(Domain.get_ibh(Domain.fields['E'])))
			if 'kinetic_energy' in output:
				out_data['kinetic_energy'].append(copy.copy(Domain.get_kinetic_energy()))

		n+=1

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

