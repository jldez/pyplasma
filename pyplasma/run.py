#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
import tqdm

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
