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
