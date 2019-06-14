#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c



def Ep(material, laser):
	""" Cycle-averaged ponderomotive energy """
	Ep = c.e**2.0*laser.E**2.0/(4.0*material.m_red*(material.damping**2.0+laser.omega**2.0))
	return Ep


def ibh(material, laser, s="eh"):
	""" 
	Cycle-averaged inverse bremsstrahlung absorption rate 

		Arguments:
			material (Material object): The material in which the plasma formation
				takes place.

			laser (Laser object): The laser that causes the plasma formation.

			s (str): Indicates if the inverse bremsstrahlung is calculated for
				both electrons and holes (s="eh"), the electrons only (s="e")
				or the holes only (s="h"). Default is s="eh".

		Returns:
			Inverse bremsstrahlung rate in 1/s (float).

	"""
	ibh = c.e**2*material.damping*laser.E**2./(2.*c.hbar*laser.omega*(material.damping**2.+laser.omega**2.))
	if s == "eh":
		return ibh/material.m_red
	if s == "e":
		return ibh/material.m_CB
	if s == "h":
		return ibh/material.m_VB



if __name__ == '__main__':
	pass