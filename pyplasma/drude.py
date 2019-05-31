#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c



# Cycle-averaged ponderomotive energy
def Ep(material, laser):
	Ep = c.e**2.0*laser.E**2.0/(4.0*material.m_red*(material.damping**2.0+laser.omega**2.0))
	return Ep

# Cycle-averaged inverse bremsstrahlung absorption rate
def ibh(material, laser, s="eh"):
	ibh = c.e**2*material.damping*laser.E**2./(2.*c.hbar*laser.omega*(material.damping**2.+laser.omega**2.))
	if s == "eh":
		return ibh/material.m_red
	if s == "e":
		return ibh/material.m_CB
	if s == "h":
		return ibh/material.m_VB



if __name__ == '__main__':
	pass