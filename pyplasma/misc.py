#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c


# electron-molecule collision rate
def g_en(material):
	return material.cross_section*(material.density-material.rho)*(2.0*abs(material.Ekin)/material.m_CB)**0.5

# hole-molecule collision rate
def g_hn(material):
	return material.cross_section*(material.density-material.rho)*(2.0*abs(material.Ekin_h)/material.m_VB)**0.5

# electron-electron collision rate
def g_ee(material):
	return 4.*c.pi*c.epsilon_0/c.e**2.*(6./material.m_CB)**.5*(2.*material.Ekin/3.)**1.5



if __name__ == '__main__':
	pass