#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})
import scipy.constants as c
import os

import pyplasma.material as mat
import pyplasma.laser as las
import pyplasma.field_ionization as fi




if __name__ == '__main__':

	Eg1 = mat.Material(index=1.5, bandgap=9.*c.e, m_CB=1, m_VB=1, density=2e28)
	Eg2 = mat.Material(index=1.5, bandgap=7.*c.e, m_CB=1, m_VB=1, density=2e28)
	Eg3 = mat.Material(index=1.5, bandgap=5.*c.e, m_CB=1, m_VB=1, density=2e28)
	laser = las.Laser(wavelength=800e-9)

	N = 1000
	Es = np.logspace(8,14,N)

	fi_Eg1, fi_Eg2, fi_Eg3 = [], [], []
	for E in Es:
		laser.E = E
		fi_Eg1.append(fi.fi_rate(Eg1,laser)/Eg1.density)
		fi_Eg2.append(fi.fi_rate(Eg2,laser)/Eg2.density)
		fi_Eg3.append(fi.fi_rate(Eg3,laser)/Eg3.density)
	fi_Eg1 = np.array(fi_Eg1)
	fi_Eg2 = np.array(fi_Eg2)
	fi_Eg3 = np.array(fi_Eg3)


	f = plt.figure(figsize=(5,4))
	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))

	I = c.c*Eg1.index*c.epsilon_0*Es**2.0/2.0*1e-4

	ax = plt.loglog(I,fi_Eg1*1e-15,color="0.5",lw=2,label=r"%d eV"%9)
	plt.loglog(I,fi_Eg2*1e-15,color="darkred",lw=2,label=r"%d eV"%7)
	plt.loglog(I,fi_Eg3*1e-15,color=colors[2],lw=2,label=r"%d eV"%5)

	plt.xlabel(r"$\mathrm{Laser\,\, intensity}~[\mathrm{W/cm}^2]$")
	plt.xlim(1e9,5e16)

	plt.ylabel(r"$\mathrm{\nu_\mathrm{fi}~[fs}^{-1}]$")
	plt.gca().yaxis.set_label_coords(-0.1,0.87)

	ax=f.axes[0]
	ax.legend(title=r"$\mathcal{E}_g$",loc=(0.07,0.62),frameon=False,labelspacing=0.4,borderaxespad=0,handletextpad=0.4)

	plt.tight_layout()
	plt.subplots_adjust(left=0.15,right=0.99,top=0.987,bottom=0.15)
	plt.savefig("fi.pdf")
	os.system("pdfcrop fi.pdf fi.pdf > /dev/null")
	
	# plt.show()