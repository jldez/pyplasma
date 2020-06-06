#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})
import os

from pyplasma import *
from pyplasma.field_ionization import fi_rate
import scipy.constants as c



if __name__ == '__main__':

	f = plt.figure(figsize=(5,4))
	colors = ['0.5','darkred',plt.cm.terrain(np.linspace(0.0, 1.0, 24))[2]]

	laser = Laser(wavelength=800*nm)
	Es = np.logspace(8,14,400)
	bandgaps = [9*eV, 7*eV, 5*eV]

	for color, bandgap in zip(colors, bandgaps):
		mat = Material(index=1.5, drude_params={'damping':0}, ionization_params={'rate_equation':'fi', 'bandgap':bandgap, 'density':2e28})

		fi = np.empty(0)
		for E in Es:
			fi = np.append(fi, fi_rate(E, mat, laser))

		I = c.c*mat.index*c.epsilon_0*Es**2.0/2.0*1e-4

		plt.loglog(I,fi*fs, color=color, lw=2, label=r"%d eV"%int(bandgap/eV))

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