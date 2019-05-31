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
import os
from pyplasma import impact_ionization as ii

if __name__ == '__main__':

	r = np.linspace(0,3,1000)
	xi1 = ii.xi1(r)
	xi2 = ii.xi2(r)

	f = plt.figure(figsize=(6,3))
	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))

	plt.semilogy(r,xi1,lw=2,c=colors[2],ls="-",label=r"$\mathrm{erfc}(r_s)$")
	plt.semilogy(r,xi2,lw=2,c="0.5",ls="-",label=r"$2r_s e^{-r_s^2}/\sqrt{\pi}$")
	plt.semilogy(r,xi1+xi2,lw=2,c="darkred",label=r"$\xi^s$")
	plt.ylim(0.0005,2)
	plt.xlabel(r"$r_s$")
	plt.legend(loc=(0.1,0.1),frameon=False,handletextpad=0.5)
	plt.tight_layout()
	plt.savefig("xi.pdf")
	os.system("pdfcrop xi.pdf xi.pdf > /dev/null")

	# plt.show()

