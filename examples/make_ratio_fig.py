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
import matplotlib.patches as patches
import scipy.constants as c
import os

import pyplasma as pp



if __name__ == '__main__':

	E0_sre, E0_mre, E0_dre = 3.98e9, 4.16e9, 3.66e9

	t_max, N = 100e-15, 1000
	dt = t_max/N
	t = np.linspace(0,t_max*1e15,N)

	# (a) SRE
	sio2 = pp.Material(rate_equation="sre", index=1.5, bandgap=9.*c.e, alpha_sre=0.0004, \
		damping=1e15, cross_section=1e-19, density=2e28, m_CB=1, m_VB=1)
	laser = pp.Laser(wavelength=800e-9, phase=False, E0=E0_sre)

	ii_contribution, ratio_sre = 0, []
	for i in range(N):
		laser.time_step(dt)
		sio2.update_rho(laser,dt)
		ii_contribution += dt*sio2.rho_ii
		ratio_sre.append(ii_contribution/sio2.rho)

	# (a) MRE
	sio2 = pp.Material(rate_equation="mre", index=1.5, bandgap=9.*c.e, \
		damping=1e15, cross_section=1e-19, density=2e28, m_CB=1, m_VB=1)
	laser = pp.Laser(wavelength=800e-9, phase=False, E0=E0_mre)

	ii_contribution, ratio_mre = 0, []
	for i in range(N):
		laser.time_step(dt)
		sio2.update_rho(laser,dt)
		ii_contribution += dt*sio2.rho_ii
		ratio_mre.append(ii_contribution/sio2.rho)

	# (a) DRE
	sio2 = pp.Material(rate_equation="dre", index=1.5, bandgap=9.*c.e, \
		damping=1e15, cross_section=1e-19, density=2e28, m_CB=1, m_VB=1)
	laser = pp.Laser(wavelength=800e-9, phase=False, E0=E0_dre)

	ii_contribution, ratio_dre = 0, []
	for i in range(N):
		laser.time_step(dt)
		sio2.update_rho(laser,dt)
		ii_contribution += dt*sio2.rho_ii
		ratio_dre.append(ii_contribution/sio2.rho)



	NF = 40
	factorF = np.logspace(-0.5,0.2,NF)

	# (b) SRE
	ratio_sre2 = []
	for i in range(NF):
		sio2 = pp.Material(rate_equation="sre", index=1.5, bandgap=9.*c.e, alpha_sre=0.0004, \
			damping=1e15, cross_section=1e-19, density=2e28, m_CB=1, m_VB=1)
		laser = pp.Laser(wavelength=800e-9, phase=False, E0=E0_sre*factorF[i]**2.)
		ii_contribution = 0
		for n in t:
			laser.time_step(dt)
			sio2.update_rho(laser,dt)
			ii_contribution += dt*sio2.rho_ii
		ratio_sre2.append(ii_contribution/sio2.rho)


	# (b) MRE
	ratio_mre2 = []
	for i in range(NF):
		sio2 = pp.Material(rate_equation="mre", index=1.5, bandgap=9.*c.e, \
			damping=1e15, cross_section=1e-19, density=2e28, m_CB=1, m_VB=1)
		laser = pp.Laser(wavelength=800e-9, phase=False, E0=E0_mre*factorF[i]**2.)
		ii_contribution = 0
		for n in t:
			laser.time_step(dt)
			sio2.update_rho(laser,dt)
			ii_contribution += dt*sio2.rho_ii
		ratio_mre2.append(ii_contribution/sio2.rho)


	# (b) MRE
	ratio_dre2 = []
	for i in range(NF):
		sio2 = pp.Material(rate_equation="dre", index=1.5, bandgap=9.*c.e, \
			damping=1e15, cross_section=1e-19, density=2e28, m_CB=1, m_VB=1)
		laser = pp.Laser(wavelength=800e-9, phase=False, E0=E0_dre*factorF[i]**2.)
		ii_contribution = 0
		for n in t:
			laser.time_step(dt)
			sio2.update_rho(laser,dt)
			ii_contribution += dt*sio2.rho_ii
		ratio_dre2.append(ii_contribution/sio2.rho)




	f = plt.figure(figsize=(6,4))
	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))

	ax1=f.add_subplot(1,2,1)
	ax1.plot(t,ratio_sre,label=r"$\mathrm{SRE}$",c="0.5",lw=2,zorder=5)
	ax1.plot(t,ratio_mre,label=r"$\mathrm{MRE}$",c="darkred",lw=2,zorder=4)
	ax1.plot(t,ratio_dre,label=r"$\mathrm{DRE}$",c=colors[2],lw=2,zorder=6)

	ax1.axvline(x = 80,ls=(1, (1, 2)),color="0.6",zorder=3)
	ax1.axhline(y = 0.375,xmin=0.25,ls=(1, (1, 2)),color="0.6",zorder=3)
	ax1.text(45,0.385,r"$75\%$",size=12,color="0.45")
	ax1.axvline(x = 24,ls=(1, (1, 1)),color="0.6")

	rect = patches.Rectangle((-2,0),26,0.6,linewidth=1,edgecolor='0.85',facecolor='0.85')
	ax1.add_patch(rect)
	x_tail = -2.0
	y_tail = 0.255
	x_head = 22
	y_head = 2*y_tail
	dx = x_head - x_tail
	dy = y_head - y_tail
	acolor = "0.5"
	arrow = patches.FancyArrowPatch((x_tail, y_tail), (dx, dy), mutation_scale=20,edgecolor=acolor,facecolor=acolor,zorder=8)
	ax1.add_patch(arrow)
	ax1.text(3,0.275,r"$9\,\hbar\omega$",size=12,color="0.35")

	plt.ylim(0,0.5)
	plt.xlim(-2,t.max()+2)
	plt.ylabel(r"$\rho_{\mathrm{ii}}/\rho$")
	plt.xlabel(r"$t~[\mathrm{fs}]$",labelpad=0)

	ax1.set_xticks([0,20,40,60,80,100])

	ax2=f.add_subplot(1,2,2)
	ax2.plot(factorF,ratio_sre2,label=r"$\mathrm{SRE}$",c="0.5",lw=2,zorder=5)
	ax2.plot(factorF,ratio_mre2,label=r"$\mathrm{MRE}$",c="darkred",lw=2)
	ax2.plot(factorF,ratio_dre2,label=r"$\mathrm{DRE}$",c=colors[2],lw=2,zorder=6)
	ax2.legend(loc=(0.5,0.2),fontsize=12,frameon=False)
	plt.xlim(factorF.min(),factorF.max())
	plt.xlabel(r"$F/F_\mathrm{av}$",labelpad=0)
	ax2.yaxis.tick_right()

	plt.yscale("log")
	plt.ylim(1e-3,1e0)

	ax1.text(2,0.46,r"$\mathrm{(a)}$")
	ax2.text(0.35,0.55,r"$\mathrm{(b)}$")

	plt.tight_layout()
	plt.subplots_adjust(wspace = 0.1)

	plt.savefig("ratio.pdf")
	os.system("pdfcrop ratio.pdf ratio.pdf > /dev/null")

	# plt.show()
