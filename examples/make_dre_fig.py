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

import pyplasma.material as mat
import pyplasma.laser as las
import pyplasma.misc as mis
import pyplasma.drude as dru
import pyplasma.run as run


# TO DO : Calculer Ekmax comme il faut. Gerer les output de calc proprement.


def calc(N,tau,fluence):

	tt = np.linspace(-2*tau, 2*tau, N)
	laser = las.Laser(wavelength=800e-9, pulse_duration=tau, fluence=fluence, t0=tt.min(), transmit=True, phase=False)
	sio2 = mat.Material(index=1.5, bandgap=9.*c.e, rate_equation="dre", density=2e28, cross_section=1e-19, damping=1e15)

	rho, rho_fi, rho_ii, rho_xi = [], [], [], []
	Ekin, g_en, ibh, g_ee = [], [], [], []
	dt = tt[1] - tt[0]
	for t in tt:
		laser.time = t
		laser.update_Electric_field(sio2)
		sio2.update_rho(laser,dt)
		rho.append(sio2.rho)
		rho_fi.append(dt*sio2.rho_fi)
		rho_ii.append(dt*sio2.rho_ii)
		rho_xi.append(sio2.xi)
		Ekin.append(sio2.Ekin)
		g_en.append(mis.g_en(sio2))
		ibh.append(dru.ibh(sio2,laser,s="e"))
		g_ee.append(mis.g_ee(sio2))
	rho = np.array(rho)
	# print(rho[-1]/sio2.density)
	rho_fi = np.cumsum(np.array(rho_fi))
	rho_ii = np.cumsum(np.array(rho_ii))
	rho_xi = np.array(rho_xi)
	Ekin = np.array(Ekin)
	g_en = np.array(g_en)
	ibh = np.array(ibh)
	g_ee = np.array(g_ee)
	return [tt,rho/sio2.density,rho_xi*rho/sio2.density,rho_fi/sio2.density,rho_ii/sio2.density,Ekin,g_en,ibh,g_ee]



def plot_dre(t,data,column, density):

	ax = f.add_subplot(3,2,column)
	t *= 1e15

	# Plasma density
	plt.semilogy(t,data["rho"]/density,label=r"$\rho$",color=colors[2],lw=lw)
	plt.semilogy(t,data["rho"]*data["xi"]/density,label=r"$\xi^e\rho$",color="darkred",lw=lw)
	plt.semilogy(t,np.cumsum(data["rho_fi"])/density,color="k",ls="--",label=r"$\rho_{\mathrm{fi}}$",lw=lw)
	plt.semilogy(t,np.cumsum(data["rho_ii"])/density,color="k",ls=":",label=r"$\rho_{\mathrm{ii}}$",lw=lw)
	plt.ylim(1e-8,1e0)
	plt.setp(ax.get_xticklabels(),visible=False)
	if (column==1): 
		plt.ylabel(r"$\mathrm{Density}/\rho_{\mathrm{mol}}$")
		plt.setp(ax.get_yticklabels(),visible=False)
		plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,borderaxespad=0.2)
		plt.text(-18.7,2e-1,r"$\mathrm{(a)}$")
	else :
		ax.yaxis.tick_right()
		plt.text(-13.1*42.9,2e-1,r"$\mathrm{(d)}$")
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.set_ticks([1e-8,1e-6,1e-4,1e-2,1e0])
	plt.xlim(t.min(),t.max())
	
	# Temperature of the electrons
	ax2 = f.add_subplot(3,2,column+2)
	EF = c.hbar**2.*(3.*c.pi**2.*data["rho"])**(2./3)/(2.*c.m_e)
	Ekinmax = [1.7822312789626924e-18,7.290911250988981e-19][column-1]*np.ones((len(t)))/c.e
	plt.semilogy(t,data["Ekin"]/c.e,color=colors[2],label=r"$\mathcal{E}^e_\mathrm{k}$",lw=lw)
	plt.semilogy(t,EF/c.e,label=r"$\mathcal{E}_F$",color="darkred",ls="-",lw=lw)
	plt.semilogy(t,Ekinmax,color="k",ls="--",label="limit",lw=lw)
	plt.ylim(0.01,30)
	plt.setp(ax2.get_xticklabels(),visible=False)
	if column==1 :
		plt.ylabel(r"$\mathrm{Energy~[eV]}$")
		plt.setp(ax2.get_yticklabels(),visible=False)
		plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,borderaxespad=0.2)
		plt.text(-18.7,15,r"$\mathrm{(b)}$")
	else:
		ax2.yaxis.tick_right()
		plt.text(-13.1*42.9,15,r"$\mathrm{(e)}$")
	ax2.yaxis.set_ticks_position('both')
	plt.xlim(t.min(),t.max())

	# Collision rates
	ax3 = f.add_subplot(3,2,column+4)
	plt.semilogy(t,data["collision_freq_en"]*1e-15,label=r"$\gamma_{n}^e$",color=colors[2],lw=lw)
	plt.semilogy(t,data["ibh"]*1e-15,label=r"$\gamma_\mathrm{ib}^e$",color="darkred",lw=lw)
	plt.semilogy(t,data["collision_freq_ee"]*1e-15,label=r"$\gamma_{e}^e$",color="k",ls="--",lw=lw)
	plt.ylim(5e-7,50)
	if column==1:
		plt.ylabel(r"$\mathrm{Frequency~[fs}^{-1}]$")
		plt.setp(ax3.get_yticklabels(),visible=False)
		plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,bbox_to_anchor=(0.97, -0.03))
		plt.text(-18.7,10,r"$\mathrm{(c)}$")
	else:
		ax3.yaxis.tick_right()
		plt.text(-13.1*42.9,10,r"$\mathrm{(f)}$")
	plt.xlabel(r"$t~[\mathrm{fs}]$")
	ax3.yaxis.set_ticks_position('both')
	ax3.yaxis.set_ticks([1e-6,1e-4,1e-2,1e0])
	plt.xlim(t.min(),t.max())
	if column==2:
		ax.xaxis.set_ticks([-500,-250,0,250,500])
		ax2.xaxis.set_ticks([-500,-250,0,250,500])
		ax3.xaxis.set_ticks([-500,-250,0,250,500])





if __name__ == '__main__':

	tau, fluence, N = 10e-15, 1.6e4, 2000
	time_abc = np.linspace(-2*tau, 2*tau, N)
	sio2_abc = mat.Material(index=1.5, bandgap=9.*c.e, rate_equation="dre", density=2e28, cross_section=1e-19, damping=1e15)
	laser_abc = las.Laser(wavelength=800e-9, pulse_duration=tau, fluence=fluence, t0=time_abc.min(), transmit=True, phase=False)
	data_abc = run.run(time_abc, sio2_abc, laser_abc, progress_bar=False, \
		output=["rho","rho_fi","rho_ii","xi","Ekin","ibh","collision_freq_en","collision_freq_ee"])
	data_abc["ibh"] /= 2 # because we want the ibh rate for electrons only (without holes)


	tau, fluence, N = 300e-15, 4.8e4, 10000
	time_def = np.linspace(-2*tau, 2*tau, N)
	sio2_def = mat.Material(index=1.5, bandgap=9.*c.e, rate_equation="dre", density=2e28, cross_section=1e-19, damping=1e15)
	laser_def = las.Laser(wavelength=800e-9, pulse_duration=tau, fluence=fluence, t0=time_def.min(), transmit=True, phase=False)
	data_def = run.run(time_def, sio2_def, laser_def, progress_bar=False, \
		output=["rho","rho_fi","rho_ii","xi","Ekin","ibh","collision_freq_en","collision_freq_ee"])
	data_def["ibh"] /= 2 # because we want the ibh rate for electrons only (without holes)


	f = plt.figure(figsize=(6,8))
	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))
	lw=2

	plot_dre(time_abc,data_abc,1, density=sio2_abc.density)
	plot_dre(time_def,data_def,2, density=sio2_def.density)

	plt.subplots_adjust(hspace = 0.1,wspace = 0.1)
	plt.savefig("dre.pdf")
	os.system("pdfcrop dre.pdf dre.pdf > /dev/null")

	# plt.show()