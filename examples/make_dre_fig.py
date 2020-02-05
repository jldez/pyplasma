#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})
import os

from pyplasma import *
from pyplasma.field_ionization import fi_rate
from pyplasma.misc import el_Ekin_max
import scipy.constants as c



if __name__ == '__main__':

	fig = plt.figure(figsize=(6,8))
	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))
	lw=2

	tau, fluence, Nt = 10*fs, 1.6e4, 2000
	time = Time(-2*tau, 2*tau, Nt)
	mat = Material(index=1.5, drude_params={'damping':1e15}, ionization_params={'rate_equation':'dre','bandgap':9*eV,'density':2e28,'cross_section':1e-19})
	las = Laser(wavelength=800*nm, pulse_duration=tau, fluence=fluence, t0=0, phase=False)
	dom = Domain()
	dom.add_laser(las, remove_reflected_part=True)
	dom.add_material(mat)

	for target in ['rho','xi_e','rho_fi','rho_ii','Ekin','E']:
		dom.add_observer(Observer(target,'return'))

	results = dom.run(time)

	# Plasma density
	ax = fig.add_subplot(321)
	ax.semilogy(time.t, results['rho']/mat.density, label=r"$\rho$", color=colors[2], lw=lw)
	ax.semilogy(time.t, results["rho"]*results["xi_e"]/mat.density, label=r"$\xi^e\rho$", color="darkred", lw=lw)
	ax.semilogy(time.t, results["rho_fi"]/mat.density, color="k", ls="--", label=r"$\rho_{\mathrm{fi}}$", lw=lw)
	ax.semilogy(time.t, results["rho_ii"]/mat.density, color="k", ls=":", label=r"$\rho_{\mathrm{ii}}$", lw=lw)
	plt.ylim(1e-8, 1e0)
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.ylabel(r"$\mathrm{Density}/\rho_{\mathrm{mol}}$")
	plt.setp(ax.get_yticklabels(), visible=False)
	plt.legend(loc=4, frameon=False, labelspacing=0.3, handletextpad=0.2, borderaxespad=0.2)
	plt.text(-18.7, 2e-1, r"$\mathrm{(a)}$")
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.set_ticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0])
	plt.xlim(time.t.min(), time.t.max())

	# Temperature of the electrons
	ax2 = fig.add_subplot(323)
	Fermi_energy = c.hbar**2*(3*c.pi**2*results["rho"])**(2/3)/(2*c.m_e)
	Ekinmax = el_Ekin_max(results['E'].max(), mat, las)
	ax2.semilogy(time.t, results["Ekin"]/eV, color=colors[2], label=r"$\mathcal{E}^e_\mathrm{k}$", lw=lw)
	ax2.semilogy(time.t, Fermi_energy/eV, label=r"$\mathcal{E}_F$", color="darkred", ls="-", lw=lw)
	ax2.semilogy(time.t, Ekinmax*np.ones(Nt)/eV, color="k",ls="--", label="limit", lw=lw)
	plt.ylim(0.01,30)
	plt.setp(ax2.get_xticklabels(),visible=False)
	plt.ylabel(r"$\mathrm{Energy~[eV]}$")
	plt.setp(ax2.get_yticklabels(),visible=False)
	plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,borderaxespad=0.2)
	plt.text(-18.7,15,r"$\mathrm{(b)}$")
	ax2.yaxis.set_ticks_position('both')
	plt.xlim(time.t.min(),time.t.max())


	plt.subplots_adjust(hspace = 0.1,wspace = 0.1)
	# plt.savefig("dre.pdf")
	# os.system("pdfcrop dre.pdf dre.pdf > /dev/null")

	plt.show()



# def plot_dre(t,data,column,density,Ekinmax):

# 	ax = f.add_subplot(3,2,column)
# 	t *= 1e15

# 	# Plasma density
# 	plt.semilogy(t,data["rho"]/density,label=r"$\rho$",color=colors[2],lw=lw)
# 	plt.semilogy(t,data["rho"]*data["xi"]/density,label=r"$\xi^e\rho$",color="darkred",lw=lw)
# 	plt.semilogy(t,data["rho_fi"]/density,color="k",ls="--",label=r"$\rho_{\mathrm{fi}}$",lw=lw)
# 	plt.semilogy(t,data["rho_ii"]/density,color="k",ls=":",label=r"$\rho_{\mathrm{ii}}$",lw=lw)
# 	plt.ylim(1e-8,1e0)
# 	plt.setp(ax.get_xticklabels(),visible=False)
# 	if (column==1): 
# 		plt.ylabel(r"$\mathrm{Density}/\rho_{\mathrm{mol}}$")
# 		plt.setp(ax.get_yticklabels(),visible=False)
# 		plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,borderaxespad=0.2)
# 		plt.text(-18.7,2e-1,r"$\mathrm{(a)}$")
# 	else :
# 		ax.yaxis.tick_right()
# 		plt.text(-13.1*42.9,2e-1,r"$\mathrm{(d)}$")
# 	ax.yaxis.set_ticks_position('both')
# 	ax.yaxis.set_ticks([1e-8,1e-6,1e-4,1e-2,1e0])
# 	plt.xlim(t.min(),t.max())
	
# 	# Temperature of the electrons
# 	ax2 = f.add_subplot(3,2,column+2)
# 	EF = c.hbar**2.*(3.*c.pi**2.*data["rho"])**(2./3)/(2.*c.m_e)
# 	plt.semilogy(t,data["Ekin"]/c.e,color=colors[2],label=r"$\mathcal{E}^e_\mathrm{k}$",lw=lw)
# 	plt.semilogy(t,EF/c.e,label=r"$\mathcal{E}_F$",color="darkred",ls="-",lw=lw)
# 	plt.semilogy(t,Ekinmax*np.ones((len(t)))/c.e,color="k",ls="--",label="limit",lw=lw)
# 	plt.ylim(0.01,30)
# 	plt.setp(ax2.get_xticklabels(),visible=False)
# 	if column==1 :
# 		plt.ylabel(r"$\mathrm{Energy~[eV]}$")
# 		plt.setp(ax2.get_yticklabels(),visible=False)
# 		plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,borderaxespad=0.2)
# 		plt.text(-18.7,15,r"$\mathrm{(b)}$")
# 	else:
# 		ax2.yaxis.tick_right()
# 		plt.text(-13.1*42.9,15,r"$\mathrm{(e)}$")
# 	ax2.yaxis.set_ticks_position('both')
# 	plt.xlim(t.min(),t.max())

# 	# Collision rates
# 	ax3 = f.add_subplot(3,2,column+4)
# 	plt.semilogy(t,data["collision_freq_en"]*1e-15,label=r"$\gamma_{n}^e$",color=colors[2],lw=lw)
# 	plt.semilogy(t,data["ibh"]*1e-15,label=r"$\gamma_\mathrm{ib}^e$",color="darkred",lw=lw)
# 	plt.semilogy(t,data["collision_freq_ee"]*1e-15,label=r"$\gamma_{e}^e$",color="k",ls="--",lw=lw)
# 	plt.ylim(5e-7,50)
# 	if column==1:
# 		plt.ylabel(r"$\mathrm{Frequency~[fs}^{-1}]$")
# 		plt.setp(ax3.get_yticklabels(),visible=False)
# 		plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,bbox_to_anchor=(0.97, -0.03))
# 		plt.text(-18.7,10,r"$\mathrm{(c)}$")
# 	else:
# 		ax3.yaxis.tick_right()
# 		plt.text(-13.1*42.9,10,r"$\mathrm{(f)}$")
# 	plt.xlabel(r"$t~[\mathrm{fs}]$")
# 	ax3.yaxis.set_ticks_position('both')
# 	ax3.yaxis.set_ticks([1e-6,1e-4,1e-2,1e0])
# 	plt.xlim(t.min(),t.max())
# 	if column==2:
# 		ax.xaxis.set_ticks([-500,-250,0,250,500])
# 		ax2.xaxis.set_ticks([-500,-250,0,250,500])
# 		ax3.xaxis.set_ticks([-500,-250,0,250,500])





# if __name__ == '__main__':

# 	tau, fluence, N = 10e-15, 1.6e4, 2000
# 	time_abc = np.linspace(-2*tau, 2*tau, N)
# 	sio2_abc = pp.Material(index=1.5, bandgap=9.*c.e, rate_equation="dre", density=2e28, cross_section=1e-19, damping=1e15)
# 	laser_abc = pp.Laser(wavelength=800e-9, pulse_duration=tau, fluence=fluence, t0=time_abc.min(), transmit=True, phase=False)
# 	data_abc = pp.run(time_abc, sio2_abc, laser_abc, progress_bar=False, \
# 		output=["rho","rho_fi","rho_ii","xi","Ekin","ibh","collision_freq_en","collision_freq_ee","electric_field"])
# 	data_abc["ibh"] /= 2 # because we want the ibh rate for electrons only (without holes)
# 	Ekinmax_abc = pp.Ekin_max(sio2_abc,laser_abc,E=data_abc["electric_field"],s="e")
# 	print(Ekinmax_abc/c.e, data_abc["Ekin"].max()/c.e)

# 	tau, fluence, N = 300e-15, 4.8e4, 10000
# 	time_def = np.linspace(-2*tau, 2*tau, N)
# 	sio2_def = pp.Material(index=1.5, bandgap=9.*c.e, rate_equation="dre", density=2e28, cross_section=1e-19, damping=1e15)
# 	laser_def = pp.Laser(wavelength=800e-9, pulse_duration=tau, fluence=fluence, t0=time_def.min(), transmit=True, phase=False)
# 	data_def = pp.run(time_def, sio2_def, laser_def, progress_bar=False, \
# 		output=["rho","rho_fi","rho_ii","xi","Ekin","ibh","collision_freq_en","collision_freq_ee","electric_field"])
# 	data_def["ibh"] /= 2 # because we want the ibh rate for electrons only (without holes)
# 	Ekinmax_def = pp.Ekin_max(sio2_def,laser_def,E=data_def["electric_field"],s="e")
# 	print(Ekinmax_def/c.e, data_def["Ekin"].max()/c.e)


# 	f = plt.figure(figsize=(6,8))
# 	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))
# 	lw=2

# 	plot_dre(time_abc,data_abc,1, density=sio2_abc.density, Ekinmax=Ekinmax_abc)
# 	plot_dre(time_def,data_def,2, density=sio2_def.density, Ekinmax=Ekinmax_def)

# 	plt.subplots_adjust(hspace = 0.1,wspace = 0.1)
# 	plt.savefig("dre.pdf")
# 	os.system("pdfcrop dre.pdf dre.pdf > /dev/null")

# 	# plt.show()