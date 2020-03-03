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
from pyplasma.misc import el_Ekin_max, ee_coll_freq
import scipy.constants as c



if __name__ == '__main__':

	fig = plt.figure(figsize=(6,8))
	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))
	lw=2

	params_abc = [10*fs, 1.5e4, 2000]
	params_def = [300*fs, 4.7e4, 10000]

	for i, params in enumerate([params_abc, params_def]):

		tau, fluence, Nt = params
		mat = Material(index=1.5, drude_params={'damping':1e15}, ionization_params={'rate_equation':'dre','bandgap':9*eV,'density':2e28,'cross_section':1e-19})
		las = Laser(wavelength=800*nm, pulse_duration=tau, fluence=fluence, t0=0, phase=False)
		dom = Domain()
		dom.add_laser(las, remove_reflected_part=True)
		dom.add_material(mat)

		for target in ['rho','xi_e','rho_fi','rho_ii','Ekin','E','coll_freq_en','el_heating_rate']:
			dom.add_observer(Returner(target))

		results = dom.run((-2*tau,2*tau), Nt=Nt)
		t_axis = dom.times/fs

		# Plasma density
		# print((results['rho']/mat.density).max())
		ax = fig.add_subplot(3,2,1+i)
		ax.semilogy(t_axis, results['rho']/mat.density, label=r"$\rho$", color=colors[2], lw=lw)
		ax.semilogy(t_axis, results["rho"]*results["xi_e"]/mat.density, label=r"$\xi^e\rho$", color="darkred", lw=lw)
		ax.semilogy(t_axis, results["rho_fi"]/mat.density, color="k", ls="--", label=r"$\rho_{\mathrm{fi}}$", lw=lw)
		ax.semilogy(t_axis, results["rho_ii"]/mat.density, color="k", ls=":", label=r"$\rho_{\mathrm{ci}}$", lw=lw)
		plt.ylim(1e-8, 1e0)
		plt.setp(ax.get_xticklabels(), visible=False)
		if i == 0:
			plt.ylabel(r"$\mathrm{Density}/\rho_{\mathrm{mol}}$")
			plt.setp(ax.get_yticklabels(), visible=False)
			plt.legend(loc=4, frameon=False, labelspacing=0.3, handletextpad=0.2, borderaxespad=0.2)
			ax.text(-18.7, 2e-1, r"$\mathrm{(a)}$")
		if i == 1:
			ax.yaxis.tick_right()
			plt.text(-13.1*42.9,2e-1,r"$\mathrm{(d)}$")
		# ax.yaxis.set_ticks_position('both')
		ax.yaxis.set_ticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0])
		plt.xlim(t_axis.min(), t_axis.max())

		# # Electric field
		# ax = ax.twinx()
		# ax.fill_between(t_axis, results['E']/las.E0, alpha=0.3)
		# plt.setp(ax.get_xticklabels(), visible=False)
		# plt.setp(ax.get_yticklabels(), visible=False)
		# plt.ylim(0,1)

		# Temperature of the electrons
		ax2 = fig.add_subplot(3,2,3+i)
		Fermi_energy = c.hbar**2*(3*c.pi**2*results["rho"])**(2/3)/(2*c.m_e)
		Ekinmax = el_Ekin_max(results['E'].max(), mat, las)
		# print(Ekinmax/eV, results["Ekin"].max()/eV)
		ax2.semilogy(t_axis, results["Ekin"]/eV, color=colors[2], label=r"$\mathcal{E}^e_\mathrm{kin}$", lw=lw)
		ax2.semilogy(t_axis, Fermi_energy/eV, label=r"$\mathcal{E}_F$", color="darkred", ls="-", lw=lw)
		ax2.semilogy(t_axis, Ekinmax*np.ones(Nt)/eV, color="k",ls="--", label="limit", lw=lw)
		plt.ylim(0.01,30)
		plt.setp(ax2.get_xticklabels(),visible=False)
		if i == 0:
			plt.ylabel(r"$\mathrm{Energy~[eV]}$")
			plt.setp(ax2.get_yticklabels(),visible=False)
			plt.legend(loc=4,frameon=False,labelspacing=0.3,handletextpad=0.2,borderaxespad=0.2)
			plt.text(-18.7, 15, r"$\mathrm{(b)}$")
		if i == 1:
			ax2.yaxis.tick_right()
			plt.text(-13.1*42.9, 15, r"$\mathrm{(e)}$")
		ax2.yaxis.set_ticks_position('both')
		plt.xlim(t_axis.min(),t_axis.max())

		# Collision rates
		ax3 = fig.add_subplot(3,2,5+i)
		plt.semilogy(t_axis, results["coll_freq_en"]*fs, label=r"$\gamma_{n}^e$", color=colors[2], lw=lw)
		plt.semilogy(t_axis, results["el_heating_rate"]*fs, label=r"$\gamma_\mathrm{jh}^e$", color="darkred", lw=lw)
		plt.semilogy(t_axis, ee_coll_freq(results['Ekin'], mat)*1e-15, label=r"$\gamma_{e}^e$", color="k", ls="--", lw=lw)
		plt.ylim(5e-7, 50)
		if i == 0:
			plt.ylabel(r"$\mathrm{Frequency~[fs}^{-1}]$")
			plt.setp(ax3.get_yticklabels(), visible=False)
			plt.legend(loc=4, frameon=False, labelspacing=0.3, handletextpad=0.2, bbox_to_anchor=(0.97, -0.03))
			plt.text(-18.7, 10, r"$\mathrm{(c)}$")
		if i == 1:
			ax3.yaxis.tick_right()
			plt.text(-13.1*42.9, 10, r"$\mathrm{(f)}$")
		plt.xlabel(r"$t~[\mathrm{fs}]$")
		ax3.yaxis.set_ticks_position('both')
		ax3.yaxis.set_ticks([1e-6, 1e-4, 1e-2, 1e0])
		plt.xlim(t_axis.min(), t_axis.max())

		if i == 1:
			ax.xaxis.set_ticks([-500,-250,0,250,500])
			ax2.xaxis.set_ticks([-500,-250,0,250,500])
			ax3.xaxis.set_ticks([-500,-250,0,250,500])


	plt.subplots_adjust(hspace = 0.1,wspace = 0.1)
	# plt.savefig("dre.pdf")
	# os.system("pdfcrop dre.pdf dre.pdf > /dev/null")

	plt.show()

