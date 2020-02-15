
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})
import os

from pyplasma import *
import scipy.constants as c


if __name__ == '__main__':

	f = plt.figure(figsize=(6,5))
	colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))
	ax1,ax2,ax3,ax4 = f.add_subplot(2,3,4),f.add_subplot(2,3,5),f.add_subplot(2,3,6),f.add_subplot(2,1,1)
	ax5 = ax4.twinx()

	for ax, damping in zip([ax1,ax2,ax3], [0, 1, 10]):

		mat = Material(index=1, drude_params={'damping':damping, 'm_VB':np.inf, 'rho':3.5e7})
		las = Laser(omega=1, E0=1, t0=-4.0*c.pi, phase=True)

		dom = Domain()
		dom.add_laser(las)
		dom.add_material(mat)
		dom.add_observer(Returner('Jf'))
		dom.add_observer(Returner('E'))

		results = dom.run((-4.0*c.pi, 1.2*c.pi), Nt=1e3, progress_bar=False)
		J = results['Jf']
		E = results['E']

		t, Nt = dom.times, dom.Nt
		ax.plot(t[int(Nt/2):],E[int(Nt/2):],color="0.5",ls="-",lw=1.5,label=r"$\tilde{E}$")
		ax.plot(t[int(Nt/2):],J[int(Nt/2):],color=colors[2],ls="-",lw=2,label=r"$\tilde{J}/\omega$")
		ax.plot(t[int(Nt/2):],J[int(Nt/2):]*E[int(Nt/2):],color="darkred",lw=2,label=r"$\tilde{P}/\omega$")
		ax.set_xticks([-c.pi,0,c.pi])
		ax.set_xticklabels([r"$-\pi/\omega$",r"$0$",r"$\pi/\omega$"])
		ax.set_xlim([-1.2*c.pi,1.2*c.pi])
		ax.set_yticks([-1,0,1])
		ax.set_yticklabels([r"$-1$",r"$0$",r"$1$"])
		ax.set_ylim([-1.35,1.35])

	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.setp(ax3.get_yticklabels(), visible=False)
	plt.subplots_adjust(wspace = 0.05)
	ax1.set_xlabel(r"$t$",horizontalalignment='right', x=0.525,labelpad=0)
	ax2.set_xlabel(r"$t$",horizontalalignment='right', x=0.525,labelpad=0)
	ax3.set_xlabel(r"$t$",horizontalalignment='right', x=0.525,labelpad=0)
	ax3.legend(loc=(0.25,0.01),labelspacing=0.3,borderaxespad=0,handletextpad=0.2,frameon=False,fontsize=13,handlelength=2.0)
	ax1.text(-3.4,1.05,r"$\mathrm{(b)}~\gamma/\omega=%g$"%0)
	ax2.text(-3.4,1.05,r"$\mathrm{(c)}~\gamma/\omega=%g$"%1)
	ax3.text(-3.4,1.05,r"$\mathrm{(d)}~\gamma/\omega=%g$"%10)


	gammas = np.logspace(-2,2,1000)
	omega = 1

	# Scaling of E_p vs gamma
	def f2(gamma):
		return 1.0/(gamma**2.0+omega**2.0)
	# Scaling of gamma_jh vs gamma
	def f1(gamma):
		return 2.0*gamma*f2(gamma)

	ax4.semilogx(gammas,f2(gammas),color="darkred",lw=2,label=r"$\mathcal{E}_p$")
	ax4.semilogx(gammas,f1(gammas),color=colors[2],lw=2,label=r"$\gamma_\mathrm{jh}$")
	ax4.legend(loc='center left',frameon=False,handletextpad=0.5)
	ax5.semilogx(gammas,np.arctan(-1.0/(gammas + 1e-30))/c.pi,color='0.5',lw=2,label=r"$\Delta\phi$")

	ax5.set_yticks([-0.5,-0.25,0])
	ax5.set_yticklabels([r"$-\pi/2$",r"$-\pi/4$",r"$0$"])

	ax4.set_xlabel(r"$\gamma/\omega$",horizontalalignment='right', x=0.64,labelpad=-7)
	plt.legend(loc='center right',frameon=False,handletextpad=0.5)
	ax4.text(0.011,0.85,r"$\mathrm{(a)}$")
	ax4.set_yticks([0,0.5,1])
	ax4.set_yticklabels([r"$0$",r"$0.5$",r"$1$"])

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.3,wspace=0.2,left=0.06,right=0.91,bottom=0.1,top=0.99)
	# plt.savefig("drude.pdf")
	# os.system("pdfcrop drude.pdf drude.pdf > /dev/null")

	plt.show()
