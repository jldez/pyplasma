"""

"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})
import os
import tqdm
import copy

from pyplasma import *
import scipy.constants as c

# FIXME: Dirty workaround for overflows when fluence is too high. 
# Stops the run and simply set the plasma density to 100%.
import warnings 
warnings.filterwarnings("error")


def fth(material, tau, tolerance=0.01):

    Fmin, Fmax = 0.1, 10 # FIXME: hardcoded limits for minimal and maximal values
    time = Time(-2*tau, 2*tau, int(10*tau+1000))

    F = 0.75*tau**0.3
    while abs(Fmax-Fmin) > tolerance:

        material_sample = copy.deepcopy(material)

        laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F*1e4, t0=0)
        dom = Domain()
        dom.add_laser(laser, remove_reflected_part=True)
        dom.add_material(material_sample)
        dom.add_observer(Returner('rho'))

        try: # See comment above near warning stuff.
            rho_max = dom.run(time, progress_bar=False)['rho'].max()
        except: rho_max = material.density

        threshold = c.epsilon_0*material.m_red*c.m_e/c.e**2*material.index**2*(laser.omega**2+material.damping**2)
        if rho_max > threshold:
            Fmax = F
        else:
            Fmin = F
        F = np.exp((np.log(Fmax)+np.log(Fmin))/2)

    return F



if __name__ == '__main__':

	taus = np.logspace(np.log10(20), np.log10(1200), 20)*fs

	materials = {

		'sio2':Material(index=1.45, drude_params={'damping':2e15}, 
                        ionization_params={'rate_equation':'dre','bandgap':9*eV,
                                           'density':2.2e28,'cross_section':6.61e-20,
                                           'recombination_rate':1/250e-15}),

		'al2o3':Material(index=1.76, drude_params={'damping':1e15, 'm_CB':0.8}, 
                        ionization_params={'rate_equation':'dre','bandgap':6.5*eV,
                                           'density':2.35e28,'cross_section':1.33e-19}),

		'hfo2':Material(index=2.09, drude_params={'damping':0.5e15, 'm_CB':0.4}, 
                        ionization_params={'rate_equation':'dre','bandgap':5.1*eV,
                                           'density':2.77e28,'cross_section':1.24e-19}),

		'ta2o5':Material(index=2.1, drude_params={'damping':0.4e15, 'm_CB':0.5}, 
                        ionization_params={'rate_equation':'dre','bandgap':3.8*eV,
                                           'density':1.12e28,'cross_section':2.50e-19}),

		'tio2':Material(index=2.52, drude_params={'damping':0.5e15, 'm_CB':0.3}, 
                        ionization_params={'rate_equation':'dre','bandgap':3.3*eV,
                                           'density':3.19e28,'cross_section':1.08e-19}),
	}

	fths_materials = {}
	for material in materials:
		fths_materials[material] = []
		for tau in tqdm.tqdm(taus, f'Calculating Fth for {material}'):
			fths_materials[material].append(fth(materials[material], tau))



	fig = plt.figure(figsize=(6,4))
	colors = plt.cm.terrain(np.linspace(-0.0, 1.0, 19))

	# mero2005 - sio2
	t = [23,30,44,108,150,300,370,635,1100]
	F = [1.7,1.85,2.2,2.8,3.35,4.2,4.45,4.9,6.36]
	plt.scatter(t, F, label=r"$\mathrm{SiO_2}$", marker="o", color=colors[0], s=30)
	# dre
	plt.plot(taus/fs, fths_materials["sio2"], color=colors[0])

	# mero2005 - al2o3
	t = [23,30,44,108,150,300,370,635,1100]
	F = [1.32,1.48,1.65,2.08,2.18,2.75,2.86,3.22,4.03]
	plt.scatter(t, F, label=r"$\mathrm{Al_2O_3}$", marker="s", color=colors[2], s=30)
	# dre - al2o3
	plt.plot(taus/fs, fths_materials["al2o3"], color=colors[2])

	# #mero2005 - hfo2
	t = [28,32,60,102,150,320,430,740,1200]
	F = [0.85,0.925,1.12,1.31,1.53,1.79,1.9,2.4,2.58]
	plt.scatter(t, F, label=r"$\mathrm{HfO_2}$", marker="D", color=colors[4], s=30)
	# dre - hfo2
	plt.plot(taus/fs, fths_materials["hfo2"], color=colors[4])

	#mero2005 - ta2o5
	t = [28,32,60,102,150,320,430,740,1200]
	F = [0.51,0.56,0.64,0.83,0.87,1.06,1.2,1.56,1.81]
	plt.scatter(t, F, label=r"$\mathrm{Ta_2O_5}$", marker="^", color=colors[5], s=30)
	# dre - ta2o5
	plt.plot(taus/fs, fths_materials["ta2o5"], color=colors[5])

	#mero2005 - tio2
	t = [28,32,60,102,150,320,430,740,1200]
	F = [0.47,0.49,0.53,0.59,0.63,0.78,0.96,1.05,1.38]
	plt.scatter(t, F, label=r"$\mathrm{TiO_2}$", marker="v", color=colors[7], s=30)
	# dre - tio2
	plt.plot(taus/fs, fths_materials["tio2"], color=colors[7])


	ax = plt.gca()
	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.xlim(10, 2000)
	plt.ylim(0.25, 10)
	plt.legend(loc=2, frameon=False, ncol=2, columnspacing=1, handletextpad=0.5)
	plt.xlabel(r"$\tau~\mathrm{[fs]}$")
	plt.ylabel(r"$F_{\mathrm{th}}~\mathrm{[J/cm}^2]$")
	plt.tight_layout()

	# plt.savefig("materials.pdf")
	# os.system("pdfcrop materials.pdf materials.pdf > /dev/null")
	plt.show()