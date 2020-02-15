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

    Fmin, Fmax = 0.05, 31 # FIXME: hardcoded limits for minimal and maximal values
    # time = Time(-2*tau, 2*tau, int(10*tau+1000))

    F = 0.75*tau**0.3
    while abs(Fmax-Fmin) > tolerance:

        material_sample = copy.deepcopy(material)

        laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F*1e4, t0=0, phase=False)
        dom = Domain()
        dom.add_laser(laser, remove_reflected_part=True)
        dom.add_material(material_sample)
        dom.add_observer(Returner('rho'))

        try: # See comment above near warning stuff.
            rho_max = dom.run((-2*tau, 2*tau), Nt=10*tau+1000, progress_bar=False)['rho'].max()
        except: rho_max = material.density

        threshold = c.epsilon_0*material.m_red*c.m_e/c.e**2*material.index**2*(laser.omega**2+material.damping**2)
        if rho_max > threshold:
            Fmax = F
        else:
            Fmin = F
        F = np.exp((np.log(Fmax)+np.log(Fmin))/2)

    return F
        


if __name__ == '__main__':

    taus = np.logspace(np.log10(3),np.log10(4500),20)*fs

    fths = []
    for tau in tqdm.tqdm(taus, 'Calculating Fth for SiO2'):
        sio2 = Material(index=1.45, drude_params={'damping':2e15}, 
                        ionization_params={'rate_equation':'dre','bandgap':9*eV,
                                           'density':2.2e28,'cross_section':6.61e-20,
                                           'recombination_rate':1/250e-15})
        fths.append(fth(sio2, tau))


    fths_noII = []
    for tau in tqdm.tqdm(taus, 'Calculating Fth for SiO2 (no II)'):
        sio2 = Material(index=1.45, drude_params={'damping':2e15}, 
                        ionization_params={'rate_equation':'sre','bandgap':9*eV,
                                           'density':2.2e28,'cross_section':6.61e-20,
                                           'recombination_rate':1/250e-15,'alpha_sre':0})
        fths_noII.append(fth(sio2, tau))
        


    fig = plt.figure(figsize=(6,4))
    colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))

    l1, = plt.plot(taus/fs, fths, c="darkred", lw=2.5)
    l2, = plt.plot(taus/fs, fths_noII, c="darkblue", lw=2.0, ls="-.")

    plt.figlegend([l1,l2],[r"$\mathrm{FI+II}$",r"$\mathrm{FI~only}$"],
    			loc=(0.7,0.21),frameon=False, fontsize=13)

    taus_fit = np.logspace(0,4,100)
    fit1, = plt.plot(taus_fit, 0.75*taus_fit**0.3, c="0.5")
    fit2, = plt.plot(taus_fit, 0.63*taus_fit**0.73, c="orange")

    plt.figlegend([fit1,fit2],[r"$F_\mathrm{th}\propto \tau^{0.30}$",r"$F_\mathrm{th}\propto \tau^{0.73}$"],
                loc=(0.7,0.75),frameon=False, fontsize=13)

    #lebugle2014-2 table1 - 800nm - N=1
    t = [7,30,100,300]  #2ln(2)
    F = [1.3,2.8,3.5,4.5]
    plt.scatter(t,F,label=r"Lebugle~et.~al.",marker="v",color=colors[0])

    #chimier2011 fig2 - 800nm - N=1 - LIDT (damage)
    t = [7,28,100,300] #2ln(2)
    F = [1.29,1.85,2.44,3.61]
    plt.scatter(t,F,label=r"Chimier~et.~al.",marker="<",color=colors[1])

    #mero2005 fig1 - 800nm - N=1
    t = [23,30,44,108,150,300,370,635,1100] #2ln(2)
    F = [1.7,1.85,2.2,2.8,3.35,4.2,4.45,4.9,6.36]
    plt.scatter(t,F,label=r"Mero~et.~al.",marker=">",color=colors[2])

    #jia2003 fig1 - 800nm - N=? - roughness less than 10nm
    t = [45,50,65,73,105,180,350,800]
    F = [1.94,1.92,2.03,1.92,2.12,2.06,2.85,3.6]
    plt.scatter(t,F,label=r"Jia~et.~al.",marker="^",color=colors[3])

    #tien1999 fig1d - 800nm - N=1
    t = [24.2,117,197,223,341,448,547,1023,4000,16865,196000,6.82e6]
    F = [3,3.49,3.62,3.47,3.6,3.7,4.2,5.18,6.6,7.48,28.1,159]
    plt.scatter(t[:-3],F[:-3],label=r"Tien~et.~al.",marker="D",color=colors[4])

    #lenzner1998 fig3 - 780nm - N=50
    t = [5,12,18,50,442,1000,3000,5000] #2ln(2)
    F = [1.35,1.8,2.1,3.34,4.61,6.03,7.41,7.18]
    plt.scatter(t,F,label=r"Lenzner~et.~al.",marker="s",color=colors[5])

    #varel1996 fig1 - 790nm - N=1 - very large error bars
    t = [200,367,575,1042,1783,3175,4450]
    F = [5.9,5.9,6.5,5.2,5.85,7.2,11.3]
    plt.scatter(t,F,label=r"Varel~et.~al.",marker="o",color=colors[6])

    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(1,10000)
    plt.ylim(1.0,30)
    plt.legend(loc=(1e-3,0.4),frameon=False,ncol=1,fontsize=13,handletextpad=0)
    plt.xlabel(r"$\tau~\mathrm{[fs]}$")
    plt.ylabel(r"$F_{\mathrm{th}}~\mathrm{[J/cm}^2]$",labelpad=0)
    plt.tight_layout()

    # plt.savefig("fth.pdf")
    # os.system("pdfcrop fth.pdf fth.pdf > /dev/null")

    plt.show()