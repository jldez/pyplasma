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
import tqdm
import copy

from pyplasma import *
import scipy.constants as c



def fth(material, tau, tolerance=0.01, criterion='energy'):

    Fmin, Fmax = 0.05, 31 # FIXME: hardcoded limits for minimal and maximal values

    F = 30
    while abs(Fmax-Fmin) > tolerance:

        material_sample = copy.deepcopy(material)

        laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F*1e4, t0=0, phase=False)
        dom = Domain()
        dom.add_laser(laser, remove_reflected_part=True)
        dom.add_material(material_sample)
        dom.add_observer(Returner('fi_rate', out_step=1))
        dom.add_observer(Returner('ii_rate', out_step=1))
        dom.add_observer(Returner('rho', out_step=1))
        dom.add_observer(Returner('el_heating_rate', out_step=1))
        dom.add_observer(Returner('hl_heating_rate', out_step=1))

        def full_ionization(rho):
            return rho > 0.99*material.density
        dom.add_observer(Stopper('rho', full_ionization, verbose=False))

        def is_above_threshold(results):
            if criterion == 'energy':
                absorbed_energy_interband = np.sum(results['fi_rate']*material.bandgap)
                absorbed_energy_intraband = np.sum(results['rho']*(results['el_heating_rate']+results['hl_heating_rate'])*c.hbar*dom.laser.omega*dom.dt)
                return absorbed_energy_intraband + absorbed_energy_interband > 3e9
            if criterion == 'optical':
                threshold = c.epsilon_0*material.m_red*c.m_e/c.e**2*material.index**2*(laser.omega**2+material.damping**2)
                return results['rho'].max() > threshold

        results = dom.run((-2*tau, 2*tau), Nt=500+tau/fs, progress_bar=False)

        if is_above_threshold(results):
            Fmax = F
        else:
            Fmin = F

        F = np.exp((np.log(Fmax)+np.log(Fmin))/2)

    return F
        


if __name__ == '__main__':

    fig = plt.figure(figsize=(6.5,4.2))
    colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))

    taus = np.logspace(np.log10(3),np.log10(4500),20)*fs

    fths = []
    for tau in tqdm.tqdm(taus, 'Calculating Fth for SiO2'):
        sio2 = Material(index=1.45, drude_params={'damping':2e15, 'm_CB':1}, 
                        ionization_params={'rate_equation':'dre','bandgap':9*eV,
                                           'density':2.2e28,'cross_section':6.61e-20,
                                           'recombination_rate':1/250e-15})
        fths.append(fth(sio2, tau, criterion='energy'))
    # print(fths)
    # fths = [0.6598646069626078, 0.7958770558798245, 0.9629276419411922, 1.1614079746163524, 1.3325135899941751, 
    #         1.6046655807434704, 1.8182152831027953, 2.0346044680086024, 2.378515314993702, 2.7632416864546827, 
    #         3.170339088515025, 3.5726672871097924, 3.8749140312753956, 4.144071441108967, 4.572540780636695, 
    #         5.172971604549961, 5.93508492654683, 6.903169741876832, 8.184249242589107, 9.902081943439397]


    fths_noII = []
    for tau in tqdm.tqdm(taus, 'Calculating Fth for SiO2 (no II)'):
        sio2 = Material(index=1.45, drude_params={'damping':2e15, 'm_CB':1}, 
                        ionization_params={'rate_equation':'dre','bandgap':9*eV,
                                           'density':2.2e28,'cross_section':0,
                                           'recombination_rate':1/250e-15})
        fths_noII.append(fth(sio2, tau, criterion='energy'))
    # print(fths_noII)
    # fths_noII = [0.6765610446109084, 0.865908498738542, 1.0443910756322798, 1.3662299461013174, 1.7458645826432764, 
    #              2.079579980865273, 2.720421986994549, 3.5782512609408923, 4.688238972709206, 6.000324734807675, 
    #              6.9845028555800575, 9.179751529374515, 12.22620179541403, 16.43376105418735, 22.275528759593175, 
    #              30.42957033121838, 30.996029611672036, 30.996029611672036, 30.996029611672036, 30.996029611672036]

    fths_noJH = []
    for tau in tqdm.tqdm(taus, 'Calculating Fth for SiO2 (no JH)'):
        sio2 = Material(index=1.45, drude_params={'damping':2e15, 'm_CB':1}, 
                        ionization_params={'rate_equation':'dre','bandgap':9*eV,
                                           'density':2.2e28,'cross_section':0,
                                           'recombination_rate':1/250e-15})
        fths_noJH.append(fth(sio2, tau, criterion='optical'))
    # print(fths_noJH)
    # fths_noJH = [1.1186818030887706, 1.4272993520099948, 1.8759035081532471, 2.3934204116198896, 3.1802571266568043, 
    #              4.124700838868995, 5.421105784725956, 7.256928748477006, 9.703069483933826, 12.51599708574316, 
    #              17.05476082076921, 23.426202513122977, 30.996029611672036, 30.996029611672036, 30.996029611672036, 
    #              30.996029611672036, 30.996029611672036, 30.996029611672036, 30.996029611672036, 30.996029611672036]

    ax = fig.add_subplot(1, 1, 1)
    l1, = plt.plot(taus/fs, fths, c="darkred", lw=2.5)
    l2, = plt.plot(taus/fs, fths_noII, c="darkgreen", lw=2.0, ls="--")
    l3, = plt.plot(taus/fs, fths_noJH, c="darkblue", lw=2.0, ls="-.")

    plt.figlegend([l1,l2,l3],[r"$\mathrm{FI+JH+II}$",r"$\mathrm{FI+JH}$",r"$\mathrm{FI~only}$"],
    			loc=(0.7,0.21),frameon=False, fontsize=13)

    pos_arrow1 = 9
    arrow = patches.FancyArrowPatch((taus[pos_arrow1]/fs, fths_noJH[pos_arrow1]), (taus[pos_arrow1]/fs, fths_noII[pos_arrow1]), mutation_scale=12, edgecolor='darkgreen', facecolor='darkgreen', zorder=8)
    ax.add_patch(arrow)
    plt.text(taus[pos_arrow1]/fs,(fths_noJH[pos_arrow1]+2*fths_noII[pos_arrow1])/3, r"$\mathrm{heating}$", rotation=55, c='darkgreen')

    pos_arrow2 = (14,18)
    arrow = patches.FancyArrowPatch((taus[pos_arrow2[0]]/fs, fths_noII[pos_arrow2[0]]), (taus[pos_arrow2[1]]/fs, fths[pos_arrow2[1]]), mutation_scale=12, edgecolor='darkred', facecolor='darkred', zorder=8, connectionstyle='arc3,rad=-0.2')
    ax.add_patch(arrow)
    plt.text((taus[pos_arrow2[0]]+taus[pos_arrow2[1]])/2/fs,(fths_noII[pos_arrow2[0]]+fths[pos_arrow2[1]])/2, r"$\mathrm{avalanche}$", c='darkred')

    #lebugle2014-2 table1 - 800nm - N=1
    t = np.array([7,30,100,300])  #2ln(2)
    F = np.array([1.3,2.8,3.5,4.5])
    plt.scatter(t,F,label=r"Lebugle~et.~al.",marker="v",color=colors[0])

    #chimier2011 fig2 - 800nm - N=1 - LIDT (damage)
    t = np.array([7,28,100,300]) #2ln(2)
    F = np.array([1.29,1.85,2.44,3.61])
    plt.scatter(t,F,label=r"Chimier~et.~al.",marker="<",color=colors[1])

    #mero2005 fig1 - 800nm - N=1
    t = np.array([23,30,44,108,150,300,370,635,1100]) #2ln(2)
    F = np.array([1.7,1.85,2.2,2.8,3.35,4.2,4.45,4.9,6.36])
    plt.scatter(t,F,label=r"Mero~et.~al.",marker=">",color=colors[2])

    #jia2003 fig1 - 800nm - N=? - roughness less than 10nm
    t = np.array([45,50,65,73,105,180,350,800])
    F = np.array([1.94,1.92,2.03,1.92,2.12,2.06,2.85,3.6])
    plt.scatter(t,F,label=r"Jia~et.~al.",marker="^",color=colors[3])

    #tien1999 fig1d - 800nm - N=1
    t = np.array([24.2,117,197,223,341,448,547,1023,4000,16865,196000,6.82e6])
    F = np.array([3,3.49,3.62,3.47,3.6,3.7,4.2,5.18,6.6,7.48,28.1,159])
    plt.scatter(t[:-3],F[:-3],label=r"Tien~et.~al.",marker="D",color=colors[4])

    #lenzner1998 fig3 - 780nm - N=50
    t = np.array([5,12,18,50,442,1000,3000,5000]) #2ln(2)
    F = np.array([1.35,1.8,2.1,3.34,4.61,6.03,7.41,7.18])
    plt.scatter(t,F,label=r"Lenzner~et.~al.",marker="s",color=colors[5])

    #varel1996 fig1 - 790nm - N=1 - very large error bars
    t = np.array([200,367,575,1042,1783,3175,4450])
    F = np.array([5.9,5.9,6.5,5.2,5.85,7.2,11.3])
    plt.scatter(t,F,label=r"Varel~et.~al.",marker="o",color=colors[6])

    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlim(0.8,12000)
    plt.ylim(1.0,30)
    plt.legend(loc=(1e-3,0.4),frameon=False,ncol=1,fontsize=13,handletextpad=0)
    plt.xlabel(r"$\tau~\mathrm{[fs]}$")
    plt.ylabel(r"$F_{\mathrm{th}}~\mathrm{[J/cm}^2]$",labelpad=0)
    plt.tight_layout()

    # plt.savefig("fth_energy.pdf")
    # os.system("pdfcrop fth_energy.pdf fth_energy.pdf > /dev/null")

    plt.show()