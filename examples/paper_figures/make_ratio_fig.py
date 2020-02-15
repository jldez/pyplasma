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
import scipy.constants as c



if __name__ == '__main__':

    ### (a) ###############################################################################

    # Ajust each E0 to obtain a ratio of 0.5
    params_sre = {'rate_equation':'sre', 'E0':3.98e9}
    params_mre = {'rate_equation':'mre', 'E0':4.16e9}
    params_dre = {'rate_equation':'dre', 'E0':3.66e9}
    
    time = Time(0, 100*fs, 1000)
    ratios = {'time':np.linspace(0,100,1000)}
    for params in [params_sre, params_mre, params_dre]:
        mat = Material(index=1.5, drude_params={'damping':1e15}, 
                       ionization_params={'rate_equation':params['rate_equation'],
                                          'bandgap':9*eV,'alpha_sre':0.0004,
                                          'density':2e28,'cross_section':1e-19})
        las = Laser(wavelength=800*nm, phase=False, E0=params['E0'])
        dom = Domain()
        dom.add_laser(las, remove_reflected_part=False)
        dom.add_material(mat)
        dom.add_observer(Returner('rho'))
        dom.add_observer(Returner('rho_ii'))
        dom.add_observer(Returner('E'))
        results = dom.run(100*fs, Nt=1000, progress_bar=False)
        ratio = results['rho_ii']/results['rho']
        ratios[params['rate_equation']] = ratio
        fluence = dom.dt*c.c*c.epsilon_0*np.sum(np.array(results['E'])**2.)
        print(params['rate_equation'] + f': ratio: {ratio.max()}, fluence: {fluence/1e4}')


    fig = plt.figure(figsize=(6,4))
    colors = plt.cm.terrain(np.linspace(0.0, 1.0, 24))

    ax1 = fig.add_subplot(121)
    ax1.plot(ratios['time'], ratios['sre'], label=r"$\mathrm{SRE}$", c="0.5", lw=2, zorder=5)
    ax1.plot(ratios['time'], ratios['mre'], label=r"$\mathrm{MRE}$", c="darkred", lw=2, zorder=4)
    ax1.plot(ratios['time'], ratios['dre'], label=r"$\mathrm{DRE}$", c=colors[2], lw=2, zorder=6)

    ax1.axvline(x = 80, ls=(1, (1, 2)), color="0.6", zorder=3)
    ax1.axhline(y = 0.375, xmin=0.25, ls=(1, (1, 2)), color="0.6", zorder=3)
    ax1.text(45, 0.385, r"$75\%$", size=12, color="0.45")
    ax1.axvline(x = 24, ls=(1, (1, 1)), color="0.6")

    rect = patches.Rectangle((-2,0), 26, 0.6, linewidth=1, edgecolor='0.85', facecolor='0.85')
    ax1.add_patch(rect)
    x_tail = -2.0
    y_tail = 0.255
    x_head = 22
    y_head = 2*y_tail
    dx = x_head - x_tail
    dy = y_head - y_tail
    acolor = "0.5"
    arrow = patches.FancyArrowPatch((x_tail, y_tail), (dx, dy), mutation_scale=20, edgecolor=acolor, facecolor=acolor, zorder=8)
    ax1.add_patch(arrow)
    ax1.text(3, 0.275, r"$9\,\hbar\omega$", size=12, color="0.35")

    plt.ylim(0, 0.5)
    plt.xlim(-2, ratios['time'].max() + 2)
    plt.ylabel(r"$\rho_{\mathrm{ii}}/\rho$")
    plt.xlabel(r"$t~[\mathrm{fs}]$", labelpad=0)

    ax1.set_xticks([0, 20, 40, 60, 80, 100])



    ### (b) ###############################################################################
    NF = 40
    factorFs = np.logspace(-0.5, 0.2, NF)

    ratios = {}
    for params in [params_sre, params_mre, params_dre]:

        ratios[params['rate_equation']] = []
        for i, factorF in enumerate(factorFs):
            mat = Material(index=1.5, drude_params={'damping':1e15}, 
                        ionization_params={'rate_equation':params['rate_equation'],
                                            'bandgap':9*eV,'alpha_sre':0.0004,
                                            'density':2e28,'cross_section':1e-19})
            las = Laser(wavelength=800*nm, phase=False, E0=params['E0']*factorF**2)
            dom = Domain()
            dom.add_laser(las, remove_reflected_part=False)
            dom.add_material(mat)
            dom.add_observer(Returner('rho'))
            dom.add_observer(Returner('rho_ii'))
            results = dom.run(100*fs, Nt=1000, progress_bar=False)
            ratios[params['rate_equation']].append((results['rho_ii']/results['rho']).max())


    ax2 = fig.add_subplot(122)
    ax2.plot(factorFs, ratios['sre'], label=r"$\mathrm{SRE}$", c="0.5", lw=2, zorder=5)
    ax2.plot(factorFs, ratios['mre'], label=r"$\mathrm{MRE}$", c="darkred", lw=2)
    ax2.plot(factorFs, ratios['dre'], label=r"$\mathrm{DRE}$", c=colors[2], lw=2, zorder=6)
    ax2.legend(loc=(0.5,0.2), fontsize=12, frameon=False)
    plt.xlim(factorFs.min(), factorFs.max())
    plt.xlabel(r"$F/F_\mathrm{av}$", labelpad=0)
    ax2.yaxis.tick_right()

    plt.yscale("log")
    plt.ylim(1e-3, 1e0)

    ax1.text(2, 0.46, r"$\mathrm{(a)}$")
    ax2.text(0.35, 0.55, r"$\mathrm{(b)}$")

    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.1)

    # plt.savefig("ratio.pdf")
    # os.system("pdfcrop ratio.pdf ratio.pdf > /dev/null")

    plt.show()