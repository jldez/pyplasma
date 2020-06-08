
from pyplasma import *
# set_backend('torch.cuda')
set_backend('numpy')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 16})
import scipy.constants as c
import tqdm
import os


def throw_continuous(n_balls, n_boxes):
    return n_boxes * (1.0 - bd.exp(-n_balls/(n_boxes + 1.0)))

def ablated_cells(arr, threshold):
    return np.sum(arr >= threshold, axis=0)



if __name__ == "__main__":

    fig = plt.figure(figsize=(6,12))

    F = 1.4e4
    tau = 10 * fs
    N = 10
    repetition_rate = 1e3

    intrinsic_trap_density = 1e25
    trap_energy_level = 3 * eV
    trapping_rate = 5e12
    trap_recombination_rate = 0e3
    trap_creation_rate = 0.1

    rho_star = 1.5e27

    saved_results = []

    trapped_electron_density = 0.0
    extrinsic_trap_density = 0.0
    surface_position = 500*nm
    for n in tqdm.tqdm(range(N)):

        dom = Domain(grid=[100,3,3], size=[2.5*um,75*nm,75*nm], pml_width=400*nm)

        laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=2*F, t0=tau, phase=False)
        dom.add_laser(laser, position='default', source_mode='tfsf', ramp=True)

        material = Material(index=1.45, resonance=120e-9,
                            drude_params={'damping':2e15},
                            ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':1e12, 'fi_table_size':200})
        dom.add_material(material, boundaries={'xmin':surface_position})

        trap_density = intrinsic_trap_density + extrinsic_trap_density
        trap = Trap(material, trap_energy_level, trap_density, trapping_rate, trap_recombination_rate)
        trap.trapped = trapped_electron_density*material.mask

        dom.add_observer(Returner('rho', out_step=-1, keep_pml=True))
        dom.add_observer(Returner('trapped_rho', out_step=-1, keep_pml=True))
        results = dom.run(2*tau, verbose=False)

        trapped_electron_density = results['trapped_rho']
        n_balls = results['rho']*(trapping_rate/(material.recombination_rate+1.))/(1+trapping_rate/(material.recombination_rate+1.))
        new_trapped = throw_continuous(n_balls=n_balls, n_boxes=trap_density-trapped_electron_density)
        trapped_electron_density += new_trapped
        trapped_electron_density *= np.exp(-trap_recombination_rate/repetition_rate)

        new_traps = throw_continuous(n_balls=trap_creation_rate * (results['rho'] - new_trapped), n_boxes=material.density - trap_density)
        extrinsic_trap_density += new_traps

        nb_ablated_cells = ablated_cells(results['rho'], rho_star)[0,0]
        ablation_depth = dom.dx*nb_ablated_cells
        surface_position += ablation_depth

        results['ablation_depth'] = ablation_depth
        results['trap_density'] = trap_density
        saved_results.append(results)



    for n in range(len(saved_results)):
        ax = fig.add_subplot(len(saved_results), 1, n+1)

        no_zero = np.where(saved_results[n]['trapped_rho'][:,1,1]>0)[0]
        ax.plot(dom.x[no_zero]/um, saved_results[n]['rho'][no_zero,1,1]/rho_star, label=r'$\rho/\rho^\ast$')
        if n == 0:
            ax.axhline(saved_results[n]['trap_density']/rho_star, label=r'$D_t/\rho^\ast$', c='C1')
        else:
            ax.plot(dom.x[no_zero]/um, saved_results[n]['trap_density'][no_zero,1,1]/rho_star, label=r'$D_t/\rho^\ast$', c='C1')
        ax.set_ylim(0, 2)
        ax.set_xlim(0.0, 2.0)
        ax.axhline(1, ls='--', alpha=0.6, c='r', label=r'$\rho^\ast/\rho^\ast$')
        ax.axvline(dom.x[no_zero][0]/um, ls=':', alpha=0.6, c='b', label='Surface')
        ax.add_patch(patches.Rectangle((surface_position/um,-np.inf), 2.5*um, np.inf, linewidth=1.5, edgecolor='0.8', facecolor='0.9'))

        if n != len(saved_results)-1:
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        if n == len(saved_results)-1:
            ax.set_xlabel(r'$z$~[$\mu$m]')
        if n == 0:
            ax.set_title(r'$F =$' + f' {F/1e4:0.1f} ' + r'[J/cm$^2$]', fontsize=16)

        ax.text(0.03, 1.6, r'N = '+f'{n+1}')


    plt.subplots_adjust(hspace=0, wspace=0.1)
    plt.show()