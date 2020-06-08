
from pyplasma import *
set_backend('numpy')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 24})
import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



def throw_continuous(n_balls, n_boxes):
    if np.max(n_boxes) > 0:
        return n_boxes * (1.0 - np.exp(-n_balls/n_boxes))
    else:
        return 0



if __name__ == "__main__":

    fig = plt.figure(figsize=(16,5))

    tau = 100 * fs
    F = 1.5 * 1e4
    N = 35
    repetition_rate = 1e3

    intrinsic_trap_density = 1e25
    trap_energy_level = 3 * eV
    trapping_rate = 5e12
    trap_recombination_rate = 0e3
    trap_creation_rate = 0.1
    trap_density_max = 1e26


    time = np.zeros(0)
    splits = []
    rho = np.zeros(0)
    trapped_rho = np.zeros(0)
    trap_density_tracking = np.zeros(0)

    trapped_electron_density = 0.0
    extrinsic_trap_density = 0.0
    for n in tqdm.tqdm(range(N)):
        plt.clf()

        dom = Domain()

        laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F, t0=0, phase=False)
        dom.add_laser(laser, remove_reflected_part=True)

        material = Material(index=1.45, resonance=120e-9,
                            drude_params={'damping':2e15,'rho':0},
                            ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':1e12, 'fi_table_size':100})
        dom.add_material(material)

        trap_density = intrinsic_trap_density + extrinsic_trap_density
        trap = Trap(material, trap_energy_level, trap_density, trapping_rate, trap_recombination_rate)
        trap.trapped = trapped_electron_density

        dom.add_observer(Returner('rho'))
        dom.add_observer(Returner('trapped_rho'))
        results = dom.run((-1.5*tau, 5*tau), Nt=20*tau/fs, progress_bar=False)

        time = np.append(time, dom.times + n*(dom.Nt*dom.dt + 0/repetition_rate))
        splits.append(time[-1])
        rho = np.append(rho, results['rho'])
        trapped_rho = np.append(trapped_rho, results['trapped_rho'])
        trap_density_tracking = np.append(trap_density_tracking, np.full(dom.times.shape, trap_density))

        trapped_electron_density = float(results['trapped_rho'][-1])
        n_balls = results['rho'][-1]*(trapping_rate/material.recombination_rate)/(1+trapping_rate/material.recombination_rate)
        new_trapped = throw_continuous(n_balls=n_balls, n_boxes=trap_density-trapped_electron_density)
        trapped_electron_density += new_trapped
        trapped_electron_density *= np.exp(-trap_recombination_rate/repetition_rate)

        new_traps = throw_continuous(n_balls=trap_creation_rate * (results['rho'][-1] - new_trapped), n_boxes=trap_density_max - trap_density)
        extrinsic_trap_density += new_traps

        for split in splits:
            plt.axvline(split/ps, c='k', ls='--', alpha=0.25)
        plt.plot(time/ps, rho, lw=2, label=r'$\rho$')
        plt.plot(time/ps, trapped_rho, lw=2, label=r'$\rho_t$')
        plt.xlim((time/ps).min(), (time/ps).max())
        plt.legend(loc=2)
        plt.ylabel('Densité')
        plt.xlabel(r'$t~[\mathrm{ps}]$')
        plt.tight_layout()

        plt.show(block=False)
        plt.pause(0.01)

    plt.show()
