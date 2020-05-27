
from pyplasma import *
# set_backend('torch.cuda')
set_backend('numpy')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
import tqdm
import os



def throw_continuous(n_balls, n_boxes):
    if np.max(n_boxes) > 0:
        return n_boxes * (1.0 - np.exp(-n_balls/n_boxes))
    else:
        return 0

def throw_discrete(n_balls, n_boxes):
    box_states = np.zeros(n_boxes)
    for ball in range(n_balls):
        box_states[np.random.randint(0, n_boxes)] = 1
    return np.sum(box_states)

# print(throw_continuous(10000,10000))
# print(throw_discrete(10000,10000))



if __name__ == "__main__":

    tau = 100 * fs
    F = 2.0 * 1e4
    N = 5

    intrinsic_trap_density = 1e26
    trap_energy_level = 3 * eV
    trapping_rate = 5e12
    trap_recombination_rate = 0e12

    trapped_electron_density = 0.0


    for n in range(N):

        dom = Domain()

        laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F, t0=0, phase=False)
        dom.add_laser(laser, remove_reflected_part=True)

        material = Material(index=1.45, resonance=120e-9,
                            drude_params={'damping':2e15,'rho':0},
                            ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':1e12})
        dom.add_material(material)

        trap_density = intrinsic_trap_density
        trap = Trap(material, trap_energy_level, trap_density, trapping_rate, trap_recombination_rate)
        trap.trapped = trapped_electron_density

        dom.add_observer(Watcher('rho', vlim=(0,1e27), out_step=30))
        dom.add_observer(Watcher('trapped_rho', vlim=(0,intrinsic_trap_density), out_step=30))
        dom.add_observer(Returner('rho', out_step=-1))
        dom.add_observer(Returner('trapped_rho', out_step=-1))
        results = dom.run((-1.5*tau, 5*tau), Nt=40*tau/fs, progress_bar=False)

        trapped_electron_density = float(results['trapped_rho'])
        n_balls = results['rho']*(trapping_rate/material.recombination_rate)/(1+trapping_rate/material.recombination_rate)
        trapped_electron_density += throw_continuous(n_balls=n_balls, n_boxes=trap_density-trapped_electron_density)

    plt.show()