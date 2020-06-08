
from pyplasma import *
# set_backend('torch.cuda')
set_backend('numpy')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
import tqdm
import os



if __name__ == "__main__":

    tau = 100 * fs
    F = 2.3 * 1e4

    intrinsic_trap_density = 1e25
    trap_energy_level = 3 * eV
    trapping_rate = 5e12
    trap_recombination_rate = 1e12


    dom = Domain()

    laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F, t0=0, phase=False)
    dom.add_laser(laser, remove_reflected_part=True)

    material = Material(index=1.45, resonance=120e-9,
                        drude_params={'damping':2e15,'rho':0},
                        ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':1e12})
    dom.add_material(material)

    trap = Trap(material, trap_energy_level, intrinsic_trap_density, trapping_rate, trap_recombination_rate)

    dom.add_observer(Watcher('rho', vlim=(0,1e25), out_step=30))
    dom.add_observer(Watcher('trapped_rho', vlim=(0, intrinsic_trap_density), out_step=30, c='C1'))
    results = dom.run((-1.5*tau, 5*tau), Nt=40*tau/fs, progress_bar=False)

    plt.show()