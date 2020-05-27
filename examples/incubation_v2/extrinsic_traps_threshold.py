
from pyplasma import *
# set_backend('torch.cuda')
set_backend('numpy')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
import tqdm
import os

from intrinsics_traps_incubation_0D import throw_continuous


if __name__ == "__main__":

    tau = 10 * fs

    intrinsic_trap_density = 1e25
    trap_energy_level = 3 * eV
    trapping_rate = 5e12
    trap_recombination_rate = 0e12

    trap_creation_rate = 0.2
    trap_density_max = 1e27

    Ns = []
    Fth = []

    tolerance = 0.05
    number_points = 7
    Nmax = 15
    Fmax = 1.8

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(1, Nmax)
    ax.set_ylim(0, Fmax)
    line = ax.semilogx([1,2], [0,0], marker='o')[0]
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    NN = np.unique(np.floor(np.logspace(0,np.log10(Nmax),number_points))).astype(int)
    for N in NN:
        F = Fmax
        Fmin = 0.1

        while abs(Fmax-Fmin) > tolerance:

            print((Fmax-Fmin)/tolerance)

            # initialize 
            trapped_electron_density = 0.0
            extrinsic_trap_density = 0.0

            # start multi pulses loop
            for n in range(N):

                # intrapulse
                dom = Domain()

                laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F*1e4, t0=0, phase=False)
                dom.add_laser(laser, remove_reflected_part=True)

                material = Material(index=1.45, resonance=120e-9,
                                    drude_params={'damping':2e15,'rho':0},
                                    ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':1e12})
                dom.add_material(material)

                trap_density = intrinsic_trap_density + extrinsic_trap_density
                trap = Trap(material, trap_energy_level, trap_density, trapping_rate, trap_recombination_rate)
                trap.trapped = trapped_electron_density

                dom.add_observer(Returner('rho', out_step=1))
                dom.add_observer(Returner('trapped_rho', out_step=-1))
                results = dom.run((-1*tau, 5*tau), Nt=40*tau/fs, progress_bar=False)

                # interpulse
                trapped_electron_density = float(results['trapped_rho'])
                n_balls = results['rho'][-1]*(trapping_rate/material.recombination_rate)/(1+trapping_rate/material.recombination_rate)
                trapped_electron_density += throw_continuous(n_balls=n_balls, n_boxes=trap_density-trapped_electron_density)

                extrinsic_trap_density += trap_creation_rate * results['rho'][-1] * (trap_density_max - trap_density)/(trap_density_max)

                threshold = c.epsilon_0*material.m_red*c.m_e/c.e**2*material.index**2*(laser.omega**2+material.damping**2)
                if results['rho'].max() >= threshold:
                    break

            if results['rho'].max() >= threshold:
                Fmax = F
            else:
                Fmin = F

            F = np.exp((np.log(Fmax)+np.log(Fmin))/2)

        # print(f'N:{N}, Fth:{F}')
        Ns.append(N)
        Fth.append(F)

        line.set_data(Ns, Fth)
        fig.canvas.draw()
        plt.pause(0.01)


    plt.show()

