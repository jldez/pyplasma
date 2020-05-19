
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


def incubation(args): 

    trap_energy_level, trapping_rate, trap_recombination_rate, intrinsic_trap_density, extrinsic_trap_creation_probability, max_trap_density = args

    filename = f'Et{trap_energy_level}_b{trapping_rate*fs}_g{trap_recombination_rate*fs}_i{intrinsic_trap_density}_a{extrinsic_trap_creation_probability}_m{max_trap_density}'
    print(f'filename: {filename}')

    Ns = []
    Fth = []
    tau = 10*fs

    tolerance = 0.05
    number_points = 15
    Nmax = 50
    Fmax = 1.8

    fig = plt.figure()
    fig.canvas.set_window_title(filename)
    ax = fig.add_subplot(111)
    ax.set_xlim(1, Nmax)
    ax.set_ylim(0, Fmax)
    line = ax.semilogx([1,2], [0,0], marker='o')[0]
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

    NN = np.unique(np.floor(np.logspace(0,np.log10(Nmax),number_points))).astype(int)
    try:
        for N in NN:
            F = Fmax
            Fmin = 0.1

            while abs(Fmax-Fmin) > tolerance:

                # print((Fmax-Fmin)/tolerance)

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
                    trap = Trap(material, trap_energy_level*eV, trap_density, trapping_rate, trap_recombination_rate)
                    trap.trapped = trapped_electron_density

                    dom.add_observer(Returner('rho', out_step=1))
                    dom.add_observer(Returner('trapped_rho', out_step=1))
                    results = dom.run((-2*tau, 2*tau), Nt=20*tau/fs, progress_bar=False)

                    # interpulse
                    trapped_electron_density = results['trapped_rho'][-1]
                    n_balls = results['rho'][-1]*(trapping_rate/material.recombination_rate)/(1+trapping_rate/material.recombination_rate)
                    trapped_electron_density += throw_continuous(n_balls=n_balls, n_boxes=trap_density)

                    # create new extrinsic traps
                    max_rho = results['rho'].max() 
                    new_traps = throw_continuous(n_balls=extrinsic_trap_creation_probability*max_rho, n_boxes=(max_trap_density - trap_density))
                    extrinsic_trap_density += new_traps
                    trapped_electron_density += new_traps

                    threshold = c.epsilon_0*material.m_red*c.m_e/c.e**2*material.index**2*(laser.omega**2+material.damping**2)
                    if max_rho >= threshold:
                        break

                if max_rho >= threshold:
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

    except KeyboardInterrupt:
        pass

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlim(1, Nmax)
    # ax.set_ylim(0, Fmax)
    # line = ax.semilogx([1,2], [0,0], marker='o')[0]

    filepath = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(f'{filepath}/{filename}.pdf')
    np.save(f'{filepath}/{filename}.npy', {'Ns':Ns, 'Fth':Fth})
    plt.close()
    return 0


if __name__ == "__main__":
    
    incubation([3.0, 1e12, 0e12, 1e22, 0.1, 8e26])
    # incubation([3.0, 1e12, 1e12, 8e26, 0.0, 8e26])

