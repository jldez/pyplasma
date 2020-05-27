
from pyplasma import *
set_backend('torch.cuda')
# set_backend('numpy')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
import tqdm
import os


def throw_continuous(n_balls, n_boxes):
    return n_boxes * (1.0 - bd.exp(-n_balls/(n_boxes + 1.0)))


def ablated_cells(arr, threshold):
    # Nx, Ny, Nz = arr.shape
    # ablated_cells = np.zeros((Ny, Nz))
    # for i in range(Ny):
    #     for j in range(Nz):
    #         for k in range(Nx):
    #             if arr[k,i,j] >= threshold:
    #                 ablated_cells[i,j] += 1
    return np.sum(arr >= threshold, axis=0)



if __name__ == "__main__":

    tau = 10 * fs
    F = 3.4 * 1e4
    N = 20

    intrinsic_trap_density = 1e25
    trap_energy_level = 3 * eV
    trapping_rate = 5e12
    trap_recombination_rate = 0e12

    trap_creation_rate = 0.1
    trap_density_max = 3e28

    rho_star = 5e26



    extrinsic_trap_density = 0.0
    trapped_electron_density = 0.0

    for n in range(N):

        dom = Domain(grid=[50,150,150], size=[1*um,3*um,3*um], pml_width=200*nm)

        laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F, t0=tau, phase=True)
        dom.add_laser(laser, position='default', source_mode='tfsf', ramp=True)

        material = Material(index=1.45, resonance=120e-9,
                            drude_params={'damping':2e15},
                            ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':1e12})
        dom.add_material(material, boundaries={'xmin':300*nm})

        if n == 0:
            surface_map = surface_roughness(material, boundary='xmin', amplitude=20*nm, noise='fractal', feature_size=100*nm)
        else:
            surface_roughness(material, boundary='xmin', roughness_map=surface_map)

        trap_density = material.mask*(intrinsic_trap_density + extrinsic_trap_density)
        trap = Trap(material, trap_energy_level, trap_density, trapping_rate, trap_recombination_rate)
        trap.trapped = trapped_electron_density

        dom.add_observer(Watcher('rho', x=400*nm, out_step=5, loop=False))
        dom.add_observer(Returner('rho', out_step=-1, keep_pml=True))
        dom.add_observer(Returner('trapped_rho', out_step=-1, keep_pml=True))
        results = dom.run(tau*2.0, stability_factor=0.9)

        trapped_electron_density = bd.array(results['trapped_rho'])
        n_balls = bd.array(results['rho']*(trapping_rate/material.recombination_rate)/(1+trapping_rate/material.recombination_rate))
        trapped_electron_density += throw_continuous(n_balls=n_balls, n_boxes=trap_density-trapped_electron_density)

        extrinsic_trap_density += bd.array(trap_creation_rate * results['rho']) * (trap_density_max - trap_density)/(trap_density_max)

        # ablation_depth = dom.dx*ablated_cells(results['rho'], rho_star)
        # print(np.mean(ablation_depth)/nm, np.max(ablation_depth)/nm, np.max(results['rho'])/rho_star)
        # surface_map -= ablation_depth
        # surface_map -= surface_map.min()

        # plt.imshow(surface_map/nm)
        # plt.imshow(np.sum(np.array(material.mask.cpu()),axis=0))
        # plt.show(block=False)
        # plt.pause(0.05)

    plt.show()