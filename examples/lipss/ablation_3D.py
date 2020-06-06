
from pyplasma import *
set_backend('torch.cuda')
# set_backend('numpy')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 16})
import scipy.constants as c
import tqdm
import os
import argparse


def throw_continuous(n_balls, n_boxes):
    return n_boxes * (1.0 - np.exp(-n_balls/(n_boxes + 1.0)))

def ablated_cells(arr, threshold):
    n_cells = np.sum(arr >= threshold, axis=0)
    rem_cells = np.zeros(n_cells.shape)

    Nx, Ny, Nz = arr.shape
    for i in range(Ny):
        for j in range(Nz):
            if n_cells[i,j] > 0:
                for k in range(Nx-1):
                    if arr[k,i,j] > threshold and arr[k+1,i,j] < threshold:
                        rem_cells[i,j] = (arr[k,i,j]-threshold)/(arr[k,i,j]-arr[k+1,i,j]) - 0.5

    return n_cells, rem_cells

def ablate(mask, nb_ablated_cells, rem_cells):
    Nx, Ny, Nz = mask.shape
    for i in range(Ny):
        for j in range(Nz):
            n_cells = nb_ablated_cells[i,j]
            n_done = 0
            k = 0
            while n_done < n_cells:
                if mask[k,i,j] > 0:
                    mask[k,i,j] = 0.0
                    n_done += 1
                k += 1

            if rem_cells[i,j] < 0:
                mask[k-1,i,j] -= rem_cells[i,j]
            else:
                mask[k,i,j] -= rem_cells[i,j]
    return mask


def single_pulse(output,
                 F,
                 tau,
                 rho_star,
                 trap_energy_level, 
                 trapping_rate,
                 trap_recombination_rate,
                 trap_creation_rate,
                 intrinsic_trap_density,
                 last_results=None):

    domain = Domain(grid=[75,100,100], size=[1.5*um,2.0*um,2.0*um], pml_width=200*nm)
    laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F, t0=tau, phase=True)
    material = Material(index=1.45, resonance=120e-9,
                        drude_params={'damping':2e15},
                        ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':0e12})

    domain.add_laser(laser, position='default', source_mode='tfsf', ramp=True)
    domain.add_material(material, boundaries={'xmin':300*nm})

    if last_results is None:
        surface_map = surface_roughness(material, boundary='xmin', amplitude=30*nm, noise='fractal', feature_size=100*nm)
        trap_density = intrinsic_trap_density
        trapped_electron_density = 0.0
    else:
        last_results = np.load(last_results, allow_pickle=True).item()
        material.mask = bd.array(last_results['mask'])
        trap_density = last_results['trap_density']
        trapped_electron_density = last_results['trapped_electron_density']

    trap = Trap(material, trap_energy_level, trap_density, trapping_rate, trap_recombination_rate)
    if bd.is_any_array(trapped_electron_density):
        trapped_electron_density = bd.array(trapped_electron_density)
    trap.trapped = trapped_electron_density

    # domain.add_observer(Watcher('rho', y=1*um, z=1*um, out_step=3))
    domain.add_observer(Returner('rho', out_step=-1, keep_pml=True))
    domain.add_observer(Returner('trapped_rho', out_step=-1, keep_pml=True))
    results = domain.run(4*laser.pulse_duration, stability_factor=0.9)

    trapped_electron_density, trap_density, nb_ablated_cells, rem = interpulse(
        results['trapped_rho'],
        trapping_rate,
        material.recombination_rate,
        trap_density,
        trap_creation_rate,
        results['rho'],
        material.density,
        rho_star,
    )

    mask = ablate(np.array(material.mask.cpu()), nb_ablated_cells, rem)

    results['mask'] = mask
    results['trap_density'] = trap_density
    results['trapped_electron_density'] = trapped_electron_density
    
    np.save(output, results)
    

    thickness = np.sum(mask, axis=0)*domain.dx
    plt.imshow(thickness/um, cmap='Blues_r', extent=[-domain.Ly/2/um,domain.Ly/2/um,-domain.Lz/2/um,domain.Lz/2/um])
    plt.colorbar(fraction=0.046, pad=0.04, format='%.2f')
    plt.savefig(output[:-4] + '.pdf')



def interpulse(trapped_electron_density, 
               trapping_rate, 
               recombination_rate, 
               trap_density, 
               trap_creation_rate, 
               rho, 
               material_density, 
               rho_star):

    n_balls = rho*(trapping_rate/(recombination_rate+1.))/(1+trapping_rate/(recombination_rate+1.))
    new_trapped = throw_continuous(n_balls=n_balls, n_boxes=trap_density-trapped_electron_density)
    trapped_electron_density += new_trapped
    # trapped_electron_density *= np.exp(-trap_recombination_rate/repetition_rate)

    new_traps = throw_continuous(n_balls=trap_creation_rate * (rho - new_trapped), n_boxes=material_density - trap_density)
    trap_density += new_traps

    nb_ablated_cells, rem = ablated_cells(rho, rho_star)

    return trapped_electron_density, trap_density, nb_ablated_cells, rem




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o')
    parser.add_argument('-F', type=float)
    parser.add_argument('-t', type=float)
    parser.add_argument('-rs', type=float)
    parser.add_argument('-tel', type=float)
    parser.add_argument('-tr', type=float)
    parser.add_argument('-trr', type=float)
    parser.add_argument('-tcr', type=float)
    parser.add_argument('-itd', type=float)
    parser.add_argument('-lr', default=None)
    args = parser.parse_args()

    single_pulse(
        args.o, 
        args.F, 
        args.t, 
        args.rs, 
        args.tel, 
        args.tr, 
        args.trr, 
        args.tcr, 
        args.itd, 
        args.lr, 
    )

