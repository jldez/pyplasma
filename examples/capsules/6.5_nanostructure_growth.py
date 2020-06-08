
from scripts.ablation_3D import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 16})
import scipy.constants as c
import tqdm
import os

from pyplasma import *
try:
    set_backend('torch.cuda')
except: pass

FILENAME = os.path.dirname(os.path.abspath(__file__))



if __name__ == "__main__":

    tau = 10 * fs
    F = 4.0 * 1e4
    N = 12
    repetition_rate = 1e3

    intrinsic_trap_density = 1e25
    trap_energy_level = 3 * eV
    trapping_rate = 5e12
    trap_recombination_rate = 0e3
    trap_creation_rate = 0.1

    rho_star = 1.5e27

    trap_density = intrinsic_trap_density
    trapped_electron_density = 0.0
    extrinsic_trap_density = 0.0
    mask = None
    for n in range(N):

        last_results = f'{FILENAME}/growth_{n-1}.npy' if n>0 else None

        command = f'python3 {FILENAME}/scripts/ablation_3D.py '
        command += f'-o={FILENAME}/growth_{n}.npy '
        command += f'-F={F} '
        command += f'-t={tau} '
        command += f'-rs={rho_star} '
        command += f'-tel={trap_energy_level} '
        command += f'-tr={trapping_rate} '
        command += f'-trr={trap_recombination_rate} '
        command += f'-tcr={trap_creation_rate} '
        command += f'-itd={intrinsic_trap_density} '
        if n > 0:
            command += f'-lr={last_results} '

        os.system(command)
    



    fig = plt.figure(figsize=(14,16))

    domain = Domain(grid=[60,100,100], size=[1.2*um,2.0*um,2.0*um], pml_width=200*nm)

    for n in range(N):

        ax = fig.add_subplot(4,3,n+1)

        results = np.load(f'{FILENAME}/growth_{n}.npy', allow_pickle=True).item()

        surface_map = np.sum(results['mask'], axis=0)*domain.dx/nm -1200

        if n > 5:
            vmin = np.percentile(surface_map, 2)
            vmax = np.percentile(surface_map, 98)
        else: 
            vmin = surface_map.min()
            vmax = surface_map.max()
        im = ax.imshow(surface_map, cmap='Blues_r', extent=[-domain.Ly/2/um,domain.Ly/2/um,-domain.Lz/2/um,domain.Lz/2/um], vmin=vmin, vmax=vmax)

        fig.colorbar(im, fraction=0.046, pad=0.04, format='%d')

        if n in [0,1,2,3,4,5,6,7,8]:
            ax.get_xaxis().set_visible(False)
        if n in [1,2,4,5,7,8,10,11]:
            ax.get_yaxis().set_visible(False)

        ax.set_title(r'$N=$' + f' {n+1}')
        ax.set_xlabel(r'x~[$\mu$m]')
        ax.set_ylabel(r'y~[$\mu$m]')


    plt.tight_layout()
    plt.show()