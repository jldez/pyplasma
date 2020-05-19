
from incubation import throw_continuous
from pyplasma import *
set_backend('torch.cuda')
import copy



def single_pulse(n, F, tau, trap_density, trapped_electron_density, recombination_rate, mask=None):

    dom = Domain(grid=[50,128,128], size=[1*um,3*um,3*um], pml_width=200*nm)

    laser = Laser(wavelength=800*nm, pulse_duration=tau, fluence=F, t0=tau, phase=True)
    dom.add_laser(laser, position='default', source_mode='tfsf', ramp=True)

    material = Material(index=1.45, resonance=120e-9,
                        drude_params={'damping':2e15},
                        ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28, 'alpha_sre':0.0004, 'recombination_rate':recombination_rate})

    dom.add_material(material, boundaries={'xmin':300*nm})

    if n == 0:
        surface_roughness(material, boundary='xmin', amplitude=20*nm, noise='fractal', feature_size=100*nm)
        mask = material.mask
    else:
        material.mask = mask

    trap = Trap(material, trap_energy_level, trap_density, trapping_rate, trap_recombination_rate)
    trap.trapped = trapped_electron_density

    # dom.add_observer(Watcher('E', x=400*nm, keep_pml=True, out_step=10))
    # dom.add_observer(Watcher('rho', x=400*nm, vlim=(0,2e28), out_step=10, loop=False, fourier=True, colormap='hot'))
    dom.add_observer(Watcher('rho', x=400*nm, out_step=5, loop=False))
    dom.add_observer(Returner('rho', out_step=-1, keep_pml=True))
    dom.add_observer(Returner('trapped_rho', out_step=-1, keep_pml=True))

    return dom.run(tau*1.5, stability_factor=0.9), mask




if __name__ == '__main__': 


    N = 15
    F = 3.5e4
    tau = 10*fs

    trap_energy_level = 3*eV
    trapping_rate = 1e12
    trap_recombination_rate = 0.0
    intrinsic_trap_density = 1e22
    extrinsic_trap_creation_probability = 0.1
    max_trap_density = 8e26

    recombination_rate = 1e12



    trapped_electron_density = 0.0
    extrinsic_trap_density = 0.0
    mask = None

    for n in range(N):

        trap_density = intrinsic_trap_density + extrinsic_trap_density

        results, mask = single_pulse(n, F, tau, trap_density, trapped_electron_density, recombination_rate, mask)

        # interpulse
        trapped_electron_density = results['trapped_rho']
        n_balls = results['rho']*(trapping_rate/recombination_rate)/(1+trapping_rate/recombination_rate)
        trapped_electron_density += throw_continuous(n_balls=n_balls, n_boxes=trap_density)

        # create new extrinsic traps
        max_rho = results['rho']
        new_traps = throw_continuous(n_balls=extrinsic_trap_creation_probability*max_rho, n_boxes=(max_trap_density - trap_density))
        extrinsic_trap_density += new_traps
        trapped_electron_density += new_traps
        trapped_electron_density = bd.array(trapped_electron_density)


    plt.show()