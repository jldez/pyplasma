from pyplasma import *
set_backend('torch.cuda')


if __name__ == '__main__': 

    dom = Domain(grid=[100,100,100], size=[3*um,3*um,3*um], pml_width=200*nm)

    laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=7e4, t0=20*fs, phase=True)
    dom.add_laser(laser, position='default', source_mode='tfsf')

    material = Material(
                        index=1.45, resonance=120e-9, chi3=2e-22,
                        drude_params={'damping':1e15},
                        # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28,'alpha_sre':0.001}
                        # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                        ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                        )

    dom.add_material(material, boundaries={'xmin':1*um,'xmax':2*um})

    dom.add_observer(Watcher('Ez', z=1.5*um, vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=False, out_step=10))
    dom.add_observer(Watcher('rho', z=1.5*um, vlim=(0, material.density), keep_pml=False, out_step=10))
    dom.add_observer(Watcher('Ez', y=1.5*um, z=1.5*um, vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=10))
    dom.add_observer(Watcher('rho', y=1.5*um, z=1.5*um, vlim=(0, material.density), keep_pml=True, out_step=10))

    results = dom.run(40*fs)