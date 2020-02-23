from pyplasma import *
set_backend('numpy')


if __name__ == '__main__': 

    dom = Domain(grid=[100], size=[3*um], pml_width=200*nm)

    laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=7e4, t0=20*fs, phase=True)
    dom.add_laser(laser, position='default', source_mode='tfsf')

    material = Material(index=1.45,
                        drude_params={'damping':1e15, 'm_VB':1},
                        ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19})
    dom.add_material(material, boundaries={'xmin':1*um})

    dom.add_observer(Watcher('Ez', vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
    dom.add_observer(Watcher('rho', vlim=(0, material.density), out_step=5))

    results = dom.run(40*fs)



