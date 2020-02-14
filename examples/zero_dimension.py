
from pyplasma import *
set_backend('numpy')


if __name__ == '__main__':  

    time = Time(start=0, end=40*fs, Nt=1e3)

    dom = Domain()

    laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=7e4, t0=20*fs, phase=True)
    dom.add_laser(laser, remove_reflected_part=True)

    material = Material(index=1.4, drude_params={'damping':1e15},
                        ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19})
    dom.add_material(material)

    dom.add_observer(Watcher('E', vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
    dom.add_observer(Watcher('rho', vlim=(0,material.density), out_step=5))

    dom.run(time)
