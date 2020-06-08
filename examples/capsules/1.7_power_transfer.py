
from pyplasma import *

domain = Domain(grid=[300], size=[10*um], pml_width=400*nm)

laser = Laser(wavelength=800*nm, pulse_duration=10*fs, t0=20*fs, E0=1e9, phase=True)
domain.add_laser(laser)

material = Material(index=1.3, drude_params={'damping':2e15, 'rho':5e25})
domain.add_material(material, boundaries={'xmin':2*um})

w1 = Watcher('Ez', vlim=(-1.2*laser.E0,1.2*laser.E0), c='r', out_step=3, figsize=(9,3), keep_pml=False)
w2 = Watcher('Powerz', vlim=(-2e20,6e20), c='k', out_step=3, loop=True, figsize=(9,3), keep_pml=False)
domain.add_observer([w1,w2])

results = domain.run(80*fs)
