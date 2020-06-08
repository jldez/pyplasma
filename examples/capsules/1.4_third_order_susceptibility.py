
from pyplasma import *

domain = Domain(grid=[1300], size=[20*um], pml_width=400*nm)

laser = Laser(wavelength=800*nm, pulse_duration=10*fs, t0=20*fs, E0=5e9, phase=True)
domain.add_laser(laser)

material = Material(index=1.33, chi3=1e-19, resonance=120e-9)
domain.add_material(material, boundaries={'xmin':10*um})

w = Watcher('Ez', vlim=(-1.2*laser.E0,1.2*laser.E0), c='r', out_step=8, loop=True, figsize=(9,3), keep_pml=False)
domain.add_observer(w)

results = domain.run(80*fs)
