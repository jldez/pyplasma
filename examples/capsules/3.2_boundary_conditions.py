
from pyplasma import *

domain = Domain(grid=[500], size=[20*um], pml_width=3200*nm)

laser = Laser(wavelength=800*nm, pulse_duration=10*fs, t0=20*fs, E0=1, phase=True)
domain.add_laser(laser, source_mode='hard')

material = Material(index=1.5)
domain.add_material(material, boundaries={'xmin':10*um})

w = Watcher('Ez', vlim=(-1.5,1.5), c='r', out_step=6, loop=True, figsize=(9,3), keep_pml=True)
domain.add_observer(w)

results = domain.run(200*fs)
