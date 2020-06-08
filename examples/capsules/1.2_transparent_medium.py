
from pyplasma import *

domain = Domain(grid=[500], size=[20*um], pml_width=400*nm)

laser = Laser(wavelength=800*nm, pulse_duration=10*fs, t0=20*fs, E0=1, phase=True)
domain.add_laser(laser)

material = Material(index=1.5)
domain.add_material(material, boundaries={'xmin':10*um})

w = Watcher('Ez', vlim=(-1.1,1.1), c='r', out_step=4, loop=True, figsize=(9,3), keep_pml=False)
domain.add_observer(w)

results = domain.run(120*fs)
