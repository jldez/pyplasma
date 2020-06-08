from pyplasma import *
try:
    set_backend('torch.cuda')
except: pass


if __name__ == '__main__': 

    dom = Domain(grid=[50,128,128], size=[1*um,3*um,3*um], pml_width=200*nm)

    laser = Laser(wavelength=800*nm, pulse_duration=20*fs, fluence=10e4, t0=20*fs, phase=True)
    dom.add_laser(laser, position='default', source_mode='tfsf', ramp=True)

    material = Material(index=1.45, resonance=120e-9, chi3=2e-22,
                        drude_params={'damping':1e15},
                        ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                        )

    dom.add_material(material, boundaries={'xmin':300*nm})
    surface_roughness(material, boundary='xmin', amplitude=20*nm, noise='fractal', feature_size=100*nm, show=False)

    dom.add_observer(Watcher('rho', x=400*nm, vlim=(0,2e28), out_step=10, loop=False, fourier=True, colormap='hot'))
    dom.add_observer(Watcher('rho', x=400*nm, out_step=10, loop=True))

    results = dom.run(30*fs, stability_factor=0.9)