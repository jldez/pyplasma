from pyplasma import *
set_backend('torch.cuda')


if __name__ == '__main__': 

    trap_density = 5e26
    xi = 0.2
    rho_trapped = 0

    for n in range(10):

        dom = Domain(grid=[50,128,128], size=[1*um,3*um,3*um], pml_width=200*nm)

        laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=3.5e4, t0=15*fs, phase=True)
        dom.add_laser(laser, position='default', source_mode='tfsf', ramp=True)

        material = Material(index=1.45, resonance=120e-9, chi3=2e-22,
                            drude_params={'damping':1e15,'rho':rho_trapped},
                            ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28,'alpha_sre':0.001,'fi_mode':'fit'}
                            # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                            # ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                            )            

        dom.add_material(material, boundaries={'xmin':300*nm})

        if n == 0:
            surface_roughness(material, boundary='xmin', amplitude=20*nm, noise='fractal', feature_size=100*nm, show=False)
            mask = material.mask
        else:
            material.mask = mask

        dom.add_observer(Watcher('rho', x=400*nm, vlim=(0,material.density), out_step=10, loop=False))
        dom.add_observer(Returner('rho', out_step=-1, keep_pml=True))

        final_rho = dom.run(25*fs, stability_factor=0.9)['rho']

        rho_trapped = trap_density*(1-np.exp(-xi*final_rho/trap_density))

    