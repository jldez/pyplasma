from pyplasma import *
import scipy.constants as c



def example_3D():
  set_backend('torch.cuda')

  time = Time(start=0, end=40*fs, Nt=3e3)

  dom = Domain(grid=[100,250,250], size=[1*um,5*um,5*um], pml_width=200*nm)

  laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=4.7e4, t0=20*fs, phase=True)
  dom.add_laser(laser, position='default')

  material = Material(
                    index=1.45, resonance=120e-9, chi3=2e-22,
                    drude_params={'damping':1e15, 'rho':0e27, 'm_VB':4},
                    ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28,'alpha_sre':0.001}
                    # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    # ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material, boundaries={'xmin':300*nm}, roughness=40*nm)

  dom.add_observer(Observer('Ez', 'watch', x=420*nm, vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=1))
  dom.add_observer(Observer('Ez', 'watch', y=1*um, z=1*um, vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=1))
  dom.add_observer(Observer('rho', 'watch', x=420*nm, vlim=(0,material.density), out_step=1))

  results = dom.run(time)



def example_1D():
  set_backend('numpy')

  time = Time(start=0, end=50*fs, Nt=1e3)

  dom = Domain(grid=[100], size=[5*um], pml_width=400*nm)

  laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=4.7e4, t0=25*fs, phase=True)
  dom.add_laser(laser, position='default')

  material = Material(
                    index=1.45, resonance=120e-9, chi3=2e-22,
                    drude_params={'damping':1e15, 'rho':0e27, 'm_VB':4},
                    ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2.2e28,'alpha_sre':0.001}
                    # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    # ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material, boundaries={'xmin':1*um})

  dom.add_observer(Observer('Ez', 'watch', vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
  dom.add_observer(Observer('rho', 'watch', vlim=(0, material.density/5), out_step=5))

  results = dom.run(time)



def example_0D():
  set_backend('numpy')

  time = Time(start=0, end=50*fs, Nt=1e3)

  dom = Domain()

  laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=5e4, t0=25*fs, phase=True)
  dom.add_laser(laser, remove_reflected_part=True)

  material = Material(
                    index=1.4, 
                    drude_params={'damping':1e15, 'rho':0e27},
                    # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2e28,'alpha_sre':0.004}
                    # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material)

  dom.add_observer(Observer('E', 'watch', vlim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
  dom.add_observer(Observer('rho', 'watch', vlim=(0,0), out_step=5))

  dom.run(time)



if __name__ == '__main__':  
  # example_0D()
  # example_1D()
  example_3D()