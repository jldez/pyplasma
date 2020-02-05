from pyplasma import *
import scipy.constants as c



def example_3D():
  set_backend('torch.cuda')

  time = Time(start=0, end=50*fs, Nt=1e3)

  dom = Domain(grid=[100,100,100], size=[5*um,5*um,5*um], pml_width=400*nm)

  laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=1e5, t0=25*fs, phase=True)
  dom.add_laser(laser, position='default')

  material = Material(
                    index=1.4, resonance=120e-9, 
                    drude_params={'damping':1e15, 'rho':0e27},
                    # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2e28,'alpha_sre':0.004}
                    # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material, boundaries={'xmin':1*um})

  dom.add_observer(Observer('Ez', 'watch', y=2*um, z=2*um, ylim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=2))
  dom.add_observer(Observer('rho', 'watch', y=2*um, z=2*um, ylim=(0,material.density), out_step=2))

  results = dom.run(time)



def example_1D():
  set_backend('numpy')

  time = Time(start=0, end=50*fs, Nt=1e3)

  dom = Domain(grid=[100], size=[5*um], pml_width=400*nm)

  laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=1e5, t0=25*fs, phase=True)
  dom.add_laser(laser, position='default')

  material = Material(
                    index=1.4, resonance=120e-9, 
                    drude_params={'damping':1e15, 'rho':0e27},
                    # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2e28,'alpha_sre':0.004}
                    # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material, boundaries={'xmin':1*um})

  dom.add_observer(Observer('Ez', 'watch', ylim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
  dom.add_observer(Observer('rho', 'watch', ylim=(0, material.density), out_step=5))

  results = dom.run(time)



def example_0D():
  set_backend('numpy')

  time = Time(start=0, end=50*fs, Nt=1e3)

  dom = Domain()

  laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=1e5, t0=25*fs, phase=True)
  dom.add_laser(laser, remove_reflected_part=True)

  material = Material(
                    index=1.4, 
                    drude_params={'damping':1e15, 'rho':0e27},
                    # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2e28,'alpha_sre':0.004}
                    ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    # ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material)

  dom.add_observer(Observer('E', 'watch', ylim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
  dom.add_observer(Observer('rho', 'watch', ylim=(0,0), out_step=5))

  dom.run(time)



if __name__ == '__main__':  
  # example_0D()
  example_1D()
  # example_3D()