import pyplasma as pp
import scipy.constants as c
from pyplasma.simulation import Domain, Time



def example_3D():
  pp.set_backend('torch.cuda')

  time = Time(start=0, end=50e-15, Nt=1e3)

  dom = Domain(grid=[100,100,100], size=[5000e-9,5000e-9,5000e-9], pml_width=400e-9)

  laser = pp.Laser(wavelength=800e-9, pulse_duration=10e-15, fluence=1e5, t0=25e-15, phase=True)
  dom.add_laser(laser, position='default')

  material = pp.Material(
                    index=1.4, resonance=120e-9, 
                    drude_params={'damping':1e15, 'rho':0e27},
                    # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2e28,'alpha_sre':0.004}
                    # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material, boundaries={'xmin':1.0e-6})

  dom.add_observer(pp.Observer('Ez', 'watch', y=2e-6, z=2e-6, ylim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=2))
  dom.add_observer(pp.Observer('rho', 'watch', y=2e-6, z=2e-6, ylim=(0,material.density), out_step=2))

  results = dom.run(time)



def example_1D():
  pp.set_backend('numpy')

  time = Time(start=0, end=50e-15, Nt=1e3)

  dom = Domain(grid=[100], size=[5000e-9], pml_width=400e-9)

  laser = pp.Laser(wavelength=800e-9, pulse_duration=10e-15, fluence=1e5, t0=25e-15, phase=True)
  dom.add_laser(laser, position='default')

  material = pp.Material(
                    index=1.4, resonance=120e-9, 
                    drude_params={'damping':1e15, 'rho':0e27},
                    # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2e28,'alpha_sre':0.004}
                    # ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material, boundaries={'xmin':1.0e-6})

  dom.add_observer(pp.Observer('Ez', 'watch', ylim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
  dom.add_observer(pp.Observer('rho', 'watch', ylim=(0,material.density), out_step=5))

  results = dom.run(time)



def example_0D():
  pp.set_backend('numpy')

  time = Time(start=0, end=50e-15, Nt=1e3)

  dom = Domain()

  laser = pp.Laser(wavelength=800e-9, pulse_duration=10e-15, fluence=1e5, t0=25e-15, phase=True)
  dom.add_laser(laser, remove_reflected_part=True)

  material = pp.Material(
                    index=1.4, 
                    drude_params={'damping':1e15, 'rho':0e27},
                    # ionization_params={'rate_equation':'sre','bandgap':9.*c.e,'density':2e28,'alpha_sre':0.004}
                    ionization_params={'rate_equation':'mre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    # ionization_params={'rate_equation':'dre','bandgap':9.*c.e,'density':2e28,'cross_section':1e-19}
                    )

  dom.add_material(material)

  dom.add_observer(pp.Observer('E', 'watch', ylim=(-laser.E0*1.1, laser.E0*1.1), keep_pml=True, out_step=5))
  dom.add_observer(pp.Observer('rho', 'watch', ylim=(0,0), out_step=5))

  dom.run(time)



if __name__ == '__main__':  
  # example_0D()
  example_1D()
  # example_3D()