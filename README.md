# pyplasma

Yet another 3D FDTD solver in Python. The focus of this package is the modelling of plasma formation in intense laser irradiated dielectric medium. Field ionization (Keldysh model) is implemented, along with three models for avalanche ionization trigger by impact ionization. Single rate equation (SRE), multiple rate equations (MRE) and the novel delayed rate equations (DRE) model which paper is available here: https://arxiv.org/abs/1906.08338. All figures of the DRE paper can be reproduced using the scripts under `examples/paper_figures/`

CPU calculations optimized with the numpy library as well as GPU calculations with Pytorch are implemented, based upon the work of http://github.com/flaport/fdtd.

Notable features are: 
- 0D, 1D, 2D and 3D solvers
- CPU or GPU calculations
- Easily deployable real time data visualization while running simulation
- Multiple modes for fast calculation of Keldysh ionization rates (Pre-calculated interpolation tables, polynomial fits), alog with brute force
- Non-linear optics (2nd and 3rd order) based upon https://arxiv.org/abs/1603.09410
- Surface roughness (white noise, perlin noise or fractal noise)

## Installation
Pyplasma is available on `pip`

```
pip install pyplasma
```

For manual installation of the developpement version, 

```
git clone https://github.com/jldez/pyplasma.git 
cd pyplasma
python setup.py develop --user
```

## Get started

A few examples are available in the examples/ directory. 

```
cd pyplasma/examples
python one_dimension.py
```

If Pytorch is installed, you can try the `three_dimensions.py` script, or the `self_reconfiguration.py` script that reproduces the results of https://arxiv.org/abs/1702.02480

## API description

### Import

In a python 3.6+ environnement, start by importing pyplasma

```python
from pyplasma import *
```

### Backend

Then, set the backend. For 0D, 1D and small 2D simulations, CPU calculations are prefered:

```python
set_backend('numpy')
```

For 3D calculations, if a GPU with CUDA is available and Pytorch is installed, use instead:

```python
set_backend('torch.cuda')
```

### Simulation domain

The simulation domain is created using the `Domain` class. For 0D, no arguments are needed. A N dimensions domain is created with N elements lists in the arguments `grid` and `size`. For example, a 3D domain can be created with

```python
dom = Domain(grid=[50,128,128], size=[1*um,3*um,3*um], pml_width=200*nm)
```

For now, the boundaries can only be PMLs at both the x-axis ends and periodic along the y and z axes. The posibility to customize this is planned for a future update.

### Laser source

The laser source is created with the `Laser` class

```python
laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=7e4, t0=10*fs)
```

Its amplitude can be set directly with the `E0` argument or indirectly set by a pulse fluence if the pulse duration (FWHM) is set and finite. The `t0` argument is the time correspoding to the peak of the gaussian enveloppe. The laser source is attached to the simulation domain with

```python
dom.add_laser(laser, position='default', source_mode='tfsf', ramp=True)
```

The default position is right next to the PML at the low x-axis boundary. Propagation is directed towards +x and the polarization is along the z axis. This will be made customizable in a future update. A ramp can be added to smoothly turn on the laser source and prevent numerical artefacts.

### Material

#### Optics and non-linear optics

The dielectric medium parameters can be set using the `Material` class, for example

```python
material = Material(index=1.45, resonance=120*nm)
```

The refractive index and the resonance are the only two mandatory parameters. If resonance is not a wanted element of the simulation, simply choose a value far from the laser wavelength. Non-linear optics is included with the 2nd and 3rd order susceptibilities `chi2` and `chi3` arguments. 

#### Drude model

The free currents from charge carriers are calculated based upon the Drude model. Add this to the simulation with 

```python
material = Material(index=1.45, resonance=120*nm, drude_params={'damping':1e15, 'rho':3e26, 'm_CB':1, 'm_VB':2})
```

The `drude_params` arguments takes a dictionary. The plasma damping is set with the `damping` key, the plasma density (electron-hole pairs density) with the `rho` key, the effective mass of the electrons in the conduction band with the `m_CB` key (relative to the electron's free mass) and the effective mass of the holes in the valence band with the `m_VB` key (relative to the electron's free mass).

#### Ionization

Plasma formation is activated by adding the `ionization_params` argument when creating the `Material` instance:

```python
material = Material(index=1.45, resonance=120*nm, drude_params={'damping':1e15},
                    ionization_params={'rate_equation':'dre','bandgap':9*eV,'density':2e28,'cross_section':1e-19})
```

The `rate_equation` key is used to set the ionization model. It can be set to 

- `fi` for field ionization (Keldysh) only
- `sre` for the single rate equation model (the key `alpha_sre` is then necessary to set the impact ionization rate in m^2/J)
- `mre` for the multiple rate equations model (the key `cross_section` is then necessary to set the carrier-neutral collisional cross-section in 1/m^2)
- `dre` for the delayed rate equations model (the key `cross_section` is then necessary to set the carrier-neutral collisional cross-section in 1/m^2)

For all cases, the `bandgap` (in J) and the molecular `density` (in 1/m^3) of the material has to be indicated. The plasma density will saturate at `density`, as only single ionization per molecule is accounted for (for now).

A electron-hole pair recombination rate can be set with the `recombination_rate` key.

An optional correction to the Keldysh model to account for plasma damping can be toggle on with the key `fi_damping_correction` set to `True`. This is unpublished work and should be kept off while it is still under investigation. Or be stolen and published by anyone, be my guest!

The Keldysh field ionization model is extremely computationaly expansive. While it can be calculated by brute force at every FDTD grid cells and time steps with `fi_mode` key set to `brute`, it is not reasonable. By default, the `fi_mode` is set to `linear`, which will pre-calculate an interpolation table and perform linear interpolation instead. The size of the table is set with the `fi_table_size` key (default is 1000) and will work for electric field amplitudes between 10^3 V/m and 3*E0 (E0 is the peak amplitude of the laser). However, the linear interpolation is performed using the CPU and may cause a speed bottleneck for GPU calculations. It becomes less of a problem for large 3D grids, as the rest of the calculations catch up in computationnal cost. For small grids, CPU is prefered anyway, but for medium grid sizes, one may want to try to set `fi_mode` to `nearest`. This is less accurate, but fully works on GPU. However, it is extremely memory expansive and the `fi_table_size` will probably have to be reduced considerably (which hurts accuracy even more). Finaly, to fix all these issues at the cost of approximating the Keldysh model to a polynomial fit, the `fi_mode` can be set to `fit`, which is quite fast, memory efficient and also works on GPU.

#### Add material to the simulation

The material instance just created is added to the simulation (after adding the laser source to the domain) with

```python
dom.add_material(material, boundaries={'xmin':300*nm})
```

If the domain is 1+ dimensions, the material is added everywhere, unless `boundaries` are specified with the keys `xmin`, `xmax`, `ymin`, `ymax`, `zmin` and `zmax`.

#### Surface roughness

To add surface roughness to the `xmin` boundary (other boundaries not possible yet), use (after adding material to the domain)

```python
surface_roughness(material, boundary='xmin', amplitude=20*nm, noise='fractal', feature_size=100*nm, show=True)
```

The `amplitude` is the maximum thickness added to the surface. The roughness is generated from a random 2D map (that can be visualized at the start of the simulation with `show` set to `True`). The randomness of that map can be tuned with the `noise` argument set to 

- `white` for a random height between 0 and `amplitude` at each grid cell
- `perlin` for Perlin noise, for which the characteristic bump sizes is set with `feature_size`
- `fractal` for layered Perlin noise maps, for which the characteristic bump sizes is set with `feature_size`

### Data extraction from simulations

Under contruction. The API is in its early phase and will ultimately change. However, one can play around with what is available based upon the examples.

### Run simulation

The simulation is ran using

```python
dom.run(15*fs)
```

The total time of the simulation is the only required argument. The time steps are automatically set, but stability can be compromised by various elements added to the simulation. If stability issues occur, the time steps can be reduced with the argument `stability_factor` that multiplies the time steps (default is 0.95).

