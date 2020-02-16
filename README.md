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

.

## Authors

* **Jean-Luc Déziel**
* **Charles Varin**
* **Louis J. Dubé**
