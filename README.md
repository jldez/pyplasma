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

In a python 3.6+ environnement, start by importing pyplasma

```python
from pyplasma import *
```

Then, set the backend. For 0D, 1D and small 2D simulations, CPU calculations are prefered:

```python
set_backend('numpy')
```

For 3D calculations, if a GPU with CUDA is available and Pytorch is installed, use instead:

```python
set_backend('torch.cuda')
```

The simulation domain is created using the `Domain` class. For 0D, no arguments are needed. A N dimensions domain is created with N elements lists in the arguments `grid` and `size`. For example, a 3D domain can be created with

```python
dom = Domain(grid=[50,128,128], size=[1*um,3*um,3*um], pml_width=200*nm)
```

For now, the boundaries can only be PMLs at both the x-axis ends and periodic along the y and z axes. The posibility to customize this is planned for a future update.

The laser source is created with the `Laser` class

```python
laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=7e4, t0=10*fs)
```

Its amplitude can be set directly with the `E0` argument or indirectly set by a pulse fluence if the pulse duration (FWHM) is set and finite. The `t0` argument is the time correspoding to the peak of the gaussian enveloppe.

## Authors

* **Jean-Luc Déziel**
* **Charles Varin**
* **Louis J. Dubé**
