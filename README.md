# pyplasma
Python package for plasma formation modelling in a dielectric irradiated by intense laser field. Single rate equation (SRE), multiple rate equations (MRE) and the delayed rate equations models are implemented. See the paper [URL].

# Installation
After cloning the repository, run

```
cd pyplasma
python setup.py install
```

# Get started
You can generate all figures from the paper by running

```
cd pyplasma/examples
python make_all_figures.py
```

Tu run a simple test simulation, try the following python commands:

```
import matplotlib.pyplot as plt
import numpy as np
import pyplasma as pp

time = np.linspace(-20e-15, 20e-15, 1000)
mat = pp.Material(index=2.0,bandgap=5e-19,density=1e28,cross_section=1e-19,damping=1e15)
las = pp.Laser(wavelength=800e-9,pulse_duration=10e-15,fluence=2e3,t0=time.min())
results = pp.run(time,mat,las)

plt.plot(time,results["rho"]/mat.density,label=r"$\rho / \rho_{\mathrm{mol}}$")
plt.plot(time,results["electric_field"]/las.E0,label=r"$E/E_0$")
plt.legend()
plt.show()
```

## Authors

* **Jean-Luc Déziel**
* **Charles Varin**
* **Louis J. Dubé**
