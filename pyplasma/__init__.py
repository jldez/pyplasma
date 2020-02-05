__version__ = '0.2'

from . import material
from . import laser
from . import run as r
from . import misc
from . import field_ionization
from . import impact_ionization
from . import domain
from . import domain3d
from . import observers

from .backend import backend
from .backend import set_backend

Material = material.Material
Laser = laser.Laser
run = r.run
# Ekin_max = misc.Ekin_max
fi_rate = field_ionization.fi_rate
ii_rate = impact_ionization.ii_rate

Domain = domain.Domain
Domain3d = domain3d.Domain3d
propagate = r.propagate

Observer = observers.Observer
