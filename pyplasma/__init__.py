
__name__ = 'pyplasma'
__author__ = 'Jean-Luc Déziel'
__version__ = '0.4.0'

from .backend import backend
from .backend import set_backend

from .simulation import Domain
from .laser import Laser
from .material import Material
from .traps import Trap
from .observers import *

import scipy.constants as c

ns = 1e-9
ps = 1e-12
fs = 1e-15

um = 1e-6
nm = 1e-9
pm = 1e-12

eV = c.e