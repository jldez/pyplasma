
import copy
import numpy as np
import scipy.constants as c

from .material import Material
from . import field_ionization as fi
from . import impact_ionization as ii
from .backend import backend as bd


class Trap(Material):

    def __init__(self, material, energy, density, trapping_rate, recombination_rate=0):
        self.material = material
        self.energy = energy
        self.trap_density = density
        self.trapping_rate = trapping_rate
        self.recombination_rate = recombination_rate
        self.trapped = 0

        drude_params = copy.deepcopy(material.drude_params)
        drude_params['rho'] = 0
        # drude_params['m_VB'] = np.inf

        ionization_params = copy.deepcopy(material.ionization_params)
        ionization_params['bandgap'] = self.energy
        ionization_params['recombination_rate'] = self.recombination_rate

        super(Trap, self).__init__(material.index, drude_params=drude_params, ionization_params=ionization_params)

        assert self.rate_equation.lower() not in ["multiple","mre","delayed","dre"], "Traps imcompatible with MRE and DRE. Use SRE instead."

        material.traps.append(self)

        if energy > 0:
            self.make_fi_table(material.domain.laser)
            if self.fi_mode == 'fit':
                self.make_fi_fit()

        self.domain = material.domain
        self.mask = material.mask



    def fi_rate(self, E_amp):

        if self.energy == 0:
            return self.mask*self.trapped/self.domain.dt

        if self.fi_mode == 'nearest':
            diff_squared = (self.fi_table[:,0][None,None,None,:] - E_amp[...,None])**2
            ind = bd.argmin(diff_squared, -1)
            fi_rate = self.fi_table[ind, 1]*self.density

        elif self.fi_mode == 'linear':
            E_flat = bd.flatten(E_amp)
            fi_rate = np.interp(bd.numpy(E_flat), bd.numpy(self.fi_table[:,0]), bd.numpy(self.fi_table[:,1]))
            fi_rate = bd.reshape(bd.array(fi_rate), E_amp.shape)*self.density

        elif self.fi_mode == 'fit':
            fi_rate = bd.zeros(self.domain.grid)
            ind = bd.where(E_amp>1e3)
            for i, c in enumerate(self.fi_fit_coefficients):
                fi_rate[ind] += c*bd.log(E_amp[ind])**(self.fi_fit_coefficients.shape[0]-1-i)
            fi_rate[ind] = bd.exp(fi_rate[ind])*self.density

        elif self.fi_mode == 'brute':
            if self.domain.D > 0:
                fi_rate = bd.zeros(self.domain.grid)
                for ix in range(self.domain.Nx):
                    for iy in range(self.domain.Ny):
                        for iz in range(self.domain.Nz):
                            fi_rate[ix,iy,iz] = fi.fi_rate(E_amp[ix,iy,iz], self, self.domain.laser)*self.density
            else:
                fi_rate = fi.fi_rate(E_amp, self, self.domain.laser)*self.density

        fi_rate *= (self.material.density-self.material.rho-self.material.trapped_rho)/self.material.density
        fi_rate *= self.trapped/self.density

        return fi_rate


    def ii_rate(self, E):
        # SRE hardcoded here, because other ones would be complicated from traps
        intensity = c.c*self.index*c.epsilon_0*E**2./2.
        ii_rate = self.alpha_sre*self.trapped*intensity
        ii_rate *= (self.material.density-self.material.rho-self.material.trapped_rho)/self.material.density
        return ii_rate


    def recombination(self):
        self.trapped -= self.domain.dt*self.trapped*self.recombination_rate