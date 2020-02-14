"""

"""
import numpy as np
import scipy.constants as c
import cmath
import copy

from . import field_ionization as fi
from . import impact_ionization as ii

from .misc import *
from .backend import backend as bd



class Material():
    """
    Object used to define the material parameters. 

    Arguments:

        index (float): Linear refractive index. Default is 1.

        resonance (float): Resonance frequency [Hz] for the Lorentz model. Ignored for 0D simulations.

        chi2 (float): Second order susceptibility [m/V]. Ignored for 0D simulations.

        chi3 (float): Third order susceptibility [m^2/V^2]. Ignored for 0D simulations.


        drude_params (dict): Contains the relevant parameters for Drude model. The keys are:

            damping (float): Plasma damping rate [Hz].

            m_CB (float): Effective mass of the electrons in the conduction band (relative to free mass). Default is 1.

            m_VB (float): Effective mass of the holes in the valence band (relative to free mass). Default is 1.

            rho (float): Plasma density [1/m^3]. Default is 0.


        ionization_params (dict): Contains the relevant parameters for ionization. The keys are:

            rate_equation (str): The rate equation model to use when calculating plasma 
                formation rate. Choose "single" or "sre" to use single rate equation. Choose 
                "multiple" or "mre" for multiple rate equations. Choose "delayed" or "dre" for 
                the delayed rate equations. Choose 'fi' for field ionization only. Default is "none".

            bandgap (float): The bandgap between the conduction and the valence bands in Joules.

            density (float): The density of atoms or molecules in 1/m^3.

            cross_section (float): The cross section for the calculation of the collision rate 
                between charge carriers and neutral atoms or molecules in 1/m^2. 
                In use if rate equation is MRE or DRE.

            recombination_rate (float): The recombination rate [Hz] of the charge carriers. Default is 0.
                
            alpha_sre (float): The impact rate coefficient [m^2/J] used if the rate equation is SRE.

            fi_damping_correction (bool): To use or not a correction factor to Keldsh field ionization
                that accounts for plasma damping. Default is False.

            fi_mode (str): The calculation mode of field ionization. Default is 'linear' and uses a pre-
                           calculated table to make linear interpolation, which is faster than calculating
                           the Keldysh formula every time. However, it requires to be done on the CPU, so
                           it may slow GPU simulations. For full GPU usage, use the 'nearest' mode, which
                           is less precise because of the nearest interpolation instead of linear. However, 
                           this mode is very memory expansive, so use with caution. Finally, the mode
                           'brute' calculates Keldysh formula at each space and time steps.

            fi_table_size (int): The number of entries in the field ionization table used for interpolation
                                 when fi_mode is 'linear' or 'nearest'. Very memory expensive if fi_mode is
                                 'nearest'. Default is 1000.
        
    """


    def __init__(self, index, resonance=0, chi2=0, chi3=0, drude_params:dict={}, ionization_params:dict={}):

        self.index = index
        self.resonance = resonance
        self.chi2 = chi2
        self.chi3 = chi3
        self.rho = 0

        self.trackables = []

        if drude_params == {}:
            self.drude = False
        else:
            self.drude = True
            self.damping = drude_params['damping']
            self.m_CB = 1.0
            self.m_VB = 1.0
            if 'm_CB' in drude_params:
                self.m_CB = drude_params['m_CB']
            if 'm_VB' in drude_params:
                self.m_VB = drude_params['m_VB']
            if self.m_CB !=0 and self.m_VB !=0 :
                self.m_red = (self.m_CB**-1.+self.m_VB**-1.)**-1.
            else: 
                self.m_red = self.m_CB
            if 'rho' in drude_params:
                self.rho = drude_params['rho']
            
        if ionization_params == {}:
            # error if D > 0 and not drude
            self.rate_equation = 'none'
        else:
            self.bandgap = ionization_params['bandgap']
            self.density = ionization_params['density']

            self.recombination_rate = 0
            if 'recombination_rate' in ionization_params:
                self.recombination_rate = ionization_params['recombination_rate']

            self.fi_damping_correction = False
            if 'fi_damping_correction' in ionization_params:
                self.fi_damping_correction = ionization_params['fi_damping_correction']

            self.fi_mode = 'linear'
            if 'fi_mode' in ionization_params:
                self.fi_mode = ionization_params['fi_mode']

            self.fi_table_size = 1000
            if 'fi_table_size' in ionization_params:
                self.fi_table_size = ionization_params['fi_table_size']

            if ionization_params['rate_equation'].lower() in ['fi']:
                self.rate_equation = 'fi'

            if ionization_params['rate_equation'].lower() in ['sre','single']:
                self.rate_equation = 'sre'
                self.alpha_sre = ionization_params['alpha_sre']

            if ionization_params['rate_equation'].lower() in ['mre','multiple']:
                self.rate_equation = 'mre'
                self.cross_section = ionization_params['cross_section']
                self.trackables += ['el_heating_rate','hl_heating_rate','coll_freq_en','coll_freq_hn','critical_energy']

            if ionization_params['rate_equation'].lower() in ['dre','delayed']:
                self.rate_equation = 'dre'
                self.cross_section = ionization_params['cross_section']
                self.trackables += ['el_heating_rate','hl_heating_rate','coll_freq_en','coll_freq_hn','critical_energy','r_e','r_h','xi_e','xi_h']

            self.mask = 1


    def place_in_domain(self, domain, boundaries):

        self.domain = domain
        self.boundaries = boundaries
        self.mask = bd.ones(self.domain.grid)

        if self.domain.D > 0:

            try: # xmin
                ind, rem = self.parse_index(boundaries['xmin']/self.domain.dx)
                # if material boundary is in pml region, fill all pmls with material
                if ind <= int(self.domain.pml_width/self.domain.dx):
                    self.boundaries['xmin'] = self.domain.x.min()
                else:
                    self.mask[:ind] = 0
                    self.mask[ind] -= rem
            except: self.boundaries['xmin'] = self.domain.x.min()

            try: # xmax
                ind, rem = self.parse_index(boundaries['xmax']/self.domain.dx)
                # if material boundary is in pml region, fill all pmls with material
                if ind >= self.domain.Nx - int(self.domain.pml_width/self.domain.dx): 
                    self.boundaries['xmax'] = self.domain.x.max()
                else:
                    self.mask[ind+1:] = 0
                    self.mask[ind] -= 1-rem
            except: self.boundaries['xmax'] = self.domain.x.max()

            try: # ymin
                ind, rem = self.parse_index(boundaries['ymin']/self.domain.dy)
                self.mask[:,:ind] = 0
                self.mask[:,ind] -= rem
            except: self.boundaries['ymin'] = self.domain.y.min()

            try: # ymax
                ind, rem = self.parse_index(boundaries['ymax']/self.domain.dy)
                self.mask[:,ind+1:] = 0
                self.mask[:,ind] -= 1-rem
            except: self.boundaries['ymax'] = self.domain.y.max()

            try: # zmin
                ind, rem = self.parse_index(boundaries['zmin']/self.domain.dz)
                self.mask[:,:,:ind] = 0
                self.mask[:,:,ind] -= rem
            except: self.boundaries['zmin'] = self.domain.z.min()

            try: # zmax
                ind, rem = self.parse_index(boundaries['zmax']/self.domain.dz)
                self.mask[:,:,ind+1:] = 0
                self.mask[:,:,ind] -= 1-rem
            except: self.boundaries['zmax'] = self.domain.z.max()



        if self.rate_equation == 'mre':

            self.k = self.get_number_mre_levels()
            self.rho_k = bd.zeros(self.k+1) # +1 for the zeroth level
            self.rho_hk = bd.zeros(self.k+1)
            # Place initial population in lower level
            self.rho_k[0] = self.rho
            self.rho_hk[0] = self.rho

            if self.domain.D > 0:
                self.rho_k = self.rho_k[:,None,None,None]*self.mask
                self.rho_hk = self.rho_hk[:,None,None,None]*self.mask

        if self.rate_equation in ['mre','dre']:

            self.Ekin = bd.zeros(self.domain.grid)
            self.Ekin_h = bd.zeros(self.domain.grid)


    def make_fi_table(self, laser):
        self.fi_table = fi.fi_table(self, laser, N=self.fi_table_size)

    def make_fi_fit(self, order=5):
        """ Fit performed on fi_table previously generated """
        fi_table_numpy = bd.numpy(self.fi_table)
        self.fi_fit_coefficients = np.polyfit(np.log(fi_table_numpy[:,0]), np.log(fi_table_numpy[:,1]), order)
        self.fi_fit_coefficients = bd.array(self.fi_fit_coefficients)

        # import matplotlib.pyplot as plt
        # nu = np.zeros(self.fi_table.shape[0])
        # for i, c in enumerate(self.fi_fit_coefficients):
        # 	nu += c*np.log(self.fi_table[:,0])**(order-i)
        # nu = np.exp(nu)
        # plt.loglog(self.fi_table[:,0], nu)
        # plt.loglog(self.fi_table[:,0], self.fi_table[:,1])
        # plt.show()


    def field_ionization(self, E_amp):

        if self.fi_mode == 'nearest':
            diff_squared = (self.fi_table[:,0][None,None,None,:] - E_amp[...,None])**2
            ind = bd.argmin(diff_squared, -1)
            self.fi_rate = self.fi_table[ind, 1]*self.density

        elif self.fi_mode == 'linear':
            E_flat = bd.flatten(E_amp)
            self.fi_rate = np.interp(bd.numpy(E_flat), bd.numpy(self.fi_table[:,0]), bd.numpy(self.fi_table[:,1]))
            self.fi_rate = bd.reshape(bd.array(self.fi_rate), E_amp.shape)*self.density

        elif self.fi_mode == 'fit':
            self.fi_rate = bd.zeros(self.domain.grid)
            ind = bd.where(E_amp>1e3)
            for i, c in enumerate(self.fi_fit_coefficients):
                self.fi_rate[ind] += c*bd.log(E_amp[ind])**(self.fi_fit_coefficients.shape[0]-1-i)
            self.fi_rate[ind] = bd.exp(self.fi_rate[ind])*self.density

        elif self.fi_mode == 'brute':
            if self.domain.D > 0:
                self.fi_rate = bd.zeros(self.domain.grid)
                for ix in range(self.domain.Nx):
                    for iy in range(self.domain.Ny):
                        for iz in range(self.domain.Nz):
                            self.fi_rate[ix,iy,iz] = fi.fi_rate(E_amp[ix,iy,iz], self, self.domain.laser)*self.density
            else:
                self.fi_rate = fi.fi_rate(E_amp, self, self.domain.laser)*self.density

        else:
            raise ValueError("Invalid field ionization mode. Use 'linear', 'nearest' or 'brute'.")

        # Saturation
        self.fi_rate *= self.mask*(self.density-self.rho)/self.density

        # Plasma formation
        self.rho += self.domain.dt*self.fi_rate

        # Energy loss of the field = Energy gain in plasma
        self.domain.fields['Jfi'] = self.bandgap*self.fi_rate[...,None]*self.domain.fields['E']/(E_amp[...,None]+1e-9)**2.0


    def impact_ionization(self, E):
        tracks = [observer.target for observer in self.domain.observers if observer.target in self.trackables]
        self.ii_rate = self.mask*ii.ii_rate(E, self, self.domain.laser, self.domain.dt, tracks)
        self.rho += self.domain.dt*self.ii_rate


    def recombination(self):
        self.rho -= self.domain.dt*self.rho*self.recombination_rate


    def get_number_mre_levels(self):
        return int(get_critical_energy(self.domain.laser.E0, self, self.domain.laser)/(c.hbar*self.domain.laser.omega) + 1)


    @staticmethod
    def parse_index(float_index):
        return int(float_index),float_index%1


    @property
    def plasma_freq(self):
        return bd.abs((c.e**2*self.rho/(c.epsilon_0*self.m_red*c.m_e)))**.5


    @property
    def _Drude_index(self):
        # torch doesn't support complex numbers yet, so we have to do a detour in numpy
        # This is mostly ok, because this is currently only used for 0D simulations.
        wp = bd.numpy(self.plasma_freq)
        self.drude_index = np.zeros(wp.shape, dtype=complex)
        self.drude_index = np.sqrt(self.index**2 - wp**2 / (self.domain.laser.omega**2 + 1j*self.domain.laser.omega*self.damping))
        return self.drude_index


    @property
    def Reflectivity(self):
        n = self._Drude_index
        self.reflectivity = np.abs((n-1.)/(n+1.))**2
        return bd.array(self.reflectivity)







# def log_interp(zz, xx, yy):
# 	logz = np.log10(zz)
# 	logx = np.log10(xx)
# 	logy = np.log10(yy)
# 	return np.power(10.0, np.interp(logz, logx, logy))

