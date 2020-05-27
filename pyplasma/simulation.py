"""

"""
import numpy as np
import scipy.constants as c
import tqdm
import copy
import matplotlib.pyplot as plt

from .boundaries import *
from .misc import format_value
from .backend import backend as bd


class Domain():

    def __init__(self, grid:list=(), size:list=(), pml_width:float=0):

        self.grid = grid
        self.size = size

        if len(grid) != len(size):
            raise ValueError(f'The number of dimensions of grid {grid} do not match the number of dimensions of size {size}')
        
        self.pml_width = pml_width

        self.set_grid_parameters()

        self.laser = None
        self.materials = []
        self.observers = []

        self.initialize_fields()

    
    def add_laser(self, laser, source_mode='TFSF', position='default', remove_reflected_part=False, ramp:bool=False):
        # TODO: polarization axis
        laser.domain = self
        laser.source_mode = source_mode
        laser.position = self.pml_width if position == 'default' else position
        laser.remove_reflected_part = remove_reflected_part
        self.laser_ramp = ramp
        self.laser = laser

    def add_material(self, material, boundaries:dict={}):
        # TODO: check if overlap
        material.place_in_domain(self, boundaries)
        self.materials.append(material)

        if material.rate_equation != 'none' and material.fi_mode != 'brute':

            if not hasattr(self, 'laser'):
                raise ValueError('A laser must be attached first in order to add an ionizable material.')

            material.make_fi_table(self.laser)
            if material.fi_mode == 'fit':
                material.make_fi_fit()


    def add_observer(self, observer):
        if type(observer) == list:
            for obs in observer:
                self.add_observer(obs)
        else:
            observer.place_in_domain(self)
            self.observers.append(observer)

    def initialize_fields(self):
        self.fields = {}
        for field in ['E','H','P','Jb','Jf','Jfi']:
            self.fields[field] = bd.zeros(self.field_shape)


    def run(self, time, stability_factor:float=0.95, Nt:int='auto', progress_bar:bool=True, verbose:bool=True):            

        if type(time) == tuple:
            self.start_time = time[0]
            self.end_time = time[1]
        else:
            self.start_time = 0
            self.end_time = time

        self.Lt = self.end_time - self.start_time
        if Nt == 'auto':
            self.dt = stability_factor*self.max_dt
            self.Nt = int(np.ceil(self.Lt/self.dt))
        else:
            self.Nt = int(Nt)
            self.dt = self.Lt/self.Nt
        self.times = np.linspace(self.start_time, self.end_time, self.Nt)

        if self.D > 0 and verbose:
            fdx, fdy, fdz = [format_value(d,'m') for d in [self.dx, self.dy, self.dz]]
            print(f'Space discretization: dx={fdx[0]:.1f}{fdx[2]}, dy={fdy[0]:.1f}{fdy[2]}, dz:{fdz[0]:.1f}{fdz[2]}.')
            fdt = format_value(self.dt,'s')
            print(f'Time discretization: dt={fdt[0]:.1f}{fdt[2]}.')

        self.set_boundaries(self.dt)

        self.progress_bar = progress_bar
        if self.progress_bar:
            self.tqdm_times = tqdm.tqdm(self.times, 'Running')
        else:
            self.tqdm_times = self.times

        self.running = True

        for self.it, self.t in enumerate(self.tqdm_times):

            if self.D == 0:
                self.update_plasma()
                self.update_currents()
                self.update_E()
                self.observe()

            else:
                self.update_pml_E()
                self.update_plasma()
                self.update_currents()
                self.update_E()
                self.update_pml_H()
                self.update_H()
                self.observe()

            if not self.running:
                break

        return self.terminate()


    def update_pml_E(self):
        for boundary in self.boundaries:
            boundary.update_phi_E()


    def update_pml_H(self):
        for boundary in self.boundaries:
            boundary.update_phi_H()


    def update_plasma(self):
        E_amp = self.E_amp
        for material in self.materials:
            if material.rate_equation != 'none':
                material.field_ionization(E_amp)
            if material.rate_equation in ['sre','mre','dre']:
                material.impact_ionization(E_amp)
            if material.rate_equation != 'none':
                material.recombination()
                material.trapping()


    def update_currents(self):
        self.update_bounded_current()
        self.update_free_current()


    def update_bounded_current(self):
        # See https://arxiv.org/abs/1603.09410
        for material in self.materials:
            w0 = 2*c.pi*c.c/material.resonance if material.resonance > 0 else 1/self.dt
            P = c.epsilon_0*((material.index**2 - 1)*self.fields['E'] + 
                              material.chi2*self.fields['E']**2 + 
                              material.chi3*self.fields['E']**3)
            mask = material.mask[...,None] if self.D > 0 else 1
            self.fields['Jb'] += self.dt*w0**2*(mask*P - self.fields['P'])
            self.fields['P'] += self.dt*self.fields['Jb']


    def update_free_current(self):
        for material in self.materials:
            
            if material.drude:
                G = material.damping*self.dt/2
                self.fields['Jf'] = material.mask[...,None]*(self.fields['Jf']*(1-G)/(1+G) \
                                    + self.dt*c.epsilon_0*material.plasma_freq[...,None]**2*self.fields['E']/(1+G))

            else:
                self.fields['Jf'] = material.mask[...,None]*2*c.epsilon_0*material.index*material.index_imag*self.laser.omega*self.fields['E']



    def update_E(self):

        if self.D == 0:
            self.fields['E'] = self.laser.E(self.t)                

        else:
            # add curl(H)
            self.fields['E'] += self.dt/c.epsilon_0 * self.curl_H(self.fields['H'])

            # add currents
            self.fields['E'] -= self.dt*self.fields['Jb']/c.epsilon_0
            self.fields['E'] -= self.dt*self.fields['Jf']/c.epsilon_0
            self.fields['E'] -= self.dt*self.fields['Jfi']/c.epsilon_0

            # add sources
            # TFSF implementation might not be perfect. See references:
            # Section 3.0 of : https://studylib.net/doc/8392930/6.-total-field---scattered-field-fdtd-implementation-in-m...
            # Section 3.10 of : https://www.eecs.wsu.edu/~schneidj/ufdtd/chap3.pdf
            if self.laser is not None:

                ramp = (np.exp((self.it/10)**2)-1)/np.exp((self.it/10)**2) if self.laser_ramp and self.it < 30 else 1

                if self.laser.source_mode.lower() == 'tfsf':
                    self.fields['E'][self.laser.index_in_domain+1,...,2] += ramp*self.laser.E(self.t+self.dt/2)*c.c*self.dt/self.dx
                elif self.laser.source_mode.lower() == 'soft':
                    self.fields['E'][self.laser.index_in_domain,...,2] += 2*ramp*self.laser.E(self.t+self.dt/2)*c.c*self.dt/self.dx
                elif self.laser.source_mode.lower() == 'hard':
                    self.fields['E'][self.laser.index_in_domain,...,2] = ramp*self.laser.E(self.t+self.dt/2)

            # boundaries
            for boundary in self.boundaries:
                boundary.update_E()



    def update_H(self):

        if self.D > 0:

            # add curl(E)
            self.fields['H'] -= self.dt/c.mu_0 * self.curl_E(self.fields['E'])

            # add sources
            if self.laser is not None and self.D > 0 and self.laser.source_mode.lower() == 'tfsf':
                ramp = (np.exp((self.it/10)**2)-1)/np.exp((self.it/10)**2) if self.laser_ramp and self.it < 30 else 1
                self.fields['H'][self.laser.index_in_domain,...,1] -= ramp*self.dt/(c.mu_0*self.dx)*self.laser.E(self.t)

            # boundaries
            for boundary in self.boundaries:
                boundary.update_H()


    @property
    def E_amp(self):
        if self.D == 0:
            return self.fields['E']
        else:
            return (self.fields['E'][...,0]**2 + self.fields['E'][...,1]**2 + self.fields['E'][...,2]**2)**0.5


    def observe(self):
        for observer in self.observers:
            if self.it%observer.out_step == 0 and observer.out_step > 0:
                observer.call()


    def terminate(self):

        out_dico = {}
        for observer in self.observers:

            if observer.out_step == -1:
                observer.call()

            if observer.mode == 'return':
                out_dico[observer.target] = observer.terminate()
            else:
                observer.terminate()

        return out_dico




    def set_grid_parameters(self):

        self.D = len(self.grid)

        if self.D == 0:
            self.grid = [1]
            self.field_shape = [1]
            return None

        if self.D == 1:
            self.Nx, self.Ny, self.Nz = self.grid[0], 3, 3
            self.grid = [self.Nx, self.Ny, self.Nz]
            self.Lx = self.size[0]
            self.dx = self.dy = self.dz = self.Lx/self.Nx
            self.Ly, self.Lz = self.Ny*self.dy, self.Nz*self.dz

        if self.D == 2:
            self.Nx, self.Ny, self.Nz = self.grid[0], self.grid[1], 3
            self.grid = [self.Nx, self.Ny, self.Nz]
            self.Lx, self.Ly = self.size[0], self.size[1]
            self.dx, self.dy = self.Lx/self.Nx, self.Ly/self.Ny
            self.dz = self.dy
            self.Lz = self.Nz*self.dz

        if self.D == 3:
            self.Nx, self.Ny, self.Nz = self.grid
            self.Lx, self.Ly, self.Lz = self.size
            self.dx, self.dy, self.dz = self.Lx/self.Nx, self.Ly/self.Ny, self.Lz/self.Nz

        self.x = bd.linspace(0, self.Lx, self.Nx)
        self.y = bd.linspace(0, self.Ly, self.Ny)
        self.z = bd.linspace(0, self.Lz, self.Nz)

        self.field_shape = (self.Nx, self.Ny, self.Nz, 3)


    def set_boundaries(self, dt:float=0):

        self.boundaries = []

        if self.D == 0:
            return None

        if self.D > 0:
            self.nb_pml = int(self.pml_width/self.dx)
            if self.nb_pml > 0:
                self.boundaries += [PML(self, 'xmin', dt), PML(self, 'xmax', dt)]
            self.boundaries += [Periodic(self, 'y'), Periodic(self, 'z')]
        

    @property
    def max_dt(self):
        return 1/(c.c*(1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2)**0.5)


    def curl_E(self, E):
        curl = bd.zeros(E.shape)

        curl[:, :-1, :, 0] += (E[:, 1:, :, 2] - E[:, :-1, :, 2])/self.dy
        curl[:, :, :-1, 0] -= (E[:, :, 1:, 1] - E[:, :, :-1, 1])/self.dz

        curl[:, :, :-1, 1] += (E[:, :, 1:, 0] - E[:, :, :-1, 0])/self.dz
        curl[:-1, :, :, 1] -= (E[1:, :, :, 2] - E[:-1, :, :, 2])/self.dx

        curl[:-1, :, :, 2] += (E[1:, :, :, 1] - E[:-1, :, :, 1])/self.dx
        curl[:, :-1, :, 2] -= (E[:, 1:, :, 0] - E[:, :-1, :, 0])/self.dy

        return curl


    def curl_H(self, H):
        curl = bd.zeros(H.shape)

        curl[:, 1:, :, 0] += (H[:, 1:, :, 2] - H[:, :-1, :, 2])/self.dy
        curl[:, :, 1:, 0] -= (H[:, :, 1:, 1] - H[:, :, :-1, 1])/self.dz

        curl[:, :, 1:, 1] += (H[:, :, 1:, 0] - H[:, :, :-1, 0])/self.dz
        curl[1:, :, :, 1] -= (H[1:, :, :, 2] - H[:-1, :, :, 2])/self.dx

        curl[1:, :, :, 2] += (H[1:, :, :, 1] - H[:-1, :, :, 1])/self.dx
        curl[:, 1:, :, 2] -= (H[:, 1:, :, 0] - H[:, :-1, :, 0])/self.dy

        return curl

