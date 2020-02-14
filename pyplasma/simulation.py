"""

"""
import numpy as np
import scipy.constants as c
import tqdm
import copy
import matplotlib.pyplot as plt

from .boundaries import *

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

    
    def add_laser(self, laser, source_mode='TFSF', position='default', remove_reflected_part=False):
        # TODO: polarization axis
        laser.domain = self
        laser.source_mode = source_mode
        laser.position = self.pml_width if position == 'default' else position
        laser.remove_reflected_part = remove_reflected_part
        self.laser = laser

    def add_material(self, material, boundaries:dict={}):
        # todo : check if overlap?
        material.place_in_domain(self, boundaries)
        self.materials.append(material)


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


    def run(self, time, progress_bar:bool=True):
        self.times = time.t
        self.dt = time.dt
        self.progress_bar = progress_bar

        if self.progress_bar:
            self.tqdm_times = tqdm.tqdm(time.t, 'Running')
        else:
            self.tqdm_times = time.t

        self.set_boundaries(time.dt)

        for material in self.materials:
            if material.rate_equation != 'none' and material.fi_mode != 'brute':
                material.make_fi_table(self.laser)
                if material.fi_mode == 'fit':
                    material.make_fi_fit()

        if self.D > 0:
            print(f'Cell size: dx={self.dx/1e-9}nm, dy={self.dy/1e-9}nm, dz:{self.dz/1e-9}nm. Time steps: {self.dt/1e-15}fs/{self.max_dt/1e-15}fs.')

        if self.D > 0 and not self.is_stable:
            print(f'Warning. Stability compromised by time steps too large by a factor: {self.dt/self.max_dt}.')

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
            if material.rate_equation is not 'none':
                material.field_ionization(E_amp)
            if material.rate_equation in ['sre','mre','dre']:
                material.impact_ionization(E_amp)
            if material.rate_equation is not 'none':
                material.recombination()


    def update_currents(self):
        self.update_bounded_current()
        self.update_free_current()


    def update_bounded_current(self):
        # TODO: Add reference to Varin's model
        for material in self.materials:
            w0 = 2*c.pi*c.c/material.resonance if material.resonance > 0 else 0
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
                laser_E = self.laser.E(self.t)
                if self.laser.source_mode.lower() == 'tfsf':
                    self.fields['E'][self.laser.index_in_domain,...,2] += self.dt/(c.epsilon_0*self.dx)*laser_E/(120*c.pi)
                elif self.laser.source_mode.lower() == 'hard':
                    self.fields['E'][self.laser.index_in_domain,...,2] = laser_E

            # boundaries
            for boundary in self.boundaries:
                boundary.update_E()



    def update_H(self):

        if self.D > 0:

            # add curl(E)
            self.fields['H'] -= self.dt/c.mu_0 * self.curl_E(self.fields['E'])

            # add sources
            if self.laser is not None and self.D > 0:
                laser_E = self.laser.E(self.t)
                if self.laser.source_mode.lower() == 'tfsf':
                    self.fields['H'][self.laser.index_in_domain,...,1] -= self.dt/(c.mu_0*self.dx)*laser_E
                elif self.laser.source_mode.lower() == 'hard':
                    pass

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
            if self.it%observer.out_step == 0:
                observer.call()


    def terminate(self):

        out_dico = {}
        for observer in self.observers:

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

        self.x = bd.linspace(0,self.Lx, self.Nx)
        self.y = bd.linspace(0,self.Ly, self.Ny)
        self.z = bd.linspace(0,self.Lz, self.Nz)

        self.field_shape = (self.Nx, self.Ny, self.Nz, 3)


    def set_boundaries(self, dt:float=0):

        if self.D == 0:
            self.boundaries = []
            return None

        if self.D > 0:
            self.nb_pml = int(self.pml_width/self.dx)
            self.boundaries = [PML(self, 'xmin', dt), PML(self, 'xmax', dt)]
            self.boundaries += [Periodic(self, 'y'), Periodic(self, 'z')]


    @property
    def max_dt(self):
        delta = min([self.dx, self.dy, self.dz])
        index_max = max([1]+[material.index for material in self.materials])
        return  delta / (self.D**0.5 * c.c * index_max)

    @property
    def is_stable(self):
        return self.dt < self.max_dt


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




class Time():

    def __init__(self, start, end, Nt):
        self.t = np.linspace(start, end, int(Nt))
        self.dt = self.t[1] - self.t[0]
        self.Nt = Nt



