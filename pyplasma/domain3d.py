#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
import copy
import tqdm

from . import laser as las
from . import drude as dru
from . import run as run

from .backend import backend as bd


class Domain3d(object):

    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, Laser=None, materials=None, pml_width=0):
        super(Domain3d, self).__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Laser = Laser
        self.materials = materials
        self.pml_width = pml_width

        self.construct_domain()

        self.fields = {}
        for field in ['E','H','P','Jb','Jf','Jfi']:
            self.fields[field] = bd.zeros((Nx,Ny,Nz,3))


    def construct_domain(self):
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.dx = np.abs(self.x[1]-self.x[0])
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.dy = np.abs(self.y[1]-self.y[0])
        self.z = np.linspace(0, self.Lz, self.Nz)
        self.dz = np.abs(self.z[1]-self.z[0])

        self.add_materials()
        self.las_ind = np.argmin(np.abs(self.Laser.pos-self.x))


    def add_materials(self):

        if self.materials is None:
            self.medium = np.full((self.shape), None)
            return None

        for m in self.materials:
            for key in ['x_min','y_min','z_min']:
                if key not in m:
                    m[key] = -np.inf
            for key in ['x_max','y_max','z_max']:
                if key not in m:
                    m[key] = np.inf

        self.medium = np.empty((self.shape), dtype=object)
        for ix in tqdm.tqdm(range(self.Nx), 'Adding materials'):
            for iy in range(self.Ny):
                for iz in range(self.Nz):
                    for m in self.materials:
                        if self.x[ix] >= m['x_min'] and self.x[ix] <= m['x_max'] and \
                           self.y[iy] >= m['y_min'] and self.y[iy] <= m['y_max'] and \
                           self.z[iz] >= m['z_min'] and self.z[iz] <= m['z_max']:
                            self.medium[ix,iy,iz] = copy.deepcopy(m['material'])
                        else:
                            self.medium[ix,iy,iz] = None


    def add_pml(self, dt):
        self.nb_pml = int(self.pml_width/self.dx)
        self.pml_xmin = PML(self, 'xmin', dt)
        self.pml_xmax = PML(self, 'xmax', dt)


    def propagate(self, time, output = ["rho","electric_field"], out_step=1, \
                  remove_pml=True, accelerate_fi=True, source_mode='TFSF', progress_bar=True):
        return run.propagate3d(time, self, output, out_step, remove_pml, accelerate_fi, source_mode, progress_bar)

    
    # @property
    # def E_center_squared(self):
    #     E_center_x = (self.fields['E'][:-1,:-1,:-1,0] + self.fields['E'][:-1,1:,:-1,0] 
    #                 + self.fields['E'][:-1,:-1,1:,0] + self.fields['E'][:-1,1:,1:,0])/4
    #     E_center_y = (self.fields['E'][:-1,:-1,:-1,1] + self.fields['E'][1:,:-1,:-1,1] 
    #                 + self.fields['E'][:-1,:-1,1:,1] + self.fields['E'][1:,:-1,1:,1])/4
    #     E_center_z = (self.fields['E'][:-1,:-1,:-1,2] + self.fields['E'][1:,:-1,:-1,2] 
    #                 + self.fields['E'][:-1,1:,:-1,2] + self.fields['E'][1:,1:,:-1,2])/4
    #     return E_center_x**2 + E_center_y**2 + E_center_z**2

    @property
    def E_amp(self):
        return (self.fields['E'][...,0]**2 + self.fields['E'][...,1]**2 + self.fields['E'][...,2]**2)**0.5


    @property
    def shape(self):
        return (self.Nx, self.Ny, self.Nz)

    @property
    def rho(self):
        return self.get_property('rho')

    @property
    def rho_fi(self):
        return self.get_property('rho_fi')

    @property
    def rho_ii(self):
        return self.get_property('rho_ii')

    @property
    def rate_fi(self):
        return self.get_property('rate_fi')

    @property
    def rate_ii(self):
        return self.get_property('rate_ii')

    @property
    def chis(self):
        index = self.get_property('index')
        chi1 = index**2 - 1
        chi2 = self.get_property('chi2')
        chi3 = self.get_property('chi3')
        return bd.stack([chi1,chi2,chi3], axis=3)

    @property
    def damping(self):
        return self.get_property('damping')

    @property
    def resonance(self):
        return self.get_property('resonance', np.inf)

    @property
    def m_red(self):
        return self.get_property('m_red', 1.)

    @property
    def bandgap(self):
        return self.get_property('bandgap')

    def get_property(self, property:str, fill_value=0.):
        prop = bd.ones(self.shape) * fill_value
        for ix in range(self.Nx):
            for iy in range(self.Ny):
                for iz in range(self.Nz):
                    try: prop[ix,iy,iz] = getattr(self.medium[ix,iy,iz], property)
                    except: pass
        return prop






class PML(object):

    def __init__(self, domain, boundary:str, dt:float):
        self.domain = domain
        self.boundary = boundary
        self.nb_pml = domain.nb_pml
        self.shape = (domain.nb_pml, domain.Ny, domain.Nz, 3)

        self.k = 1.0
        self.a = 1e-8

        if self.boundary == 'xmin':

            sigma = 40 * bd.arange(self.nb_pml - 0.5, -0.5, -1.0) ** 3 / (self.nb_pml + 1) ** 4
            self.sigmaE = bd.zeros(self.shape)
            self.sigmaE[:, :, :, 0] = sigma[:, None, None]
            
            sigma = 40 * bd.arange(self.nb_pml - 1.0, 0, -1.0) ** 3 / (self.nb_pml + 1) ** 4
            self.sigmaH = bd.zeros(self.shape)
            self.sigmaH[:-1, :, :, 0] = sigma[:, None, None]
            
            self.phi_H = bd.zeros(self.shape)
            self.phi_E = bd.zeros(self.shape)
            self.psi_Ex = bd.zeros(self.shape)
            self.psi_Ey = bd.zeros(self.shape)
            self.psi_Ez = bd.zeros(self.shape)
            self.psi_Hx = bd.zeros(self.shape)
            self.psi_Hy = bd.zeros(self.shape)
            self.psi_Hz = bd.zeros(self.shape)
            
            self.bE = bd.exp(-(self.sigmaE / self.k + self.a) * c.c*dt/domain.dx)
            self.cE = ((self.bE - 1.0)* self.sigmaE / (self.sigmaE * self.k + self.a * self.k**2))
            self.bH = bd.exp(-(self.sigmaH / self.k + self.a) * c.c*dt/domain.dx)
            self.cH = ((self.bH - 1.0)* self.sigmaH / (self.sigmaH * self.k + self.a * self.k**2))
            
        if self.boundary == 'xmax':

            sigma = 40 * bd.arange(0.5, self.nb_pml + 0.5, 1.0) ** 3 / (self.nb_pml + 1) ** 4
            self.sigmaE = bd.zeros(self.shape)
            self.sigmaE[:, :, :, 0] = sigma[:, None, None]

            sigma = 40 * bd.arange(1.0, self.nb_pml, 1.0) ** 3 / (self.nb_pml + 1) ** 4
            self.sigmaH = bd.zeros(self.shape)
            self.sigmaH[:-1, :, :, 0] = sigma[:, None, None]

            self.phi_E = bd.zeros(self.shape)
            self.phi_H = bd.zeros(self.shape)
            self.psi_Ex = bd.zeros(self.shape)
            self.psi_Ey = bd.zeros(self.shape)
            self.psi_Ez = bd.zeros(self.shape)
            self.psi_Hx = bd.zeros(self.shape)
            self.psi_Hy = bd.zeros(self.shape)
            self.psi_Hz = bd.zeros(self.shape)

            self.bE = bd.exp(-(self.sigmaE / self.k + self.a) * c.c*dt/domain.dx)
            self.cE = ((self.bE - 1.0)* self.sigmaE / (self.sigmaE * self.k + self.a * self.k**2))
            self.bH = bd.exp(-(self.sigmaH / self.k + self.a) * c.c*dt/domain.dx)
            self.cH = ((self.bH - 1.0)* self.sigmaH / (self.sigmaH * self.k + self.a * self.k**2))


    def update_phi_E(self):
        """ Called *before* the electric field is updated"""
        self.psi_Ex *= self.bE
        self.psi_Ey *= self.bE
        self.psi_Ez *= self.bE

        if self.boundary == 'xmin':
            Hx = self.domain.fields['H'][:self.nb_pml,:,:,0]
            Hy = self.domain.fields['H'][:self.nb_pml,:,:,1]
            Hz = self.domain.fields['H'][:self.nb_pml,:,:,2]
        if self.boundary == 'xmax':
            Hx = self.domain.fields['H'][-self.nb_pml:,:,:,0]
            Hy = self.domain.fields['H'][-self.nb_pml:,:,:,1]
            Hz = self.domain.fields['H'][-self.nb_pml:,:,:,2]

        self.psi_Ex[:, 1:, :, 1] += (Hz[:, 1:, :] - Hz[:, :-1, :]) * self.cE[:, 1:, :, 1]
        self.psi_Ex[:, :, 1:, 2] += (Hy[:, :, 1:] - Hy[:, :, :-1]) * self.cE[:, :, 1:, 2]

        self.psi_Ey[:, :, 1:, 2] += (Hx[:, :, 1:] - Hx[:, :, :-1]) * self.cE[:, :, 1:, 2]
        self.psi_Ey[1:, :, :, 0] += (Hz[1:, :, :] - Hz[:-1, :, :]) * self.cE[1:, :, :, 0]

        self.psi_Ez[1:, :, :, 0] += (Hy[1:, :, :] - Hy[:-1, :, :]) * self.cE[1:, :, :, 0]
        self.psi_Ez[:, 1:, :, 1] += (Hx[:, 1:, :] - Hx[:, :-1, :]) * self.cE[:, 1:, :, 1]

        self.phi_E[..., 0] = self.psi_Ex[..., 1] - self.psi_Ex[..., 2]
        self.phi_E[..., 1] = self.psi_Ey[..., 2] - self.psi_Ey[..., 0]
        self.phi_E[..., 2] = self.psi_Ez[..., 0] - self.psi_Ez[..., 1]


    def update_phi_H(self):
        """ Called *before* the magnetic field is updated"""
        self.psi_Hx *= self.bH
        self.psi_Hy *= self.bH
        self.psi_Hz *= self.bH

        if self.boundary == 'xmin':
            Ex = self.domain.fields['E'][:self.nb_pml,:,:,0]
            Ey = self.domain.fields['E'][:self.nb_pml,:,:,1]
            Ez = self.domain.fields['E'][:self.nb_pml,:,:,2]
        if self.boundary == 'xmax':
            Ex = self.domain.fields['E'][-self.nb_pml:,:,:,0]
            Ey = self.domain.fields['E'][-self.nb_pml:,:,:,1]
            Ez = self.domain.fields['E'][-self.nb_pml:,:,:,2]

        self.psi_Hx[:, :-1, :, 1] += (Ez[:, 1:, :] - Ez[:, :-1, :]) * self.cH[:, :-1, :, 1]
        self.psi_Hx[:, :, :-1, 2] += (Ey[:, :, 1:] - Ey[:, :, :-1]) * self.cH[:, :, :-1, 2]

        self.psi_Hy[:, :, :-1, 2] += (Ex[:, :, 1:] - Ex[:, :, :-1]) * self.cH[:, :, :-1, 2]
        self.psi_Hy[:-1, :, :, 0] += (Ez[1:, :, :] - Ez[:-1, :, :]) * self.cH[:-1, :, :, 0]

        self.psi_Hz[:-1, :, :, 0] += (Ey[1:, :, :] - Ey[:-1, :, :]) * self.cH[:-1, :, :, 0]
        self.psi_Hz[:, :-1, :, 1] += (Ex[:, 1:, :] - Ex[:, :-1, :]) * self.cH[:, :-1, :, 1]

        self.phi_H[..., 0] = self.psi_Hx[..., 1] - self.psi_Hx[..., 2]
        self.phi_H[..., 1] = self.psi_Hy[..., 2] - self.psi_Hy[..., 0]
        self.phi_H[..., 2] = self.psi_Hz[..., 0] - self.psi_Hz[..., 1]
