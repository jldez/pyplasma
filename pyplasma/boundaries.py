
import scipy.constants as c

from .backend import backend as bd


# TODO : Documentation


class Boundary():

    def __init__(self, domain, boundary:str):
        self.domain = domain
        self.boundary = boundary

    def update_phi_E(self):
        pass

    def update_phi_H(self):
        pass

    def update_E(self):
        pass

    def update_H(self):
        pass


class Periodic(Boundary):

    def __init__(self, domain, boundary:str):
        super(Periodic, self).__init__(domain, boundary)

    def update_E(self):
        if self.boundary == 'x':
            self.domain.fields['E'][0, :, :, :] = self.domain.fields['E'][-1, :, :, :]
        if self.boundary == 'y':
            self.domain.fields['E'][:, 0, :, :] = self.domain.fields['E'][:, -1, :, :]
        if self.boundary == 'z':
            self.domain.fields['E'][:, :, 0, :] = self.domain.fields['E'][:, :, -1, :]            

    def update_H(self):
        if self.boundary == 'x':
            self.domain.fields['H'][-1, :, :, :] = self.domain.fields['H'][0, :, :, :]
        if self.boundary == 'y':
            self.domain.fields['H'][:, -1, :, :] = self.domain.fields['H'][:, 0, :, :]
        if self.boundary == 'z':
            self.domain.fields['H'][:, :, -1, :] = self.domain.fields['H'][:, :, 0, :]



class PML(Boundary):

    def __init__(self, domain, boundary:str, dt:float):
        super(PML, self).__init__(domain, boundary)
        self.nb_pml = domain.nb_pml
        self.shape = (domain.nb_pml, domain.Ny, domain.Nz, 3)

        for field in ['phi_H','phi_E','psi_Ex','psi_Ey','psi_Ez','psi_Hx','psi_Hy','psi_Hz','sigmaE','sigmaH']:
                self.__setattr__(field, bd.zeros(self.shape))

        self.k = 1.0
        self.a = 1e-8

        if self.boundary == 'xmin':
            sigmaE = 40 * bd.arange(self.nb_pml - 0.5, -0.5, -1.0) ** 3 / (self.nb_pml + 1) ** 4
            sigmaH = 40 * bd.arange(self.nb_pml - 1.0, 0, -1.0) ** 3 / (self.nb_pml + 1) ** 4
            
        if self.boundary == 'xmax':
            sigmaE = 40 * bd.arange(0.5, self.nb_pml + 0.5, 1.0) ** 3 / (self.nb_pml + 1) ** 4
            sigmaH = 40 * bd.arange(1.0, self.nb_pml, 1.0) ** 3 / (self.nb_pml + 1) ** 4

        self.sigmaE[:, :, :, 0] = sigmaE[:, None, None]
        self.sigmaH[:-1, :, :, 0] = sigmaH[:, None, None]

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


    def update_E(self):
        if self.boundary == 'xmin':
            self.domain.fields['E'][:self.nb_pml,...] += self.domain.dt/self.domain.dx/c.epsilon_0 * self.phi_E
        if self.boundary == 'xmax':
            self.domain.fields['E'][-self.nb_pml:,...] += self.domain.dt/self.domain.dx/c.epsilon_0 * self.phi_E

    def update_H(self):
        if self.boundary == 'xmin':
            self.domain.fields['H'][:self.nb_pml,...] -= self.domain.dt/self.domain.dx/c.mu_0 * self.phi_H
        if self.boundary == 'xmax':
            self.domain.fields['H'][-self.nb_pml:,...] -= self.domain.dt/self.domain.dx/c.mu_0 * self.phi_H


