#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import scipy.constants as c
import tqdm
import time
import copy

from . import misc as mis
from . import drude as dru
from . import laser as las
from . import field_ionization as fi

from .backend import backend as bd



def run(Time, Material, Laser, output = ["rho","electric_field"], progress_bar=True):
    """
    This function run an entire simulation of plasma formation in a 
        dielectric irradiated by a laser.

        Arguments:
            Time (numpy array): 1D array containing all discrete times for the simulation.
                Exemple: Time=np.linspace(start,end,N) where N is the number of time steps.

            Material (Material object): The material in which the plasma formation
                takes place.

            Laser (Laser object): The laser that causes the plasma formation.

            output (list): A list of strings. The strings indicate what data to output.
                -"rho": plasma density
                -"rho_fi": plasma density from field ionization
                -"rho_ii": plasma density from impact ionization
                -"rate_fi": field ionization rate
                -"rate_ii": impact ionization rate
                -"xi": If the impact ionization model is DRE, xi is the fraction
                    of the electrons E_kin > E_c.
                -"xi_h": Same as "xi", for holes.
                -"Ekin": Mean kinetic energy of the electrons.
                -"Ekin_h": Mean kinetic energy of the holes.
                -"collision_freq_en": Electron-molecule collision frequency.
                -"collision_freq_hn": Hole-molecule collision frequency.
                -"collision_freq_ee": Electron-electron collision frequency.
                -"ibh": Inverse Bremmstrahlung heating rate.
                -"electric_field": Electric field of the laser.
                -"Reflectivity": Reflectivity of the material.
                Default is ["rho","electric_field"].

            progress_bar (bool): If True, a progress bar will be displayed 
                when running the simulation. Default is True.

        Returns:
            (dict): A dictionnary that contains all requested data. 
                The keys correspond to the strings in the output argument.

    """

    out_data = {}
    for obj in output:
        out_data[obj] = []

    dt = abs(Time[1]-Time[0])

    if progress_bar:
        Time = tqdm.tqdm(Time)

    for t in Time:
        Laser.time = t
        Laser.update_Electric_field(Material)
        Material.update_rho(Laser,dt)

        if "rho" in output:
            out_data["rho"].append(Material.rho)
        if "rho_fi" in output:
            out_data["rho_fi"].append(Material.rho_fi)
        if "rho_ii" in output:
            out_data["rho_ii"].append(Material.rho_ii)
        if "rate_fi" in output:
            out_data["rate_fi"].append(Material.rate_fi)
        if "rate_ii" in output:
            out_data["rate_ii"].append(Material.rate_ii)
        if "xi" in output and Material.rate_equation.lower() in ["delayed","dre"]:
            out_data["xi"].append(Material.xi)
        if "xi_h" in output and Material.rate_equation.lower() in ["delayed","dre"]:
            out_data["xi_h"].append(Material.xi_h)
        if "Ekin" in output:
            out_data["Ekin"].append(Material.Ekin)
        if "Ekin_h" in output:
            out_data["Ekin_h"].append(Material.Ekin_h)
        if "collision_freq_en" in output:
            out_data["collision_freq_en"].append(mis.g_en(Material))
        if "collision_freq_hn" in output:
            out_data["collision_freq_hn"].append(mis.g_hn(Material))
        if "collision_freq_ee" in output:
            out_data["collision_freq_ee"].append(mis.g_ee(Material))
        if "ibh" in output:
            out_data["ibh"].append(dru.ibh(Material,Laser,s="eh"))
        if "electric_field" in output:
            out_data["electric_field"].append(Laser.E)
        if "Reflectivity" in output:
            out_data["Reflectivity"].append(Material.Reflectivity(Laser))

    for obj in output:
        out_data[obj] = np.array(out_data[obj])

    return out_data






def propagate(Time, Domain, output = ["rho","electric_field"], out_step=1, \
              remove_pml=True, accelerate_fi=True, source_mode='TFSF', progress_bar=True):

    # TFSF implementation might not be perfect. See references:
    # Section 3.0 of : https://studylib.net/doc/8392930/6.-total-field---scattered-field-fdtd-implementation-in-m...
    # Section 3.10 of : https://www.eecs.wsu.edu/~schneidj/ufdtd/chap3.pdf

    out_data = {}
    for obj in output:
        out_data[obj] = []

    dt = abs(Time[1]-Time[0])
    chis = Domain.chis
    print('Stability condition : dt = {:f} * dt_max'.format(dt/Domain.dx*c.c*(chis[:,0].max()+1)**0.5))

    if progress_bar:
        Time = tqdm.tqdm(Time)

    resonance = Domain.resonance
    damping = Domain.damping
    m_red = Domain.m_red
    bandgap = Domain.bandgap

    if accelerate_fi:
        for m in range(len(Domain.materials)):
            if Domain.materials[m]['material'] != None:
                fi_table = fi.fi_table(Domain.materials[m]['material'], Domain.Laser, \
                                N=5e3, tol=1e-3, output=None, progress_bar=False)
            for i in range(len(Domain.x)):
                if Domain.x[i]>=Domain.materials[m]['x_min'] and Domain.x[i]<=Domain.materials[m]['x_max']:
                    if Domain.medium[i] != None:
                        Domain.medium[i].add_fi_table(fi_table)


    n=0
    fake_laser = las.Fake_Laser(0,Domain.Laser.omega,Domain.Laser.E0)
    for t in Time:

        # Update laser source
        Domain.Laser.time = t
        Domain.Laser.update_Electric_field(Domain.medium[Domain.las_ind])

        # Update ionization state
        for i in range(len(Domain.x)):
            try:
                fake_laser.E = Domain.fields['E'][i]
                Domain.medium[i].update_rho(fake_laser, dt)
            except:
                pass
        rho = Domain.rho
        rate_fi = Domain.rate_fi

        # Update mpi interband currents
        Domain.fields['Jfi'] = bandgap*rate_fi*Domain.fields['E']/(Domain.fields['E']+1.0)**2.0

        # Update bounded currents
        w0 = 2*c.pi*c.c/resonance
        P = c.epsilon_0*(chis[:,0]*Domain.fields['E']+chis[:,1]*Domain.fields['E']**2+chis[:,2]*Domain.fields['E']**3)
        Domain.fields['Jb'] = Domain.fields['Jb'] + dt*w0**2*(P - Domain.fields['P'])
        Domain.fields['P'] += dt*Domain.fields['Jb']

        # Update free currents
        G = damping*dt/2
        omega_p = abs(c.e**2*rho/(c.epsilon_0*m_red*c.m_e))**.5
        Domain.fields['Jf'] = Domain.fields['Jf']*(1-G)/(1+G) + dt*c.epsilon_0*omega_p**2.0*Domain.fields['E']/(1+G)

        # Coefficients for FDTD scheme
        A = (c.epsilon_0 - dt*Domain.sigma_pml/2)/(c.epsilon_0 + dt/2*Domain.sigma_pml)
        B = dt/Domain.dx/(c.epsilon_0 + dt*Domain.sigma_pml/2)*c.epsilon_0/c.mu_0
        C = c.mu_0/c.epsilon_0*B

        # Update magnetic field
        Domain.fields['H'][:-1] = A[:-1]*Domain.fields['H'][:-1]\
                                 -B[:-1]*(Domain.fields['E'][1:]-Domain.fields['E'][:-1])

        # Magnetic field correction for TFSF laser source
        if source_mode == 'TFSF':
            Domain.fields['H'][Domain.las_ind] += dt/(c.mu_0*Domain.dx)*Domain.Laser.E

        # Update electric field
        Domain.fields['E'][1:-1] = A[1:-1]*Domain.fields['E'][1:-1]\
                                  -C[1:-1]*(Domain.fields['H'][1:-1]-Domain.fields['H'][:-2])\
                                  -dt*(Domain.fields['Jb'][1:-1])/c.epsilon_0\
                                  -dt*(Domain.fields['Jf'][1:-1])/c.epsilon_0\
                                  -dt*(Domain.fields['Jfi'][1:-1])/c.epsilon_0

        # Electric field correction for TFSF laser source
        if source_mode == 'TFSF':
            Domain.fields['E'][Domain.las_ind] += dt/(c.epsilon_0*Domain.dx)*Domain.Laser.E/(120*c.pi)

        # Hard source update
        if source_mode == 'hard':
            Domain.fields['E'][Domain.las_ind] = Domain.Laser.E

        if n%out_step == 0:
            # Output data
            if 'rho' in output:
                out_data["rho"].append(rho)
            if 'rho_fi' in output:
                out_data["rho_fi"].append(Domain.rho_fi)
            if 'rho_ii' in output:
                out_data["rho_ii"].append(Domain.rho_ii)
            if 'rho_k' in output:
                out_data["rho_k"].append(Domain.rho_k)
            if 'rate_fi' in output:
                out_data["rate_fi"].append(rate_fi)
            if 'rate_ii' in output:
                out_data["rate_ii"].append(Domain.rate_ii)
            if "electric_field" in output:
                out_data["electric_field"].append(copy.copy(Domain.fields['E']))
            if "magnetic_flux" in output:
                out_data["magnetic_flux"].append(copy.copy(Domain.fields['H']))
            if "bounded_current" in output:
                out_data["bounded_current"].append(copy.copy(Domain.fields['Jb']))
            if "free_current" in output:
                out_data["free_current"].append(copy.copy(Domain.fields['Jf']))
            if "fi_current" in output:
                out_data["fi_current"].append(copy.copy(Domain.fields['Jfi']))
            if 'ponderomotive_energy' in output:
                out_data["ponderomotive_energy"].append(copy.copy(Domain.get_ponderomotive_energy(Domain.fields['E'])))
            if 'ibh' in output:
                out_data["ibh"].append(copy.copy(Domain.get_ibh(Domain.fields['E'])))
            if 'kinetic_energy' in output:
                out_data['kinetic_energy'].append(copy.copy(Domain.kinetic_energy))
            if 'critical_energy' in output:
                out_data['critical_energy'].append(copy.copy(Domain.critical_energy))

        n+=1

    if remove_pml:
        Domain.remove_pml()
    for obj in out_data:
        out_data[obj] = np.array(out_data[obj])
        if remove_pml:
            if Domain.nb_pml > 0:
                try:
                    out_data[obj] = out_data[obj][:,Domain.nb_pml:-Domain.nb_pml]
                except:
                    pass

    return out_data



def propagate3d(Time, Domain, output = ["rho","electric_field"], out_step=1, \
                remove_pml=True, accelerate_fi=True, source_mode='hard', progress_bar=True):

    out_data = {}
    for obj in output:
        out_data[obj] = []

    dt = abs(Time[1]-Time[0])
    chis = stack_3(Domain.chis, -2)
    # print('Stability condition : dt = {:f} * dt_max'.format(dt/Domain.dx*c.c*(chis[:,0].max()+1)**0.5))

    Domain.add_pml(dt)

    if progress_bar:
        Time = tqdm.tqdm(Time, 'Running simulation')

    resonance = stack_3(Domain.resonance)
    damping = stack_3(Domain.damping)
    m_red = stack_3(Domain.m_red)
    bandgap = stack_3(Domain.bandgap)

    if accelerate_fi and Domain.materials is not None:
        for m in range(len(Domain.materials)):
            if Domain.materials[m]['material'] != None:
                fi_table = fi.fi_table(Domain.materials[m]['material'], Domain.Laser, \
                                N=5e3, tol=1e-3, output=None, progress_bar=False)
            for ix, x in enumerate(Domain.x):
                for iy, y in enumerate(Domain.y):
                    for iz, z in enumerate(Domain.z):
                        if x >= Domain.materials[m]['x_min'] and x <= Domain.materials[m]['x_max'] and \
                           y >= Domain.materials[m]['y_min'] and y <= Domain.materials[m]['y_max'] and \
                           z >= Domain.materials[m]['z_min'] and z <= Domain.materials[m]['z_max']:
                            if Domain.medium[ix,iy,iz] != None:
                                Domain.medium[ix,iy,iz].add_fi_table(fi_table)

    n=0
    fake_laser = las.Fake_Laser(0,Domain.Laser.omega,Domain.Laser.E0)
    for t in Time:

        s=time.time()

        # Update PML (phi_E)
        Domain.pml_xmin.update_phi_E()
        Domain.pml_xmax.update_phi_E()

        # Update bounded currents
        w0 = 2*c.pi*c.c/resonance
        P = c.epsilon_0*(chis[...,0]*Domain.fields['E']+chis[...,1]*Domain.fields['E']**2+chis[...,2]*Domain.fields['E']**3)
        Domain.fields['Jb'] = Domain.fields['Jb'] + dt*w0**2*(P - Domain.fields['P'])
        Domain.fields['P'] += dt*Domain.fields['Jb']

        # Update ionization state - major speed bottleneck        
        # E_amp = Domain.E_amp
        # for ix in range(len(Domain.x)):
        #     for iy in range(len(Domain.y)):
        #         for iz in range(len(Domain.z)):
        #             try:
        #                 fake_laser.E = E_amp[ix,iy,iz]
        #                 Domain.medium[ix,iy,iz].update_rho(fake_laser, dt)
        #             except:
        #                 pass
        # rho = stack_3(Domain.rho)
        # rate_fi = stack_3(Domain.rate_fi)

        # # Update free currents
        # G = damping*dt/2
        # omega_p = abs(c.e**2*rho/(c.epsilon_0*m_red*c.m_e))**.5
        # Domain.fields['Jf'] = Domain.fields['Jf']*(1-G)/(1+G) + dt*c.epsilon_0*omega_p**2.0*Domain.fields['E']/(1+G)

        # # Update mpi interband currents
        # Domain.fields['Jfi'] = bandgap*rate_fi*Domain.fields['E']/(Domain.fields['E']+1.0)**2.0

        # Update E
        Domain.fields['E'] += dt/Domain.dx/c.epsilon_0 * curl_H(Domain.fields['H'])

        # Update E for periodic boundaries
        Domain.fields['E'][:, 0, :, :] = Domain.fields['E'][:, -1, :, :]
        Domain.fields['E'][:, :, 0, :] = Domain.fields['E'][:, :, -1, :]
        
        # Update laser source
        Domain.Laser.time = t
        Domain.Laser.update_Electric_field(Domain.medium[Domain.las_ind])

        # Update E with currents
        Domain.fields['E'] += -dt*(Domain.fields['Jb'])/c.epsilon_0 \
                              -dt*(Domain.fields['Jf'])/c.epsilon_0 \
                              -dt*(Domain.fields['Jfi'])/c.epsilon_0

        # Hard source update
        if source_mode == 'hard':
            Domain.fields['E'][Domain.las_ind,...,2] = Domain.Laser.E

        # Electric field correction for TFSF laser source
        if source_mode == 'TFSF':
            Domain.fields['E'][Domain.las_ind,...,2] += dt/(c.epsilon_0*Domain.dx)*Domain.Laser.E/(120*c.pi)

        # Update PML (E)
        Domain.fields['E'][:Domain.nb_pml,...] += dt/Domain.dx/c.epsilon_0 * Domain.pml_xmin.phi_E
        Domain.fields['E'][-Domain.nb_pml:,...] += dt/Domain.dx/c.epsilon_0 * Domain.pml_xmax.phi_E
    
        # Update PML (phi_H)
        Domain.pml_xmin.update_phi_H()
        Domain.pml_xmax.update_phi_H()

        # Update H
        Domain.fields['H'] -= dt/Domain.dx/c.mu_0 * curl_E(Domain.fields['E'])

        # Magnetic field correction for TFSF laser source
        if source_mode == 'TFSF':
            Domain.fields['H'][Domain.las_ind,...,1] -= dt/(c.mu_0*Domain.dx)*Domain.Laser.E

        # Update H for periodic boundaries
        Domain.fields['H'][:, -1, :, :] = Domain.fields['H'][:, 0, :, :]
        Domain.fields['H'][:, :, -1, :] = Domain.fields['H'][:, :, 0, :]

        # Update PML (H)
        Domain.fields['H'][:Domain.nb_pml,...] -= dt/Domain.dx/c.mu_0 * Domain.pml_xmin.phi_H
        Domain.fields['H'][-Domain.nb_pml:,...] -= dt/Domain.dx/c.mu_0 * Domain.pml_xmax.phi_H

        print(int((time.time()-s)*1e6), 'us')


        if n%out_step == 0:
            if "electric_field" in output:
                out_data["electric_field"].append(copy.copy(Domain.fields['E']))
            if "magnetic_field" in output:
                out_data["magnetic_field"].append(copy.copy(Domain.fields['H']))
            # if "rho" in output:
            #     out_data["rho"].append(rho[...,0])

    for key in out_data:
        out_data[key] = bd.array(out_data[key])

    return out_data


def curl_E(E):
    curl = bd.zeros(E.shape)

    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= E[:, :, 1:, 1] - E[:, :, :-1, 1]

    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= E[1:, :, :, 2] - E[:-1, :, :, 2]

    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= E[:, 1:, :, 0] - E[:, :-1, :, 0]

    return curl


def curl_H(H):
    curl = bd.zeros(H.shape)

    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2]
    curl[:, :, 1:, 0] -= H[:, :, 1:, 1] - H[:, :, :-1, 1]

    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0]
    curl[1:, :, :, 1] -= H[1:, :, :, 2] - H[:-1, :, :, 2]

    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1]
    curl[:, 1:, :, 2] -= H[:, 1:, :, 0] - H[:, :-1, :, 0]

    return curl


def stack_3(arr, axis=-1):
    return bd.stack([arr,arr,arr], axis)




    # def update_E(self):
    #     """ update the electric field by using the curl of the magnetic field """

    #     # update boundaries: step 1
    #     for boundary in self.boundaries:
    #         boundary.update_phi_E()

    #     curl = curl_H(self.H)
    #     self.E += self.courant_number * self.inverse_permittivity * curl

    #     # update objects
    #     for obj in self.objects:
    #         obj.update_E(curl)

    #     # update boundaries: step 2
    #     for boundary in self.boundaries:
    #         boundary.update_E()

    #     # add sources to grid:
    #     for src in self.sources:
    #         src.update_E()

    #     # detect electric field
    #     for det in self.detectors:
    #         det.detect_E()

    # def update_H(self):
    #     """ update the magnetic field by using the curl of the electric field """

    #     # update boundaries: step 1
    #     for boundary in self.boundaries:
    #         boundary.update_phi_H()

    #     curl = curl_E(self.E)
    #     self.H -= self.courant_number * self.inverse_permeability * curl

    #     # update objects
    #     for obj in self.objects:
    #         obj.update_H(curl)

    #     # update boundaries: step 2
    #     for boundary in self.boundaries:
    #         boundary.update_H()

    #     # add sources to grid:
    #     for src in self.sources:
    #         src.update_H()

    #     # detect electric field
    #     for det in self.detectors:
    #         det.detect_H()

        

    # 	# Coefficients for FDTD scheme
    # 	A = (c.epsilon_0 - dt*Domain.sigma_pml/2)/(c.epsilon_0 + dt/2*Domain.sigma_pml)
    # 	B = dt/Domain.dx/(c.epsilon_0 + dt*Domain.sigma_pml/2)*c.epsilon_0/c.mu_0
    # 	C = c.mu_0/c.epsilon_0*B

    # 	# Update magnetic field
    # 	Domain.fields['H'][:-1] = A[:-1]*Domain.fields['H'][:-1]\
    # 							 -B[:-1]*(Domain.fields['E'][1:]-Domain.fields['E'][:-1])

    # 	# Magnetic field correction for TFSF laser source
    # 	if source_mode == 'TFSF':
    # 		Domain.fields['H'][Domain.las_ind] += dt/(c.mu_0*Domain.dx)*Domain.Laser.E

    # 	# Update electric field
    # 	Domain.fields['E'][1:-1] = A[1:-1]*Domain.fields['E'][1:-1]\
    # 							  -C[1:-1]*(Domain.fields['H'][1:-1]-Domain.fields['H'][:-2])\
    # 							  -dt*(Domain.fields['Jb'][1:-1])/c.epsilon_0\
    # 							  -dt*(Domain.fields['Jf'][1:-1])/c.epsilon_0\
    # 							  -dt*(Domain.fields['Jfi'][1:-1])/c.epsilon_0

    # 	# Electric field correction for TFSF laser source
    # 	if source_mode == 'TFSF':
    # 		Domain.fields['E'][Domain.las_ind] += dt/(c.epsilon_0*Domain.dx)*Domain.Laser.E/(120*c.pi)

    # 	# Hard source update
    # 	if source_mode == 'hard':
    # 		Domain.fields['E'][Domain.las_ind] = Domain.Laser.E

    # 	if n%out_step == 0:
    # 		# Output data
    # 		if 'rho' in output:
    # 			out_data["rho"].append(rho)
    # 		if 'rho_fi' in output:
    # 			out_data["rho_fi"].append(Domain.rho_fi)
    # 		if 'rho_ii' in output:
    # 			out_data["rho_ii"].append(Domain.rho_ii)
    # 		if 'rho_k' in output:
    # 			out_data["rho_k"].append(Domain.rho_k)
    # 		if 'rate_fi' in output:
    # 			out_data["rate_fi"].append(rate_fi)
    # 		if 'rate_ii' in output:
    # 			out_data["rate_ii"].append(Domain.rate_ii)
    # 		if "electric_field" in output:
    # 			out_data["electric_field"].append(copy.copy(Domain.fields['E']))
    # 		if "magnetic_flux" in output:
    # 			out_data["magnetic_flux"].append(copy.copy(Domain.fields['H']))
    # 		if "bounded_current" in output:
    # 			out_data["bounded_current"].append(copy.copy(Domain.fields['Jb']))
    # 		if "free_current" in output:
    # 			out_data["free_current"].append(copy.copy(Domain.fields['Jf']))
    # 		if "fi_current" in output:
    # 			out_data["fi_current"].append(copy.copy(Domain.fields['Jfi']))
    # 		if 'ponderomotive_energy' in output:
    # 			out_data["ponderomotive_energy"].append(copy.copy(Domain.get_ponderomotive_energy(Domain.fields['E'])))
    # 		if 'ibh' in output:
    # 			out_data["ibh"].append(copy.copy(Domain.get_ibh(Domain.fields['E'])))
    # 		if 'kinetic_energy' in output:
    # 			out_data['kinetic_energy'].append(copy.copy(Domain.kinetic_energy))
    # 		if 'critical_energy' in output:
    # 			out_data['critical_energy'].append(copy.copy(Domain.critical_energy))

    # 	n+=1

    # if remove_pml:
    # 	Domain.remove_pml()
    # for obj in out_data:
    # 	out_data[obj] = np.array(out_data[obj])
    # 	if remove_pml:
    # 		if Domain.nb_pml > 0:
    # 			try:
    # 				out_data[obj] = out_data[obj][:,Domain.nb_pml:-Domain.nb_pml]
    # 			except:
    # 				pass

    # return out_data