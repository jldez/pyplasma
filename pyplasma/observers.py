
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .misc import *
from .backend import backend as bd




class Observer():

    def __init__(self, target:str=None, mode:str='watch', x=None, y=None, z=None, ylim=(0,0), keep_pml=False, out_step=1):
        self.target = target
        self.mode = mode
        self.x = x
        self.y = y
        self.z = z
        self.ylim = ylim
        self.keep_pml = keep_pml
        self.out_step = out_step

        self.cache = 0

        if self.mode == 'return':
            self.stack_data = []

        self.viewer = None


    def place_in_domain(self, domain):

        self.domain = domain
        
        if self.domain.D == 0:
            self.D = 0
            self.x = self.y = self.z = None

        if self.domain.D == 1:
            self.y = self.domain.y[1]
            self.z = self.domain.z[1]
            self.D = int(self.x == None)

        if self.domain.D == 2:
            self.z = self.domain.z[1]
            self.D = int(self.x == None) + int(self.y == None)

        if self.domain.D == 3:
            self.D = int(self.x == None) + int(self.y == None) + int(self.z == None)
            if self.D == 3 and self.mode == 'watch':
                raise ValueError('You must select a slice at any x, y or z position for your observer in watch mode. 3D view is not available.')


    def call(self):
        if self.mode == 'dump':
            self.dump()
        if self.mode == 'watch':
            self.watch()
        if self.mode == 'print':
            self.print()
        if self.mode == 'return':
            self.stack()


    def dump(self):
        raise NotImplementedError('Mode dump is yet to be implemented.')


    def watch(self):

        data = self.get_data()

        if self.viewer is None:

            if self.D == 0:
                self.viewer = Viewer0d(self.domain, self.target, self.ylim)

            if self.D == 1:
                if self.x is None:
                    axis = {'x': np.linspace(0, self.domain.Lx, self.domain.Nx)}
                if self.y is None:
                    axis = {'y': np.linspace(0, self.domain.Ly, self.domain.Ny)}
                if self.z is None:
                    axis = {'z': np.linspace(0, self.domain.Lz, self.domain.Nz)}
                self.viewer = Viewer1d(axis, self.domain, self.target, self.ylim, self.keep_pml)

            if self.D == 2:
                raise NotImplementedError('2D view is not available.')

            if self.D == 3:
                raise NotImplementedError('3D view is not available.')

        self.viewer.update(self.domain.t, data)


    def print(self):
        print(self.get_data().mean())


    def stack(self):
        self.stack_data.append(self.get_data())



    def get_data(self):

        data = None

        #### Vector Fields (E, H, Jb, Jf, Jfi, P) ##########################################
        if self.target in self.domain.fields:
            data = self.domain.fields[self.target]

        # x, y or z individual component (ex: 'Ez' or 'Hy')
        if self.D > 0:
            if self.target[:-1] in self.domain.fields and self.target[-1] == 'x':
                data = self.domain.fields[self.target[:-1]][...,0]
            elif self.target[:-1] in self.domain.fields and self.target[-1] == 'y':
                data = self.domain.fields[self.target[:-1]][...,1]
            elif self.target[:-1] in self.domain.fields and self.target[-1] == 'z':
                data = self.domain.fields[self.target[:-1]][...,2]
            if data is not None:
                if len(data.shape) == 4:
                    data = (data[...,0]**2 + data[...,1]**2 + data[...,2]**2)**0.5

        #### Plasma density, energies and formation rates ##################################
        # FIXME : get the actual list of material's tracktables
        if self.target in ['rho', 'Ekin', 'Ekin_h', 'el_heating_rate', 'critical_energy','r_e','r_h','xi_e','xi_h']:
            data = 0
            for material in self.domain.materials:
                data += getattr(material, self.target)

        if self.target in ['fi_rate', 'ii_rate']: # multiply by dt
            data = 0
            for material in self.domain.materials:
                data += getattr(material, self.target)*self.domain.dt

        if self.target in ['rho_fi', 'rho_ii']: # use of cache to mimic a cumsum
            rate = 'fi_rate' if self.target == 'rho_fi' else 'ii_rate'
            data = self.cache
            for material in self.domain.materials:
                data += getattr(material, rate)*self.domain.dt
            self.cache = copy.deepcopy(data)

        if self.target == 'ponderomotive_energy':
            data = 0
            for material in self.domain.materials:
                data += material.mask*ponderomotive_energy(self.domain.E_amp, material, self.domain.laser)

        #### Slicing #######################################################################
        if self.z is not None:
            data = data[:,:,int((self.z-self.domain.z.min())/self.domain.dz)]
        if self.y is not None:
            data = data[:,int((self.y-self.domain.y.min())/self.domain.dy)]
        if self.x is not None:
            data = data[int((self.x-self.domain.x.min())/self.domain.dx)]

        if data is not None:
            return bd.numpy(data)
        else:
            raise ValueError(f'Could not get {self.target} from simulation domain.')





class Viewer():

    def __init__(self, domain, target='', ylim=(0,0)):
        self.domain = domain
        self.target = target

        self.min_value = ylim[0]
        self.max_value = ylim[1]

        self.fig = plt.figure()
        self.fig.canvas.set_window_title(self.target)
        self.fig.show()
        self.fig.canvas.draw()

    def update(self):
        pass



class Viewer0d(Viewer):

    def __init__(self, domain, target, ylim=(0,0)):
        super(Viewer0d, self).__init__(domain, target, ylim)

        self.time_axis = []
        self.data_axis = []

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(min(domain.times),max(domain.times))
        self.line = self.ax.plot(self.time_axis, self.data_axis)[0]
        


    def update(self, time, data):

        self.min_value = min(self.min_value, data.min())
        self.max_value = max(self.max_value, data.max())

        self.time_axis.append(time)
        self.data_axis.append(data)
        self.line.set_data(self.time_axis, self.data_axis)
        self.ax.set_ylim(self.min_value, self.max_value)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()




class Viewer1d(Viewer):

    def __init__(self, axis, domain, target, ylim=(0,0), show_pml=False):
        super(Viewer1d, self).__init__(domain, target, ylim)

        axis_name = list(axis.keys())[0]
        axis_values = axis[axis_name]

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(axis_values.min(), axis_values.max())
        self.line = self.ax.plot(axis_values, np.zeros(axis_values.shape[0]))[0]

        for material in self.domain.materials:
            self.ax.add_patch(patches.Rectangle((material.boundaries[f'{axis_name}min'],-1e90),
                                                 material.boundaries[f'{axis_name}max']-material.boundaries[f'{axis_name}min'],
                                                 1e200,linewidth=1.5,edgecolor='0.8',facecolor='0.9'))
        
        if show_pml:
            self.ax.add_patch(patches.Rectangle((0,-1e90),domain.pml_width,1e200,linewidth=1.5,edgecolor='0.6',facecolor='0.7'))
            self.ax.add_patch(patches.Rectangle((domain.Lx-domain.pml_width,-1e90),domain.pml_width,1e200,linewidth=1.5,edgecolor='0.6',facecolor='0.7'))
        else:
            self.ax.set_xlim(domain.pml_width, domain.Lx-domain.pml_width)

        self.ax.axvline(self.domain.laser.position, color='r', lw=2)


    def update(self, time, data):

        self.min_value = min(self.min_value, data.min())
        self.max_value = max(self.max_value, data.max())

        self.line.set_ydata(data)
        self.ax.set_ylim(self.min_value, self.max_value)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

