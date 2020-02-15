import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gzip
import pickle

from .misc import *
from .backend import backend as bd




class Observer():

    def __init__(self, target:str=None, mode:str='watch', x=None, y=None, z=None, keep_pml=False, out_step=1):
        self.target = target
        self.mode = mode
        self.x = x
        self.y = y
        self.z = z
        self.keep_pml = keep_pml
        self.out_step = out_step

        self.cache = 0            


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
            

    def get_data(self, target=None, slicing=True):

        target = self.target if target is None else target
        data = None

        #### Vector Fields (E, H, Jb, Jf, Jfi, P) ##########################################
        if target in self.domain.fields:
            data = copy.deepcopy(self.domain.fields[target])

        # x, y or z individual component (ex: 'Ez' or 'Hy')
        if target[:-1] in self.domain.fields and target[-1] == 'x':
            data = copy.deepcopy(self.domain.fields[target[:-1]][...,0])
        elif target[:-1] in self.domain.fields and target[-1] == 'y':
            data = copy.deepcopy(self.domain.fields[target[:-1]][...,1])
        elif target[:-1] in self.domain.fields and target[-1] == 'z':
            data = copy.deepcopy(self.domain.fields[target[:-1]][...,2])
        if data is not None:
            if len(data.shape) == 4:
                data = (data[...,0]**2 + data[...,1]**2 + data[...,2]**2)**0.5

        #### Plasma density, energies and formation rates ##################################
        # TODO : get the actual list of material's tracktables
        if target in ['rho','rho_k','rho_hk','Ekin','Ekin_h',
                      'el_heating_rate','critical_energy',
                      'r_e','r_h','xi_e','xi_h','coll_freq_en','coll_freq_hn']:
            data = 0
            for material in self.domain.materials:
                data += getattr(material, target)

        if target in ['rho_k', 'rho_hk']:
            data = np.moveaxis(bd.numpy(data), 0, -1) # Spatial axes first

        if target in ['fi_rate', 'ii_rate']: # multiply by dt
            data = 0
            for material in self.domain.materials:
                data += getattr(material, target)*self.domain.dt

        if target in ['rho_fi', 'rho_ii']: # use of cache to mimic a cumsum
            rate = 'fi_rate' if target == 'rho_fi' else 'ii_rate'
            data = self.cache
            for material in self.domain.materials:
                data += getattr(material, rate)*self.domain.dt*self.out_step
            self.cache = copy.deepcopy(data)

        if target == 'ponderomotive_energy':
            data = 0
            for material in self.domain.materials:
                data += ponderomotive_energy(self.domain.E_amp, material, self.domain.laser)

        #### Others ########################################################################

        if target == 'Powerz':
            data = 0
            for material in self.domain.materials:
                data += self.get_data('Ez', slicing=False)*self.get_data('Jfz', slicing=False)

        #### Slicing #######################################################################
        if slicing:
            if self.z is not None:
                data = data[:,:,int((self.z-self.domain.z.min())/self.domain.dz)]
            if self.y is not None:
                data = data[:,int((self.y-self.domain.y.min())/self.domain.dy)]
            if self.x is not None:
                data = data[int((self.x-self.domain.x.min())/self.domain.dx)]

            if not self.keep_pml and self.x is None and self.domain.D > 0:
                data = data[self.domain.nb_pml:-self.domain.nb_pml]

        if data is not None:
            return bd.numpy(data)
        else:
            raise ValueError(f'Could not get {target} from simulation domain.')


    def terminate(self):
        pass


    





class Watcher(Observer):

    # TODO: watch back dumped data
    # TODO: Documentation
    # TODO: Log graphs

    def __init__(self, target:str=None, x=None, y=None, z=None, vlim=(0,0), figsize='default', c='C0', colormap='seismic', keep_pml=False, loop=False, out_step=1):
        super(Watcher, self).__init__(target, 'watch', x, y, z, keep_pml, out_step)

        self.vlim = vlim
        self.figsize = figsize
        self.c = c
        self.colormap = colormap
        self.viewer = None

        self.loop = loop
        self.save_data = loop
        self.stack_data = []


    def call(self, data=None):

        data = self.get_data() if data is None else data

        if self.save_data:
            self.stack_data.append(copy.deepcopy(data))

        if self.viewer is None:

            if self.D == 0:
                self.viewer = Viewer0d(self.domain, self.target, self.vlim, self.c, self.figsize)

            if self.D == 1:
                if self.x is None:
                    if self.keep_pml:
                        axis = {'x': np.linspace(0, self.domain.Lx, self.domain.Nx)}
                    else:
                        axis = {'x': np.linspace(self.domain.pml_width, self.domain.Lx-self.domain.pml_width, self.domain.Nx - 2*self.domain.nb_pml)}
                if self.y is None:
                    axis = {'y': np.linspace(0, self.domain.Ly, self.domain.Ny)}
                if self.z is None:
                    axis = {'z': np.linspace(0, self.domain.Lz, self.domain.Nz)}
                self.viewer = Viewer1d(axis, self.domain, self.target, self.vlim, self.c, self.keep_pml, self.figsize)

            if self.D == 2:
                if self.x is not None:
                    axes = {'y': np.linspace(0, self.domain.Ly, self.domain.Ny),
                            'z': np.linspace(0, self.domain.Lz, self.domain.Nz)}
                if self.y is not None:
                    axes = {'x': np.linspace(0, self.domain.Lx, self.domain.Nx),
                            'z': np.linspace(0, self.domain.Lz, self.domain.Nz)}
                if self.z is not None:
                    axes = {'x': np.linspace(0, self.domain.Lx, self.domain.Nx),
                            'y': np.linspace(0, self.domain.Ly, self.domain.Ny)}
                self.viewer = Viewer2d(axes, self.domain, self.target, self.vlim, self.colormap, self.keep_pml, self.figsize)

            if self.D == 3:
                raise NotImplementedError('3D view is not available.')

            self.viewer.fig.canvas.mpl_connect('close_event', self.exit_loop)

        self.viewer.update(self.domain.t, data)

        


    def terminate(self):

        # FIXME: loop over multiple windows doesn't work because stuck in while loop

        self.save_data = False

        while self.loop:

            if self.D == 0:
                self.viewer.times, self.viewer.data_axis = [], []

            for t, data in zip(self.domain.times[::self.out_step], self.stack_data):
                self.domain.t = t
                self.call(data)

            if type(self.loop) == int:
                self.loop -= 1

    def exit_loop(self, event):
        self.loop = False
        


class Dumper(Observer):

    # TODO: Documentation

    def __init__(self, target:str=None, x=None, y=None, z=None, keep_pml=False, out_step=1):
        super(Dumper, self).__init__(target, 'dump', x, y, z, keep_pml, out_step)

        self.zipfile = gzip.GzipFile(f'{target}.zip', 'wb')


    def call(self):
        data = self.get_data()
        self.zipfile.write(pickle.dumps(data))


    def terminate(self):
        self.zipfile.close()


class Printer(Observer):

    # TODO: Documentation

    def __init__(self, target:str=None, x=None, y=None, z=None, keep_pml=False, out_step=1):
        super(Printer, self).__init__(target, 'print', x, y, z, keep_pml, out_step)


    def call(self):
        print(self.get_data().mean())


class Returner(Observer):

    # TODO: Documentation

    def __init__(self, target:str=None, x=None, y=None, z=None, keep_pml=False, out_step=1):
        super(Returner, self).__init__(target, 'return', x, y, z, keep_pml, out_step)

        self.stack_data = []

    def call(self, data=None):
        data = self.get_data() if data is None else data
        self.stack_data.append(copy.deepcopy(data))

    def terminate(self):
        return np.squeeze(np.stack(self.stack_data))





class Viewer():

    def __init__(self, domain, target='', vlim=(0,0), figsize='default'):
        self.domain = domain
        self.target = target

        self.min_value = vlim[0]
        self.max_value = vlim[1]

        self.fig = plt.figure() if figsize == 'default' else plt.figure(figsize=figsize)
        self.fig.canvas.set_window_title(self.target)
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0.2)

    def update(self, time, data):

        self.min_value = min(self.min_value, data.min())
        self.max_value = max(self.max_value, data.max())



class Viewer0d(Viewer):

    def __init__(self, domain, target, vlim=(0,0), c='C0', figsize='default'):
        super(Viewer0d, self).__init__(domain, target, vlim, figsize)

        self.times = []
        self.data_axis = []

        self.time_axis, self.units_factor, units_name = format_axis(domain.times, mode='time')

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(min(self.time_axis),max(self.time_axis))
        self.line = self.ax.plot(self.times, self.data_axis, c=c)[0]
        self.ax.set_xlabel(f't [{units_name}]')



    def update(self, time, data):
        super(Viewer0d, self).update(time, data)

        self.times.append(time/self.units_factor)
        self.data_axis.append(data)
        self.line.set_data(self.times, self.data_axis)
        self.ax.set_ylim(self.min_value, self.max_value)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()




class Viewer1d(Viewer):

    def __init__(self, axis, domain, target, vlim=(0,0), c='C0', show_pml=False, figsize='default'):
        super(Viewer1d, self).__init__(domain, target, vlim, figsize)

        axis_name = list(axis.keys())[0]
        axis_values, units_factor, units_name = format_axis(axis[axis_name], mode='length')

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(axis_values.min(), axis_values.max())
        self.line = self.ax.plot(axis_values, np.zeros(axis_values.shape[0]), c=c)[0]

        for material in self.domain.materials:
            self.ax.add_patch(patches.Rectangle((material.boundaries[f'{axis_name}min']/units_factor,-1e90),
                                                (material.boundaries[f'{axis_name}max']-material.boundaries[f'{axis_name}min'])/units_factor,
                                                 1e200, linewidth=1.5, edgecolor='0.8', facecolor='0.9'))
        
        if axis_name == 'x':
            if show_pml:
                self.ax.add_patch(patches.Rectangle((self.domain.x.min(), -1e90),
                                                    domain.pml_width/units_factor, 1e200,
                                                    linewidth=1.5, edgecolor='0.6', facecolor='0.7'))
                self.ax.add_patch(patches.Rectangle(((domain.Lx-domain.pml_width)/units_factor, -1e90),
                                                    domain.pml_width/units_factor, 1e200,
                                                    linewidth=1.5, edgecolor='0.6', facecolor='0.7'))
            else:
                self.ax.set_xlim(domain.pml_width/units_factor, (domain.Lx-domain.pml_width)/units_factor)

        self.ax.axvline(self.domain.laser.position, color='r', lw=2)

        self.ax.set_xlabel(f'{axis_name} [{units_name}]')


    def update(self, time, data):
        super(Viewer1d, self).update(time, data)

        self.line.set_ydata(data)
        self.ax.set_ylim(self.min_value, self.max_value)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



class Viewer2d(Viewer):

    def __init__(self, axes, domain, target, vlim=(0,0), colormap='seismic', show_pml=False, figsize='default'):
        super(Viewer2d, self).__init__(domain, target, vlim, figsize)

        self.colormap = colormap
        self.axes = axes
        self.show_pml = show_pml
        self.ax = self.fig.add_subplot(111)
        self.img = None

    def update(self, time, data):
        # super(Viewer2d, self).update(time, data)
        # TODO: Optional rotation

        # data = np.rot90(data)

        if self.img is None:

            extent = []

            for axis in ['z','y','x']:

                if axis not in self.axes:
                    continue

                if axis == 'x' and not self.show_pml:
                    no_pml_ax = self.axes[axis][self.domain.nb_pml:-self.domain.nb_pml]
                else:
                    no_pml_ax = self.axes[axis]

                if len(extent) == 0:
                    axis_values, units_factor, units_name = format_axis(no_pml_ax, mode='length')
                    self.ax.set_xlabel(f'{axis} [{units_name}]')
                else:
                    axis_values = no_pml_ax/units_factor
                    self.ax.set_ylabel(f'{axis} [{units_name}]')

                extent.append(axis_values.min())
                extent.append(axis_values.max())

            self.img = self.ax.imshow(data, extent=extent, cmap=self.colormap, aspect=1)

        self.img.set_data(data)
        self.img.set_clim(data.min(), data.max())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()




def format_axis(ax, mode='length'):

    # TODO: use the format value function in misc.py

    axis = copy.deepcopy(ax)
    max_value = max([abs(axis.max()), abs(axis.min())])
    factor, units = 1, ''

    if max_value < 1 and max_value >= 1e-3:
        axis *= 1e3
        factor, units = 1e-3, 'm'
    if max_value < 1e-3 and max_value >= 1e-6:
        axis *= 1e6
        factor, units = 1e-6, r'$\mu$'
    if max_value < 1e-6 and max_value >= 1e-9:
        axis *= 1e9
        factor, units = 1e-9, 'n'
    if max_value < 1e-9 and max_value >= 1e-12:
        axis *= 1e12
        factor, units = 1e-12, 'p'
    if max_value < 1e-12:
        axis *= 1e15
        factor, units = 1e-15, 'f'

    if mode == 'length':
        units += 'm'
    if mode == 'time':
        units += 's'

    return axis, factor, units