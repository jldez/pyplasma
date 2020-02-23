import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import numpy as np



class Viewer():

    def __init__(self, domain, target='', vlim=(0,0), figsize='default'):
        self.domain = domain
        self.target = target

        if vlim == (0,0):
            self.adaptative_vlim = True
        else:
            self.adaptative_vlim = False
        self.min_value = vlim[0]
        self.max_value = vlim[1]

        self.fig = plt.figure() if figsize == 'default' else plt.figure(figsize=figsize)
        self.fig.canvas.set_window_title(self.target)
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0.2)

    def update(self, time, data):

        if self.adaptative_vlim:
            if self.D < 2:
                self.min_value = min(self.min_value, data.min())
                self.max_value = max(self.max_value, data.max())
            else: 
                self.min_value = data.min()
                self.max_value = data.max()



class Viewer0d(Viewer):

    def __init__(self, domain, target, vlim=(0,0), c='C0', figsize='default'):
        super(Viewer0d, self).__init__(domain, target, vlim, figsize)

        self.D = 0
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

    def __init__(self, axis, domain, target, fourier=False, vlim=(0,0), c='C0', show_pml=False, figsize='default'):
        super(Viewer1d, self).__init__(domain, target, vlim, figsize)

        self.D = 1

        axis_name = list(axis.keys())[0]
        axis_values, units_factor, units_name = format_axis(axis[axis_name], mode='length')
        if fourier:
            axis_values, units_name = make_fourier_axis(axis_values, units_name)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(axis_values.min(), axis_values.max())
        self.line = self.ax.plot(axis_values, np.zeros(axis_values.shape[0]), c=c)[0]

        if not fourier:
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

    def __init__(self, axes, domain, target, fourier=False, vlim=(0,0), colormap='seismic', show_pml=False, figsize='default'):
        super(Viewer2d, self).__init__(domain, target, vlim, figsize)

        self.D = 2
        self.fourier = fourier
        self.colormap = colormap
        self.axes = axes
        self.show_pml = show_pml
        self.ax = self.fig.add_subplot(111)
        self.img = None

    def update(self, time, data):
        super(Viewer2d, self).update(time, data)
        # TODO: Optional rotation
        # FIXME: vlim does nothing

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
                    if self.fourier:
                        axis_values, units_name = make_fourier_axis(axis_values, units_name)
                    self.ax.set_xlabel(f'{axis} [{units_name}]')
                else:
                    axis_values = no_pml_ax/units_factor
                    if self.fourier:
                        axis_values, _ = make_fourier_axis(axis_values, units_name)
                    self.ax.set_ylabel(f'{axis} [{units_name}]')

                extent.append(axis_values.min())
                extent.append(axis_values.max())

            self.img = self.ax.imshow(data, extent=extent, cmap=self.colormap, aspect=1)

        self.img.set_data(data)
        self.img.set_clim(self.min_value, self.max_value)

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


def make_fourier_axis(axis, units):
    spacing = abs(axis[1] - axis[0])
    axis = np.fft.fftshift(np.fft.fftfreq(axis.shape[0],spacing))
    units = f'{units}'+r'$^{-1}$'
    return axis, units
