import numpy as np
import copy
import gzip
import pickle

from .misc import *
from .viewer import *
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
                      'el_heating_rate','hl_heating_rate','critical_energy',
                      'r_e','r_h','xi_e','xi_h','coll_freq_en','coll_freq_hn',
                      'trapped_rho']:
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

            if not self.keep_pml and self.x is None and self.domain.D > 0 and self.domain.nb_pml > 0:
                data = data[self.domain.nb_pml:-self.domain.nb_pml]

        if data is not None:
            return bd.numpy(data)
        else:
            raise ValueError(f'Could not get {target} from simulation domain.')

    
    def get_fourier_data(self, target=None, slicing=True):
        data = self.get_data(target, slicing)

        fft = np.fft.fft if self.D < 2 else np.fft.fft2

        complex_fourier_data = fft(data)
        fourier_data = (complex_fourier_data.real**2 + complex_fourier_data.imag**2)**0.5
        fourier_data = np.fft.fftshift(fourier_data)

        return fourier_data


    def terminate(self):
        pass


    





class Watcher(Observer):

    # TODO: watch back dumped data
    # TODO: Documentation
    # TODO: Log graphs

    def __init__(self, target:str=None, x=None, y=None, z=None, fourier:bool=False, 
                 vlim=(0,0), figsize='default', c='C0', colormap='seismic', keep_pml=False, 
                 loop=False, out_step=1):
        super(Watcher, self).__init__(target, 'watch', x, y, z, keep_pml, out_step)

        self.fourier = fourier
        self.vlim = vlim
        self.figsize = figsize
        self.c = c
        self.colormap = colormap
        self.viewer = None

        self.loop = loop
        self.save_data = loop
        self.stack_data = []


    def call(self, data=None):

        get_data = self.get_data if not self.fourier else self.get_fourier_data
        data = get_data() if data is None else data

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
                self.viewer = Viewer1d(axis, self.domain, self.target, self.fourier, self.vlim, self.c, self.keep_pml, self.figsize)

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
                self.viewer = Viewer2d(axes, self.domain, self.target, self.fourier, self.vlim, self.colormap, self.keep_pml, self.figsize)

            if self.D == 3:
                raise NotImplementedError('3D view is not available.')

            self.viewer.fig.canvas.mpl_connect('close_event', self.exit_loop)

        self.viewer.update(self.domain.t, data)

        

    def terminate(self):

        # FIXME: loop over multiple windows doesn't work because stuck in while loop

        self.save_data = False

        if self.loop:
            print(f'Looping over {self.target}.')
            self.start_loop()


    def start_loop(self):

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



class Stopper(Observer):

    def __init__(self, target:str, condition:callable, x=None, y=None, z=None, verbose:bool=True):
        super(Stopper, self).__init__(target, 'stop', x, y, z, keep_pml=False, out_step=1)
        self.condition = condition
        self.verbose = verbose

    def call(self):
        data = self.get_data()
        if self.condition(data):
            self.domain.running = False
            if self.verbose:
                print(f'Stopped simulation because the condition on {self.target} is True.')
