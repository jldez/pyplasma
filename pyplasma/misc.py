"""

"""
import numpy as np
import scipy.constants as c
import matplotlib.pyplot as plt
import copy



def get_el_heating_rate(E, material, laser):
    A = c.e**2*material.damping*E**2/(2*c.hbar*laser.omega*(material.damping**2+laser.omega**2))
    return A/(material.m_CB*c.m_e)

def get_hl_heating_rate(E, material, laser):
    A = c.e**2*material.damping*E**2/(2*c.hbar*laser.omega*(material.damping**2+laser.omega**2))
    return A/(material.m_VB*c.m_e)

def ponderomotive_energy(E, material, laser):
    return c.e**2*E**2 / (4*material.m_red*c.m_e * (material.damping**2 + laser.omega**2))

def get_critical_energy(E, material, laser):
    return (1+material.m_red/material.m_VB) * (material.bandgap + ponderomotive_energy(E, material, laser))

def ee_coll_freq(Ekin, material):
    """ electron-electron collision rate """
    return 4*c.pi*c.epsilon_0/c.e**2*(6/material.m_CB/c.m_e)**0.5*(2*Ekin/3)**1.5

def el_Ekin_max(E, material, laser):
    """ Upper bound estimation function for the mean kinetic energy of the electrons. 
    
        Arguments:

            E: Electric field of the laser during the simulation.

            material (Material object): The material in which the plasma formation
                takes place.

            laser (Laser object): The laser that causes the plasma formation.


        Returns:
            Upper bound of the mean kinetic energy in Joules (float).
    """

    Ec = get_critical_energy(0, material, laser) # For some reason, we have to use Ec at rest?
    el_heating_rate = get_el_heating_rate(E, material, laser)

    return (-1.5*Ec/(np.log(el_heating_rate*c.hbar*laser.omega\
        /(2.*Ec*material.cross_section*material.density)*(material.m_CB*c.m_e*c.pi/(3.*Ec))**.5))).max()


def surface_roughness(material, boundary, amplitude, noise='white', feature_size='default', show=False):

    if not hasattr(material, 'domain'):
        raise ValueError('Material has to be attached to a Domain in order to add surface roughness to it.')

    if boundary != 'xmin':
        raise NotImplementedError('Surface roughness can only be attached to the xmin boundary.')

    if noise.lower() == 'white':
        roughness_map = white_roughness_map(shape=(material.domain.Ny,material.domain.Nz))

    else:
        feature_size = amplitude if feature_size is 'default' else feature_size
        res = (int(material.domain.Ly/feature_size),int(material.domain.Lz/feature_size))
        if noise.lower() == 'perlin':
            roughness_map = perlin_roughness_map(shape=(material.domain.Ny,material.domain.Nz), res=res)
        if noise.lower() == 'fractal':
            roughness_map = fractal_roughness_map(shape=(material.domain.Ny,material.domain.Nz), res=res)

    roughness_map *= amplitude

    if show:
        fig = plt.figure()
        plt.imshow(roughness_map, cmap='gray')
        fig.canvas.set_window_title('Roughness map')
        plt.show(block=False)

    ix, rem = material.parse_index(material.boundaries[boundary]/material.domain.dx)

    for iy in range(0,material.domain.Ny-1):
        for iz in range(0,material.domain.Nz-1):
            ir, rem_r = material.parse_index((material.boundaries[boundary]-roughness_map[iy,iz])/material.domain.dx)
            material.mask[ir:ix+1,iy,iz] = 1
            material.mask[ir,iy,iz] -= rem_r

    for iy in range(0,material.domain.Ny):
        material.mask[:,iy,-1] = material.mask[:,iy,0]
    for iz in range(0,material.domain.Nz):
        material.mask[:,-1,iz] = material.mask[:,0,iz]


def white_roughness_map(shape):
    return np.random.random(shape)


def perlin_roughness_map(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    max_size = max(shape)
    shape_up = 2
    while shape_up < max_size:
        shape_up = shape_up*2
    shape_up = (shape_up, shape_up)

    resx = closest_divisor(shape_up[0], int(res[0]*shape_up[0]/shape[0]))
    resy = closest_divisor(shape_up[1], int(res[1]*shape_up[1]/shape[1]))
    res = (resx, resy)
    
    delta = (res[0] / shape_up[0], res[1] / shape_up[1])
    d = (shape_up[0] // res[0], shape_up[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    noise = np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)
    noise -= noise.min()
    noise /= noise.max()
    return noise[:shape[0],:shape[1]]


def fractal_roughness_map(shape, res, octaves=5, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * perlin_roughness_map(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    noise -= noise.min()
    noise /= noise.max()
    return noise



def closest_divisor(N, m):
    """ Find the closest number to m that divides N """
    C = 0

    for c in range(1,N):
        if N%c == 0 and abs(c-m) < abs(C-m):
            C = c

    return C



def format_value(value, base_unit='m'):

    f_value = copy.deepcopy(value)
    factor, units = 1, ''

    if value < 1 and value >= 1e-3:
        f_value *= 1e3
        factor, units = 1e-3, 'm'
    if value < 1e-3 and value >= 1e-6:
        f_value *= 1e6
        factor, units = 1e-6, r'$\mu$'
    if value < 1e-6 and value >= 1e-9:
        f_value *= 1e9
        factor, units = 1e-9, 'n'
    if value < 1e-9 and value >= 1e-12:
        f_value *= 1e12
        factor, units = 1e-12, 'p'
    if value < 1e-12 and value >= 1e-15:
        f_value *= 1e15
        factor, units = 1e-15, 'f'
    if value < 1e-15:
        f_value *= 1e18
        factor, units = 1e-18, 'a'

    units += base_unit

    return f_value, factor, units
