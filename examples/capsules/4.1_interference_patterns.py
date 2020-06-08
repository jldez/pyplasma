
from scripts.sipe import eta
import matplotlib.pyplot as plt

from pyplasma import *
try:
    set_backend('torch.cuda')
except: pass


index = 1.5
plasma_density = 0e27
kappa_max = 2.5 #Max frequency axes value
number_simulations = 1 #Increase to average and smooth the numerical solution


numerical_solution = 0
for n in range(number_simulations):

    dom = Domain(grid=[50,120,120], size=[600*nm,12*um,12*um], pml_width=200*nm)

    laser = Laser(wavelength=800*nm, pulse_duration=10*fs, fluence=7e4, t0=10*fs, phase=True)
    dom.add_laser(laser, position='default', source_mode='tfsf', ramp=True)

    mat = Material(index=index, resonance=120e-9, drude_params={'rho':plasma_density,'damping':1e15})
    dom.add_material(mat, boundaries={'xmin':300*nm})
    surface_roughness(mat, boundary='xmin', amplitude=20*nm, noise='fractal')
    complex_refractive_index = mat._Drude_index
    if n==0:
        print(f'Complex refractive index (Drude): {complex_refractive_index}')

    dom.add_observer(Returner('E', x=300*nm))
    results = dom.run(15*fs, stability_factor=0.9)

    I = results['E']**2
    I_moy = np.sum(I, axis=0)/dom.Nt

    complex_fourier = np.fft.fft2(I_moy)
    fourier_tmp = (complex_fourier.real**2 + complex_fourier.imag**2)**0.5
    fourier_tmp[0,0] = 0 #Remove zero frequency peak
    fourier_tmp = np.fft.fftshift(fourier_tmp)
    numerical_solution += fourier_tmp/number_simulations

freqy = np.fft.fftfreq(numerical_solution.shape[0], dom.dy)*laser.wavelength
freqz = np.fft.fftfreq(numerical_solution.shape[1], dom.dz)*laser.wavelength

fig = plt.figure()

ax1 = fig.add_subplot(121)
ind_kappa_min = np.argmin(abs(np.fft.fftshift(freqy) + kappa_max))
ind_kappa_max = np.argmin(abs(np.fft.fftshift(freqy) - kappa_max))
ax1.imshow(numerical_solution[ind_kappa_min:ind_kappa_max,ind_kappa_min:ind_kappa_max]
         , cmap='hot', extent=[-kappa_max,kappa_max,-kappa_max,kappa_max])

ax2 = fig.add_subplot(122)
analytical_solution = eta(complex_refractive_index**2, 0.1, 0.4, kappa_max, 500)
ax2.imshow(analytical_solution, extent=[-kappa_max,kappa_max,-kappa_max,kappa_max], cmap='hot')

ax1.set_ylabel(r"$\kappa_y$", rotation=0)
for ax in [ax1, ax2]:
    ax.set_xlabel(r"$\kappa_z$")

ax1.set_title('Numérique', fontsize=12)
ax2.set_title('Analytique', fontsize=12)

plt.show()