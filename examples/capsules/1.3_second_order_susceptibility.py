
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import numpy as np

from pyplasma import *

Nx = 1000
domain = Domain(grid=[Nx], size=[20*um], pml_width=400*nm)

laser = Laser(wavelength=800*nm, pulse_duration=10*fs, t0=20*fs, E0=3e9, phase=True)
domain.add_laser(laser)

material = Material(index=2, chi2=1e-10, resonance=120e-9)
domain.add_material(material, boundaries={'xmin':10*um})

r = Returner('Ez', out_step=8, keep_pml=False)
domain.add_observer(r)

results = domain.run(100*fs)
Nt = domain.Nt



ft = np.fft.fft(results['Ez'][:,int(results['Ez'].shape[-1]/2):], axis=1)
results['fft'] = (ft.real**2 + ft.imag**2)**.5
results['freq'] = np.fft.fftfreq(ft.shape[-1])/domain.dx*laser.wavelength

# Video
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(2,1,1)    
x = np.linspace(0.4,19.6,results['Ez'].shape[-1])
ax.set_xlim((x.min(), x.max()))
ax.set_ylim((-1.2, 1.2))
ax.set_xlabel(r"$x~[\mu\mathrm{m}]$")
ax.add_patch(patches.Rectangle((10,-2),20,4,linewidth=1.5,edgecolor='0.8',facecolor='0.9'))
lines = [ax.plot([], [], c='r')[0] for _ in range(1)]

ax2 = fig.add_subplot(2,1,2)
ax2.set_xlim((0, 12))
ax2.set_ylim((1e-4, 2))
ax2.set_xlabel(r"$\mathrm{Fréquence}~[\omega]$")
lines += [ax2.semilogy([1], [1], c='k')[0] for _ in range(1)]

def init():
	for line in lines:
		line.set_data([],[])
	return lines
def animate(i):
	for j,line in enumerate(lines):
		if j == 0:
			line.set_data(x, results['Ez'][i]/results['Ez'].max())
		if j == 1:
			line.set_data(results['freq'][1:int(Nx/4)-10], results['fft'][i][1:int(Nx/4)-10]/results['fft'][1:int(Nx/4)-10].max())
	return lines
anim = animation.FuncAnimation(fig, animate, init_func=init,
							   frames=int(Nt/8), interval=1e3/60, blit=True)

plt.tight_layout()
plt.show()
