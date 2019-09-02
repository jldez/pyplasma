import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pyplasma as pp


time = np.linspace(-20e-15, 40e-15, 3000)
las = pp.Laser(wavelength=800e-9, pulse_duration=10e-15, fluence=2e4, t0=time.min(), phase=True)
mat = pp.Material(index=2.0, bandgap=5e-19, density=2e28, cross_section=1e-19, damping=1e15, resonance=120e-9, rate_equation='sre')
dom = pp.Domain(2000e-9, Nx=100, pml_width=800e-9, lasers=[{'x':0,'laser':las}], materials=[{'x_min':300e-9,'x_max':1700e-9,'material':mat}])

results = pp.propagate(time, dom, output=['rho','electric_field','bounded_current'], remove_pml=True, accelerate_fi=True, progress_bar=True)
print(results['rho'].max())


fig = plt.figure()
ax = plt.axes(xlim=(dom.x.min(), dom.x.max()), ylim=(-1.2, 1.2))
lines = [plt.plot([], [])[0] for _ in range(2)]

def init():
	for line in lines:
		line.set_data([],[])
	return lines
def animate(i):
	for j,line in enumerate(lines):
		if j == 0:
			line.set_data(dom.x, results['electric_field'][i]/results['electric_field'].max())
		if j == 1:
			line.set_data(dom.x, results['rho'][i]/results['rho'].max())
	return lines
anim = animation.FuncAnimation(fig, animate, init_func=init,
							   frames=len(time), interval=10/len(time)*1e3, blit=True)


plt.show()

