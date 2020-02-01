import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pyplasma as pp

pp.set_backend('numpy')
# pp.set_backend('torch.cuda')


time = np.linspace(-20e-15, 300e-15, 4000)
las = pp.Laser(wavelength=800e-9, pulse_duration=10e-15, fluence=2e3, t0=time.min(), phase=True, pos=5e-7)
mat = pp.Material(index=1.2, bandgap=5e-19, density=2e28, cross_section=1e-19, damping=1e15, resonance=120e-9, rate_equation='none')
dom = pp.Domain3d(5e-6,1e-6,1e-6,100,3,3, pml_width=500e-9, Laser=las, materials=[{'material':mat, 'x_min':2e-6, 'x_max':4e-6}])

results = dom.propagate(time, output=['electric_field','rho'], source_mode='TFSF', out_step=5)



fig = plt.figure()
ax = plt.axes(xlim=(dom.x.min(), dom.x.max()), ylim=(-1.2, 1.2))
lines = [plt.plot([], [])[0] for _ in range(1)]

def init():
    for line in lines:
        line.set_data([],[])
    return lines
def animate(i):
    for j,line in enumerate(lines):
        if j == 0:
            line.set_data(dom.x, results['electric_field'][i,:,1,1,2]/results['electric_field'].max())
        # if j == 1:
        #     line.set_data(dom.x, results['rho'][i,:,1,1]/results['rho'].max())
    return lines
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(len(time)/5), interval=10/len(time)*5*1e3, blit=True)


plt.show()
