
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import numpy as np

from pyplasma import *


domain = Domain(grid=[150], size=[5*um], pml_width=400*nm)

laser = Laser(wavelength=800*nm, pulse_duration=10*fs, t0=20*fs, fluence=3e4, phase=False)
domain.add_laser(laser)

material = Material(index=1.3, drude_params={'damping':2e15}, 
                    ionization_params={'rate_equation':'mre','bandgap':9*eV,'density':2e28,'cross_section':1e-19})
domain.add_material(material, boundaries={'xmin':2*um})
ind_surface = int(np.ceil(1.6*um*domain.Nx/domain.size[0]))

for data in ['Ez','rho','rho_fi','rho_ii','rho_k']:
    r = Returner(data, out_step=3, keep_pml=False)
    domain.add_observer(r)

results = domain.run(60*fs)




# Video
fig = plt.figure(figsize=(12,8))
x = np.linspace(0.4,4.6,results['Ez'].shape[-1])
ax = fig.add_subplot(211,xlim=(x.min(), x.max()), ylim=(-0.5, 1.2))
ax.set_xlabel(r"$x~[\mu\mathrm{m}]$")
ax.add_patch(patches.Rectangle((2,-2),20,4,linewidth=1.5,edgecolor='0.8',facecolor='0.9'))
lines = [plt.plot([], [], c='0.5',label=r'$|\vec{E}|/E_0$')[0] , \
         plt.plot([], [], c='darkred',label=r'$\rho/\rho_0$')[0] ,\
         plt.plot([], [], c='darkred',ls='--',label=r'$\rho_\mathrm{fi}/\rho_0$')[0] ,\
         plt.plot([], [], c='darkred',ls=':',label=r'$\rho_\mathrm{ii}/\rho_0$')[0] ,\
         plt.plot([], [], c='red',ls='-.',label=r'$\rho_\mathrm{k}/\rho_0$')[0] ,\
        ]
ax.legend(loc=1,frameon=False,borderaxespad=0)

ax2 = fig.add_subplot(212)
k = results['rho_k'].shape[-1]
colors = ['C0' for k in range(k-1)] + ['r']
bars = ax2.bar([str(ik) for ik in range(k)], [0 for ik in range(k)], color=colors)
ax2.set(frame_on=False)
rho_0 = results['rho'].max()
ax2.set_ylim(0, results['rho_k'].max()/rho_0)
ax2.set_xlabel('j')
ax2.set_ylabel(r'$\rho_j/\rho_0$')


def init():
    for line in lines:
        line.set_data([],[])
    for bar in bars:
        bar.set_height(0)
    return lines + list(bars)
def animate(i):
    for j,line in enumerate(lines):
        if j == 0:
            line.set_data(x[1:-1], results['Ez'][i][1:-1]/results['Ez'].max())
        if j == 1:
            line.set_data(x[1:-1], results['rho'][i][1:-1]/rho_0)
        if j == 2:
            line.set_data(x[1:-1], results['rho_fi'][i][1:-1]/rho_0)
        if j == 3:
            line.set_data(x[1:-1], results['rho_ii'][i][1:-1]/rho_0)
        if j == 4:
            line.set_data(x[1:-1], results['rho_k'][i,1:-1,-1]/rho_0)
    for ik, bar in enumerate(bars):
        bar.set_height(results['rho_k'][i,ind_surface,ik]/rho_0)
    return lines + list(bars)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=results['Ez'].shape[0], interval=1200/60, blit=True)

plt.tight_layout()
plt.show()
