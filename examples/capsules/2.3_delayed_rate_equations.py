
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import numpy as np

from pyplasma import *


domain = Domain(grid=[150], size=[5*um], pml_width=400*nm)

laser = Laser(wavelength=800*nm, pulse_duration=10*fs, t0=20*fs, fluence=2e4, phase=False)
domain.add_laser(laser)

material = Material(index=1.3, drude_params={'damping':2e15}, 
                    ionization_params={'rate_equation':'dre','bandgap':9*eV,'density':2e28,'cross_section':1e-19})
domain.add_material(material, boundaries={'xmin':2*um})
ind_surface = int(np.ceil(1.6*um*domain.Nx/domain.size[0]))

for data in ['Ez','rho','rho_fi','rho_ii','Ekin','critical_energy']:
    r = Returner(data, out_step=3, keep_pml=False)
    domain.add_observer(r)

results = domain.run(60*fs)





def mb_distribution(E, Ekin):
    dist = np.zeros(len(E))
    for i in range(len(E)):
        if Ekin > 0:
            dist[i] = E[i]**.5 * np.exp(-1.5*E[i]/Ekin)
    return dist

# Video
fig = plt.figure(figsize=(12,8))
x = np.linspace(0.4,4.6,results['Ez'].shape[-1])
rho_0 = results['rho'].max()
ax = fig.add_subplot(211,xlim=(x.min(), x.max()), ylim=(-0.5, 1.2))
ax.set_xlabel(r"$x~[\mu\mathrm{m}]$")
ax.add_patch(patches.Rectangle((2,-2),20,4,linewidth=1.5,edgecolor='0.8',facecolor='0.9'))
lines = [ax.plot([], [], c='0.5',label=r'$|\vec{E}|/E_0$')[0] , \
         ax.plot([], [], c='darkred',label=r'$\rho/\rho_0$')[0] ,\
         ax.plot([], [], c='darkred',ls='--',label=r'$\rho_\mathrm{fi}/\rho_0$')[0] ,\
         ax.plot([], [], c='darkred',ls=':',label=r'$\rho_\mathrm{ii}/\rho_0$')[0] ,\
        ]
ax.legend(loc=1,frameon=False,borderaxespad=0)

E_axis = np.linspace(0,30*c.e,200)
ax2 = fig.add_subplot(212,xlim=(E_axis.min()/c.e,E_axis.max()/c.e), ylim=(0, 1.2*2e-10))
lines += ax2.plot([],[],color='k', label=r'$\rho/\rho_0 * \mathcal{E}^{1/2}\exp(-3\mathcal{E}/2\mathcal{E}_\mathrm{kin})$')
line_ec = ax2.axvline(color='r', ls='--', label=r'$\mathcal{E}_c$')
ax2.legend(loc=1,frameon=False,borderaxespad=0)
ax2.set_xlabel(r'$\mathcal{E}$ [eV]')

fill = plt.fill_between(E_axis/c.e, E_axis*0)


def init():
    for line in lines:
        line.set_data([],[])
    return lines + [line_ec] + [fill]
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
            line.set_data(E_axis/c.e, results['rho'][i][ind_surface]/rho_0*mb_distribution(E_axis, results['Ekin'][i][ind_surface]))

    line_ec.set_xdata(results['critical_energy'][i][ind_surface]/c.e)

    ax2.collections.clear()
    ind_ec = np.argmin((E_axis - results['critical_energy'][i][ind_surface])**2)
    fill = plt.fill_between(E_axis[ind_ec:]/c.e, results['rho'][i][ind_surface]/rho_0*mb_distribution(E_axis[ind_ec:], results['Ekin'][i][ind_surface]), color='r')

    return lines + [line_ec] + [fill]
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=results['Ez'].shape[0], interval=1200/50, blit=True)

plt.tight_layout()
plt.show()
