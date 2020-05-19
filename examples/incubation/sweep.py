
from incubation import incubation
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
filepath = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":

    extrinsic_trap_creation_probability=0.01
    max_trap_density=1e27

    args = []

    for trap_energy_level in [0.0,3.0,5.0,7.0]:
        for trapping_rate in [1e13,1e14]:
            for trap_recombination_rate in [0.0, 1e12, 1e13]:
                for intrinsic_trap_density in [1e25,1e26,1e27,1e28]:
                    for extrinsic_trap_creation_probability in [0.0, 0.0001, 0.001, 0.01, 0.1]:
                        for max_trap_density in [1e26, 1e27, 1e28]:

                            if intrinsic_trap_density > max_trap_density:
                                continue

                            args.append([trap_energy_level,
                                         trapping_rate,
                                         trap_recombination_rate,
                                         intrinsic_trap_density,
                                         extrinsic_trap_creation_probability,
                                         max_trap_density])

    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # for _ in pool.imap_unordered(incubation, args):
    #     pass


    dtype = [('Et','f8'),('b','f8'),('g','f8'),('i','f8'),('a','f8'),('m','f8')]
    data = np.empty(0, dtype=dtype)
    results = []
    for arg in args:
        results.append(np.load(f'{filepath}/data/Et{arg[0]}_b{arg[1]*1e-15}_g{arg[2]*1e-15}_i{arg[3]}_a{arg[4]}_m{arg[5]}.npy', allow_pickle=True).item()['Fth'])
        data = np.append(data, np.array(tuple(arg), dtype=dtype))
    results = np.array(results)
    N = np.array(np.load(f'{filepath}/data/Et{arg[0]}_b{arg[1]*1e-15}_g{arg[2]*1e-15}_i{arg[3]}_a{arg[4]}_m{arg[5]}.npy', allow_pickle=True).item()['Ns'])
    

    condition = np.full(data.shape, True)

    ## trap energy level - 
    # condition *= data['Et']==0.0 ###
    condition *= data['Et']==3.0
    # condition *= data['Et']==5.0
    # condition *= data['Et']==7.0

    ## trapping rate -  - need to add 1e12 and below
    condition *= data['b']==1e13 ### 
    # condition *= data['b']==1e14

    ## trapped recombination rate - forget this - always 0
    condition *= data['g']==0.0 ###
    # condition *= data['g']==1e12
    # condition *= data['g']==1e13

    ## intrinsic trap density - how fast incubation kicks in - need lower values
    condition *= data['i']==1e25 ###
    # condition *= data['i']==1e26
    # condition *= data['i']==1e27
    # condition *= data['i']==1e28

    # trap creation - delay before incubation kicks in - 0.001 seems good
    # condition *= data['a']==0.0
    # condition *= data['a']==0.0001
    condition *= data['a']==0.001 ###
    # condition *= data['a']==0.01
    # condition *= data['a']==0.1

    ## max trap density - height of the final plateau - 5e26 seems good
    condition *= data['m']==1e26
    # condition *= data['m']==1e27 ###
    # condition *= data['m']==1e28

    condition = np.where(condition)[0]
    Fs = results[condition]

    fig = plt.figure(figsize=(8,6))

    for F, arg in zip(Fs, data[condition]):
        label = r'$\mathcal{E}_t$'+f'{arg[0]}_'
        label += r'$\beta_t$'+f'{arg[1]*1e-15}_'
        label += r'$\gamma_{r,t}$'+f'{arg[2]*1e-15}_'
        label += r'$\rho_t^i$'+f'{arg[3]}_'
        label += r'$\alpha_t$'+f'{arg[4]}_'
        label += r'$\rho_{t,\mathrm{max}}$'+f'{arg[5]}'
        plt.semilogx(N, F, marker='o',label=label)

    # plt.xlim(1,50)
    # plt.ylim(0,1.8)
    plt.legend()
    plt.show()
        
    