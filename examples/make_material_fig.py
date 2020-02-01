#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['ps.useafm'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 14})
import scipy.constants as c
import os

import pyplasma as pp


def Threshold(material,laser):
	return c.epsilon_0*material.m_red*c.m_e/c.e**2.*material.index**2.*(laser.omega**2.+material.damping**2.)


def above_threshold(tt,material,laser):

	dt = tt[1]-tt[0]
	threshold = Threshold(material,laser)

	for t in tt:
		laser.time = t
		laser.update_Electric_field(material)
		material.update_rho(laser,dt)
		if material.rho >= threshold:
			return True
	return False




if __name__ == '__main__':

	tolerance = 0.01

	materials = {}
	materials["sio2"] = {"index":1.45,"bandgap":9.0*c.e,"m_CB":1.0,"m_VB":1.0\
						 ,"density":2.2e28,"cross_section":0.661e-19,"damping":2.0e15, "gr":1/250e-15}
	materials["al2o3"] = {"index":1.76,"bandgap":6.5*c.e,"m_CB":0.8,"m_VB":1.0\
						 ,"density":2.35e28,"cross_section":1.33e-19,"damping":1.0e15,"gr":0}
	materials["hfo2"] = {"index":2.09,"bandgap":5.1*c.e,"m_CB":0.4,"m_VB":1.0\
						 ,"density":2.77e28,"cross_section":1.24e-19,"damping":0.5e15,"gr":0}
	materials["ta2o5"] = {"index":2.1,"bandgap":3.8*c.e,"m_CB":0.5,"m_VB":1.0\
						 ,"density":1.12e28,"cross_section":2.50e-19,"damping":0.4e15,"gr":0}
	materials["tio2"] = {"index":2.52,"bandgap":3.3*c.e,"m_CB":0.3,"m_VB":1.0\
						 ,"density":3.19e28,"cross_section":1.08e-19,"damping":0.5e15,"gr":0}


	taus = np.logspace(np.log10(20),np.log10(1200),20)*1e-15
	

	for m in materials:
		fth_tau = []
		for tau in taus:
			Fmin,Fmax = 0.1,10
			while abs(Fmax-Fmin) > tolerance:
				F = np.exp((np.log(Fmax)+np.log(Fmin))/2)
				t = np.linspace(-2*tau,2*tau,10*tau+1000)
				laser = pp.Laser(wavelength=800e-9, pulse_duration=tau,fluence=F*1e4,t0=t.min(),transmit=True)
				material = pp.Material(rate_equation="dre",index=materials[m]["index"],bandgap=materials[m]["bandgap"], \
					m_CB=materials[m]["m_CB"], m_VB=materials[m]["m_VB"], density=materials[m]["density"], \
					cross_section=materials[m]["cross_section"],damping=materials[m]["damping"],recombination_rate=materials[m]["gr"])

				if above_threshold(t,material,laser):
					Fmax = F
				else:
					Fmin = F
				
			fth_tau.append(np.exp((np.log(Fmax)+np.log(Fmin))/2))
		materials[m]["fth_tau"] = np.array(fth_tau)


	fig = plt.figure(figsize=(6,4))
	colors = plt.cm.terrain(np.linspace(-0.0, 1.0, 19))

	# mero2005 - sio2
	t = [23,30,44,108,150,300,370,635,1100]
	F = [1.7,1.85,2.2,2.8,3.35,4.2,4.45,4.9,6.36]
	plt.scatter(t,F,label=r"$\mathrm{SiO_2}$",marker="o",color=colors[0],s=30)
	# dre
	plt.plot(taus*1e15,materials["sio2"]["fth_tau"],color=colors[0])

	# mero2005 - al2o3
	t = [23,30,44,108,150,300,370,635,1100]
	F = [1.32,1.48,1.65,2.08,2.18,2.75,2.86,3.22,4.03]
	plt.scatter(t,F,label=r"$\mathrm{Al_2O_3}$",marker="s",color=colors[2],s=30)
	# dre - al2o3
	plt.plot(taus*1e15,materials["al2o3"]["fth_tau"],color=colors[2])

	# #mero2005 - hfo2
	t = [28,32,60,102,150,320,430,740,1200]
	F = [0.85,0.925,1.12,1.31,1.53,1.79,1.9,2.4,2.58]
	plt.scatter(t,F,label=r"$\mathrm{HfO_2}$",marker="D",color=colors[4],s=30)
	# dre - hfo2
	plt.plot(taus*1e15,materials["hfo2"]["fth_tau"],color=colors[4])

	#mero2005 - ta2o5
	t = [28,32,60,102,150,320,430,740,1200]
	F = [0.51,0.56,0.64,0.83,0.87,1.06,1.2,1.56,1.81]
	plt.scatter(t,F,label=r"$\mathrm{Ta_2O_5}$",marker="^",color=colors[5],s=30)
	# dre - ta2o5
	plt.plot(taus*1e15,materials["ta2o5"]["fth_tau"],color=colors[5])

	#mero2005 - tio2
	t = [28,32,60,102,150,320,430,740,1200]
	F = [0.47,0.49,0.53,0.59,0.63,0.78,0.96,1.05,1.38]
	plt.scatter(t,F,label=r"$\mathrm{TiO_2}$",marker="v",color=colors[7],s=30)
	# dre - tio2
	plt.plot(taus*1e15,materials["tio2"]["fth_tau"],color=colors[7])


	ax = plt.gca()
	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.xlim(10,2000)
	plt.ylim(0.25,10)
	plt.legend(loc=2,frameon=False,ncol=2,columnspacing=1,handletextpad=0.5)
	plt.xlabel(r"$\tau~\mathrm{[fs]}$")
	plt.ylabel(r"$F_{\mathrm{th}}~\mathrm{[J/cm}^2]$")
	plt.tight_layout()
	plt.savefig("materials.pdf")
	os.system("pdfcrop materials.pdf materials.pdf > /dev/null")
	# plt.show()