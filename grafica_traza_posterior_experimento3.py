from random import sample
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import corner

#DATA
#Step of the algorithm
k = 2
#RECOVER LOG-KERNEL OF POSTERIOR AND SAMPLES
kernelpos = np.loadtxt(F"log_kernel_exp3_{k}.txt")
samples = np.loadtxt(F"muestras_exp3_{k}.txt")
#RECOVER LOG-KERNEL AND SAMPLES WITH BURN
kernelpos = kernelpos[int(0.1*len(kernelpos)):] 
samples = samples[int(0.1*len(kernelpos)):]

#TRACE
traza = plt.figure(0)
plt.plot(np.arange(len(kernelpos)),kernelpos)
traza.savefig(F"traza_exp3_{k}.png")

"""
#TRACE (PARTS OF THE TRACE)
kernelpos_a=kernelpos[:int(0.69*len(kernelpos))]
traza = plt.figure(1)
plt.plot(np.arange(len(kernelpos_a)),kernelpos_a)
traza.savefig(F"traza_exp3_a{k}.png")
kernelpos_b=kernelpos[int(0.75*len(kernelpos)):int(0.9*len(kernelpos))]
traza = plt.figure(2)
plt.plot(np.arange(len(kernelpos_b)),kernelpos_b)
traza.savefig(F"traza_exp3_b{k}.png")
kernelpos_c=kernelpos[int(0.92*len(kernelpos)):]
traza = plt.figure(3)
plt.plot(np.arange(len(kernelpos_c)),kernelpos_c)
traza.savefig(F"traza_exp3_c{k}.png")
"""

#FIND MAP
mapindex = np.where( kernelpos== np.amax(kernelpos))[0][0]
map = samples[mapindex]
np.savetxt(F"map_exp3_{k}.txt",map)

#PSAMPLES WITHOUT REPETITION
samples= np.reshape(samples,(len(samples),3))
samples = np.unique(samples,axis=0)

#POSTERIOR PLOT
plot_posterior= corner.corner(samples, bins=40,
labels=[Fr"$x_{k}$", Fr"$y_{k}$", Fr"$z_{k}$"],
title_kwargs={"fontsize": 12},fontweight='bold', color = 'black')
value2 = np.mean(samples, axis=0)
value3 = np.median(samples, axis=0)
value4 = np.std(samples,axis=0)
np.savetxt(F"media_media_exp3_{k}.txt",value2)
np.savetxt(F"mediana_mediana_exp3_{k}.txt",value3)
np.savetxt(F"sd_posterior_exp3_{k}.txt",value4)
corner.overplot_lines(plot_posterior, value3, color="blue")
corner.overplot_points(plot_posterior, value3[None], marker="s", color="blue")
corner.overplot_lines(plot_posterior, map, color="orange")
corner.overplot_points(plot_posterior, map[None], marker="s", color="orange")
corner.overplot_lines(plot_posterior, value2, color="lightcoral",linewidth=1.0)
corner.overplot_points(plot_posterior, value2[None], marker="s", color="lightcoral")
plot_posterior.savefig(F"grafica_posterior_exp3_{k}.png")