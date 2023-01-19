from random import sample
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import corner


#DATA
#Step of the algorithm
k = 4
#RECOVER LOG-KERNEL OF POSTERIOR AND SAMPLES
kernelpos = np.loadtxt(F"log_kernel_exp1_{k}.txt")
samples = np.loadtxt(F"muestras_exp1_{k}.txt")
#RECOVER LOG-KERNEL AND SAMPLES WITH BURN
kernelpos = kernelpos[int(0.2*len(kernelpos)):] 
samples = samples[int(0.2*len(kernelpos)):]

#TRACE
traza = plt.figure(0)
plt.plot(np.arange(len(kernelpos)),kernelpos)
traza.savefig(F"traza_exp1_{k}.png")

#FIND MAP
mapindex = np.where( kernelpos== np.amax(kernelpos))[0][0]
map = samples[mapindex]
np.savetxt(F"map_exp1_{k}.txt",map)

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
np.savetxt(F"media_media_exp1_{k}.txt",value2)
np.savetxt(F"mediana_mediana_exp1_{k}.txt",value3)
np.savetxt(F"sd_posterior_exp1_{k}.txt",value4)
corner.overplot_lines(plot_posterior, value3, color="blue")
corner.overplot_points(plot_posterior, value3[None], marker="s", color="blue")
corner.overplot_lines(plot_posterior, map, color="orange")
corner.overplot_points(plot_posterior, map[None], marker="s", color="orange")
corner.overplot_lines(plot_posterior, value2, color="lightcoral",linewidth=1.0)
corner.overplot_points(plot_posterior, value2[None], marker="s", color="lightcoral")
plot_posterior.savefig(F"grafica_posterior_exp1_{k}.png")