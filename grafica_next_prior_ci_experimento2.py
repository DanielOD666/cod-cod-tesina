from random import sample
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import corner

k = 2
p=k+1
#RECOVER INFORMATION FOR PRIOR
info_prior_x = np.loadtxt(F"info_prior_x_exp2_{p}.txt")
info_prior_y = np.loadtxt(F"info_prior_y_exp2_{p}.txt")
info_prior_z = np.loadtxt(F"info_prior_z_exp2_{p}.txt")
matriz_info_prior = np.zeros((3,len(info_prior_x)))
matriz_info_prior[0,:] = info_prior_x
matriz_info_prior[1,:] = info_prior_y
matriz_info_prior[2,:] = info_prior_z 
matriz_info_prior = matriz_info_prior.T

#MEDIAN, MEAN AND STANDAR DEVIATION
value1x = np.median(info_prior_x)
value2x = np.std(info_prior_x)

value1y = np.median(info_prior_y)
value2y = np.std(info_prior_y)

value1z = np.median(info_prior_z)
value2z = np.std(info_prior_z)

#PRIOR K+1 PLOT
plot_prior = corner.corner(matriz_info_prior,
bins=100,
labels=[Fr"$x_{p}$", Fr"$y_{p}$", Fr"$z_{p}$"],
title_kwargs={"fontsize": 12},fontweight='bold', color = 'black')
plot_prior.savefig(F"grafica_prior_ci_exp2_{p}.png")