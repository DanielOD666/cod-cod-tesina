from random import sample
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import corner


#DATA
#STEP OF THE ALGORITHM
k = 4
p=k+1
#RECOVER INFORMATION FOR PRIOR
info_prior_x = np.loadtxt(F"info_prior_x_exp1_{p}.txt")
info_prior_y = np.loadtxt(F"info_prior_y_exp1_{p}.txt")
info_prior_z = np.loadtxt(F"info_prior_z_exp1_{p}.txt")
matriz_info_prior = np.zeros((3,len(info_prior_x)))
matriz_info_prior[0,:] = info_prior_x
matriz_info_prior[1,:] = info_prior_y
matriz_info_prior[2,:] = info_prior_z 
matriz_info_prior = matriz_info_prior.T

#PRIOR K+1 PLOT
plot_prior = corner.corner(matriz_info_prior, bins=40,
labels=[Fr"$x_{p}$", Fr"$y_{p}$", Fr"$z_{p}$"],
title_kwargs={"fontsize": 12},fontweight='bold', color = 'black')
plot_prior.savefig(F"grafica_prior_exp1_{p}.png")
