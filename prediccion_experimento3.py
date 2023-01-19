import numpy as np
from scipy.integrate import solve_ivp


#DATA
#STEP OF THE ALGORITHM
k = 2
#DAYS OF LEARNING
a = 4
#DAYS OF RELAY
r = a/2
#DAYS OF PREDICTION AND FIT
b = 20

#CHARGE SINTHETYC DATA
time = np.loadtxt("datot_exp3.txt")
datax = np.loadtxt("datox_exp3.txt")
datay = np.loadtxt("datoy_exp3.txt")
dataz = np.loadtxt("datoz_exp3.txt")

#DATA ON THE INTERVALS OF PREDICTION (we have 10 points per day)
time = time[int(k*r*10):int((k*r+b)*10)]
datax = datax[int(k*r*10):int((k*r+b)*10)]
datay = datay[int(k*r*10):int((k*r+b)*10)]
dataz = dataz[int(k*r*10):int((k*r+b)*10)]

#RECOVER SAMPLES
samples = np.loadtxt(F"muestras_exp3_{k}.txt")
#RECOVER LOG-KERNEL AND SAMPLES WITH BURN 
samples = samples[int(0.2*len(samples)):]
#SAMPLES WITHOUT REPETITION
samples = np.reshape(samples,(len(samples),3))
samples = np.unique(samples,axis=0)

#Solution of the ODE
#Dinamical System
def model(t,L):
    x,y,z=L
    return [10.0*(y - x),28.0*x - y - x*z,x*y - (8/3)*z ]
#Jacobian Matrix of the Model
def jacobian_model(t,L):
    x,y,z = L
    return [[-10.0,10.0,0],[28.0-z,-1,-x],[y,x,-8/3]]

#SOLUTION FOR PREDICTION (IT HAS 100 POINTS PER DAY)
btime = np.arange(time[0],time[-1], 0.01)
def solution_model(c):
    #c = initial condition
    #t = data time
    return solve_ivp(model,t_span=(time[0],time[-1]),y0=c,dense_output=True,jac=jacobian_model,method="LSODA").sol(btime)

#SOLUTION FOR THE MAP
print("SOLVING FOR MAP")
map = np.loadtxt(F"map_exp3_{k}.txt")
solmap = solution_model(map)
np.savetxt(F"sol_map_exp3_{k}.txt",solmap)
print("SOLUTION OF MAP STORED")

#SOLUTION FOR EVERY INITIAL CONDITION
print("SOLVING ODE FOR EVERY SAMPLE")
matriz_x = np.zeros((len(samples),len(btime)))
matriz_y = np.zeros((len(samples),len(btime)))
matriz_z = np.zeros((len(samples),len(btime)))
for i in range(len(samples)):
    solution = solution_model(samples[i])
    matriz_x[i,:] = solution[0]
    matriz_y[i,:] = solution[1]
    matriz_z[i,:] = solution[2]
print("EQUATIONS SOLVED")

#QUANTILES 5%, 25%, 50%, 75%, 95%
#X
print("STORING QUANTILES")
mediana_x = np.median(matriz_x,axis=0)
q_5_x = np.quantile(matriz_x,0.05,axis=0)
q_25_x = np.quantile(matriz_x,0.25,axis=0)
q_75_x = np.quantile(matriz_x,0.75,axis=0)
q_95_x = np.quantile(matriz_x,0.95,axis=0)
np.savetxt(F"cuantil_5_x_exp3_{k}.txt",q_5_x)
np.savetxt(F"cuantil_25_x_exp3_{k}.txt",q_25_x)
np.savetxt(F"cuantil_75_x_exp3_{k}.txt",q_75_x)
np.savetxt(F"cuantil_95_x_exp3_{k}.txt",q_95_x)
np.savetxt(F"mediana_x_exp3_{k}.txt",mediana_x)
#Y
mediana_y = np.median(matriz_y,axis=0)
q_5_y = np.quantile(matriz_y,0.05,axis=0)
q_25_y = np.quantile(matriz_y,0.25,axis=0)
q_75_y = np.quantile(matriz_y,0.75,axis=0)
q_95_y = np.quantile(matriz_y,0.95,axis=0)
np.savetxt(F"cuantil_5_y_exp3_{k}.txt",q_5_y)
np.savetxt(F"cuantil_25_y_exp3_{k}.txt",q_25_y)
np.savetxt(F"cuantil_75_y_exp3_{k}.txt",q_75_y)
np.savetxt(F"cuantil_95_y_exp3_{k}.txt",q_95_y)
np.savetxt(F"mediana_y_exp3_{k}.txt",mediana_y)
#Z
mediana_z = np.median(matriz_z,axis=0)
q_5_z = np.quantile(matriz_z,0.05,axis=0)
q_25_z = np.quantile(matriz_z,0.25,axis=0)
q_75_z = np.quantile(matriz_z,0.75,axis=0)
q_95_z = np.quantile(matriz_z,0.95,axis=0)
np.savetxt(F"cuantil_5_z_exp3_{k}.txt",q_5_z)
np.savetxt(F"cuantil_25_z_exp3_{k}.txt",q_25_z)
np.savetxt(F"cuantil_75_z_exp3_{k}.txt",q_75_z)
np.savetxt(F"cuantil_95_z_exp3_{k}.txt",q_95_z)
np.savetxt(F"mediana_z_exp3_{k}.txt",mediana_z)
print("QUANTILES HAVE BEEN STORED")

#RECOVER PRIOR DATA
print("RECOVERING THE DATA FOR THE NEXT PRIOR")
prior_x = matriz_x[:,int(r*100)]
prior_y = matriz_y[:,int(r*100)]
prior_z = matriz_z[:,int(r*100)]
p=k+1
np.savetxt(F"info_prior_x_exp3_{p}.txt",prior_x)
np.savetxt(F"info_prior_y_exp3_{p}.txt",prior_y)
np.savetxt(F"info_prior_z_exp3_{p}.txt",prior_z)
print("DATA FOR THE NEXT PRIOR HAS BEEN STORED")
