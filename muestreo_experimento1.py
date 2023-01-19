import numpy as np
import base_experimento1 as base
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import corner
"""
input : Noisy data
output : samples and log-kernel-posterior without burn and it's plot
"""

#DATA
#Step of the algorithm
k = 4
#Days of learning
a = 2
#Days of delay
r = a/2
#Charge the syntetic data
times = np.loadtxt("datot_exp1.txt")
dataxx = np.loadtxt("datox_exp1.txt")
datayy = np.loadtxt("datoy_exp1.txt")
datazz = np.loadtxt("datoz_exp1.txt")
sdx = np.loadtxt("sdx_exp1.txt")[0]
sdy = np.loadtxt("sdy_exp1.txt")[0]
sdz = np.loadtxt("sdz_exp1.txt")[0]
varx = sdx**2
vary = sdy**2
varz = sdz**2
invvarx = 1/varx
invvary = 1/vary
invvarz = 1/varz
#Data on the intervals of learning (we have 10 points per day)
timess = times[int(k*r*10):int((k*r+a)*10)]
datax = dataxx[int(k*r*10):int((k*r+a)*10)]
datay = datayy[int(k*r*10):int((k*r+a)*10)]
dataz = datazz[int(k*r*10):int((k*r+a)*10)]


#Solution of the ODE
#Dinamical System
def model(t,L):
    x,y,z=L
    return [10.0*(y - x),28.0*x - y - x*z,x*y - (8/3)*z ]
#Jacobian Matrix of the Model
def jacobian_model(t,L):
    x,y,z = L
    return [[-10.0,10.0,0],[28.0-z,-1,-x],[y,x,-8/3]]
#solution
def solution_model(c):
    #c = initial condition
    #t = data time
    return solve_ivp(model,t_span=(timess[0],timess[-1]),y0=c,dense_output=True,jac=jacobian_model,method="LSODA").sol(timess)    

#LOG_POSTERIOR
#Log-Likelihood   
def log_likelihood(c):
    #c = initial conditions
    loglik = -0.5*(invvarx*np.sum((datax-solution_model(c)[0])**2)
    +invvary*np.sum((datay-solution_model(c)[1])**2)
    +invvarz*np.sum((dataz-solution_model(c)[2])**2))
    return loglik
"""
#log prior (it depends on the step)
#k=0
#X normal
mex=-0.1958365
sdx=8.0575685
varx=sdx**2
invvarx=1/varx
#Y normal
mey=-0.1734002
sdy=7.732907
vary=sdy**2
invary=1/vary
#Z Normal
mez=26.50877
sdz=8.84641
varz=sdz**2
invarz=1/varz
"""

#k larger than 0
info_prior_x = np.loadtxt(F"info_prior_x_exp1_{k}.txt")
info_prior_y = np.loadtxt(F"info_prior_y_exp1_{k}.txt")
info_prior_z = np.loadtxt(F"info_prior_z_exp1_{k}.txt")
mediana_prior_x=np.median(info_prior_x)
sd_prior_x=np.std(info_prior_x)
mediana_prior_y=np.median(info_prior_y)
sd_prior_y=np.std(info_prior_y)
mediana_prior_z=np.median(info_prior_z)
sd_prior_z=np.std(info_prior_z)
#X normal
mex=mediana_prior_x
varx=sd_prior_x**2
invvarx=1/varx
#Y normal
mey=mediana_prior_y
vary=sd_prior_y**2
invary=1/vary
#Z Normal
mez=mediana_prior_z
varz=sd_prior_z**2
invarz=1/varz

def log_prior(c):
    log_pr = -0.5*invvarx*(c[0]-mex)
    -0.5*invvary*(c[1]-mey) 
    -0.5*invvarz*(c[2]-mez) 
    return log_pr

#log posterior
def log_posterior(c):
    return log_likelihood(c) + log_prior(c)

# PARAMETERS OF SAMPLER AND SAMPLES
p=3;#number of parameters
N=400000 #number of samples
N_temp=6;#number of temperatures
Ns=1; #How often do we swap
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]); #preallocates proposals
bt=np.zeros([N,N_temp])
#TEMPERATURES
T0=20000.0**(1.0/5.0)
beta = np.array(T0**np.arange(0,N_temp))
beta_original = np.copy(beta)

#INITIAL POINT

x0=np.zeros((p,N_temp))
"""
x0[0]=np.random.uniform(-7.14220379,-7.14220378,(1,N_temp))
x0[1]=np.random.uniform(-12.0024063,-12.0024062,(1,N_temp))
x0[2]=np.random.uniform(12.8998525,12.8998524,(1,N_temp))
"""

x0[0]=np.random.uniform(mex,mex,(1,N_temp))
x0[1]=np.random.uniform(mey,mey,(1,N_temp))
x0[2]=np.random.uniform(mez,mez,(1,N_temp))

#JUMP PROPOSALS
"""
sigma_is=np.array([[0.015,0.015,0.015],
[0.06,0.06,0.06],
[0.15,0.15,0.15],
[0.5,0.5,0.5],
[2.0,2.0,2.0]])
"""
"""
sigma_is=np.array([[0.0045,0.0045,0.0045],
[0.02,0.02,0.02],
[0.06,0.06,0.06],
[0.16,0.16,0.16],
[0.45,0.45,0.45],
[1.2,1.2,1.2]])
"""
"""
sigma_is=np.array([[0.00008,0.00008,0.00008],
[0.0002,0.0002,0.0002],
[0.00055,0.00055,0.00055],
[0.02,0.02,0.02],
[0.073,0.073,0.073],
[0.22,0.22,0.22]])
"""
"""
sigma_is=np.array([[0.0000055,0.0000055,0.0000055],
[0.000055,0.000055,0.000055],
[0.0002,0.0002,0.0002],
[0.0005,0.0005,0.0005],
[0.0035,0.0035,0.0035],
[0.0115,0.0115,0.0115]])
"""

sigma_is=np.array([[0.00000003,0.00000003,0.00000003],
[0.00000031,0.00000031,0.00000031],
[0.0000025,0.0000025,0.0000025],
[0.0000091,0.0000091,0.0000091],
[0.00018,0.00018,0.00018],
[0.0035,0.0035,0.0035]])

#SAMPLE
#Xuw=base.unweighted_IS(log_posterior,N,beta,sigma_is,x0,1,1,Disp=1) #unweighted Is
Xptf=base.parallel_tempering(log_posterior,N,beta,sigma_is,x0,'full_random',Ns=1,Disp=1)

#SAMPLES, LOGKERNEL AND TRACE
#samples = Xuw[:,:,0]
samples = Xptf[:,:,0]
kernelpos = np.zeros(len(samples))
for i in range(len(samples)):
    kernelpos[i]=log_posterior(samples[i])
#STORE DATA
np.savetxt(F"muestras_exp1_{k}.txt",samples)
np.savetxt(F"log_kernel_exp1_{k}.txt",kernelpos)
#TRACE
traza = plt.figure(0)
plt.plot(np.arange(len(kernelpos)),kernelpos)
traza.savefig(F"traza_exp1_{k}.png")
