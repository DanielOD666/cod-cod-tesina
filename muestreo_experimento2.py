import numpy as np
import base_experimento2 as base
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import corner
"""
input : Noisy data
output : samples and log-kernel-posterior without burn and it's plot
"""
#DATA
#Step of the algorithm
k = 2
#Days of learning
a = 4
#Days after
r = a/2
#Charge the syntetic data
times = np.loadtxt("datot_exp2.txt")
dataxx = np.loadtxt("datox_exp2.txt")
datayy = np.loadtxt("datoy_exp2.txt")
datazz = np.loadtxt("datoz_exp2.txt")
sdx = np.loadtxt("sdx_exp2.txt")[0]
sdy = np.loadtxt("sdy_exp2.txt")[0]
sdz = np.loadtxt("sdz_exp2.txt")[0]
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
def solution_model(c):
    #Dinamical System
    def model(t,L):
        x,y,z=L
        return [10.0*(y - x),c[3]*x - y - x*z,x*y - (8/3)*z ]
    #Jacobian Matrix of the Model
    def jacobian_model(t,L):
        x,y,z = L
        return [[-10.0,10.0,0],[c[3]-z,-1,-x],[y,x,-8/3]]
        #solution
        #c = initial condition and rayleigh
        #t = data time
    return solve_ivp(model,t_span=(timess[0],timess[-1]),y0=[c[0],c[1],c[2]],dense_output=True,jac=jacobian_model,method="LSODA").sol(timess)    

#LOG_POSTERIOR
#Log-Likelihood   
def log_likelihood(c):
    #c = initial conditions
    loglik = -0.5*(invvarx*np.sum((datax-solution_model(c)[0])**2)
    +invvary*np.sum((datay-solution_model(c)[1])**2)
    +invvarz*np.sum((dataz-solution_model(c)[2])**2))
    return loglik

#log prior (it depends on the step)

#k larger than 0
info_prior_x = np.loadtxt(F"info_prior_x_exp2_{k}.txt")
info_prior_y = np.loadtxt(F"info_prior_y_exp2_{k}.txt")
info_prior_z = np.loadtxt(F"info_prior_z_exp2_{k}.txt")
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
#r Normal
mediana_prior_r=np.loadtxt(F"median_prior_r_exp2_{k}.txt")[0]
sd_prior_r=np.loadtxt(F"sd_prior_r_exp2_{k}.txt")[0]
mer=mediana_prior_r
sdr=sd_prior_r
varr=sdr**2
invvarr=1/varr

"""
def log_prior(c):    
    log_pr =np.log(float(np.all(c[0]>-100))*float(np.all(c[0]<100)))
    +np.log(float(np.all(c[1]>-100))*float(np.all(c[1]<100)))
    +np.log(float(np.all(c[2]>0))*float(np.all(c[2]<150)))
    np.log(float(np.all(c[3]>0))*float(np.all(c[3]<100)))
    return log_pr

"""
def log_prior(c):    
    log_pr =-0.5*invvarx*(c[0]-mex)
    -0.5*invvary*(c[1]-mey)
    -0.5*invvarz*(c[2]-mez)
    -0.5*invvarr*(c[3]-mer)
    return log_pr

#log posterior
def log_posterior(c):
    return log_likelihood(c) + log_prior(c)

# PARAMETERS OF SAMPLER AND SAMPLES
p=4#number of parameters
N=800000 #number of samples
N_temp=7#number of temperatures
Ns=1 #How often do we swap
X=np.zeros([N,p,N_temp])#prealloactes
y=np.zeros([N_temp,p]) #preallocates proposals
bt=np.zeros([N,N_temp])

#TEMPERATURES
T0=19000.0**(1.0/6.0)
beta = np.array(T0**np.arange(0,N_temp))
beta_original = np.copy(beta)
#INITIAL POINT

"""
x0=np.zeros((p,N_temp))
x0[0]=np.random.uniform(8.01859978,8.01859979,(1,N_temp))
x0[1]=np.random.uniform(9.06194448,9.06194449,(1,N_temp))
x0[2]=np.random.uniform(61.58440661,61.58440662,(1,N_temp))
x0[3]=np.random.uniform(69.99494788,69.9949479,(1,N_temp))
"""

x0=np.zeros((p,N_temp))
x0[0]=np.random.uniform(mex,mex,(1,N_temp))
x0[1]=np.random.uniform(mey,mey,(1,N_temp))
x0[2]=np.random.uniform(mez,mez,(1,N_temp))
x0[3]=np.random.uniform(mediana_prior_r,mediana_prior_r,(1,N_temp))

"""
#JUMP PROPOSALS
sigma_is=np.array([[0.0003,0.0003,0.0003,0.0003],
[0.0008,0.0008,0.0008,0.0008],
[0.005,0.005,0.005,0.005],
[0.1,0.1,0.1,0.1],
[0.4,0.4,0.4,0.4],
[12.0,12.0,12.0,12.0]])
"""

"""
sigma_is=np.array([[0.000005,0.000005,0.000005,0.000005],
[0.000021,0.000021,0.000021,0.000021],
[0.000055,0.000055,0.000055,0.000055],
[0.000094,0.000094,0.000094,0.000094],
[0.0005,0.0005,0.0005,0.0005],
[0.03,0.03,0.03,0.03],
[0.179,0.179,0.179,0.179]])
"""

sigma_is=np.array([[0.000000000002,0.000000000002,0.000000000002,0.000000000002],
[0.000000000004,0.000000000004,0.000000000004,0.000000000004],
[0.000000000008,0.000000000008,0.000000000008,0.000000000008],
[0.000000000033,0.000000000033,0.000000000033,0.000000000033],
[0.000000005,0.000000005,0.000000005,0.000000005],
[0.00000006,0.00000006,0.00000006,0.00000006],
[0.0000065,0.0000065,0.0000065,0.0000065]])


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
np.savetxt(F"muestras_exp2_{k}.txt",samples)
np.savetxt(F"log_kernel_exp2_{k}.txt",kernelpos)
#TRACE
traza = plt.figure(0)
plt.plot(np.arange(len(kernelpos)),kernelpos)
traza.savefig(F"traza_exp2_{k}.png")