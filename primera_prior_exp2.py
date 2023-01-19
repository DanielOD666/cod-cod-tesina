import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

def solution_model(r):
    #Dinamical System
    def model(t,L):
        x,y,z=L
        return [10.0*(y - x),r*x - y - x*z,x*y - (8/3)*z ]
    #Jacobian Matrix of the Model
    def jacobian_model(t,L):
        x,y,z = L
        return [[-10.0,10.0,0],[r-z,-1,-x],[y,x,-8/3]]
        #solution
        #c = initial condition and rayleigh
        #t = data time
    return solve_ivp(model,t_span=(0,1000),y0=[0,1,0],jac=jacobian_model,method="LSODA").y   

x=[]
y=[]
z=[]
for r in range(25,99):
    solution=solution_model(r)
    for i in range(0,len(solution[0])):
        x.append(solution[0][i])
        y.append(solution[1][i])
        z.append(solution[2][i])
        
np.savetxt("construir_primera_prior_x_exp2.txt",x)
np.savetxt("construir_primera_prior_y_exp2.txt",y)
np.savetxt("construir_primera_prior_z_exp2.txt",z)
