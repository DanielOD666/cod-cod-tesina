import numpy as np
from scipy.integrate import solve_ivp

    
def model(t,L):
    x,y,z=L
    return [10.0*(y - x),28.0*x - y - x*z,x*y - (8/3)*z ]
#Jacobian Matrix of the Model
def jacobian_model(t,L):
    x,y,z = L
    return [[-10.0,10.0,0],[28.0-z,-1,-x],[y,x,-8/3]]
#solution
#c = initial condition and rayleigh
#t = data time
a = solve_ivp(model,t_span=(0,1000),y0=[0,1,0],jac=jacobian_model,method="LSODA")    
x=a.y[0]
y=a.y[1]
z=a.y[2]
np.savetxt("construir_primera_prior_x_exp1.txt",x)
np.savetxt("construir_primera_prior_y_exp1.txt",y)
np.savetxt("construir_primera_prior_z_exp1.txt",z)