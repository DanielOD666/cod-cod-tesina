from decimal import DefaultContext
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math 
import random


#Definition of the dinamical system : The Lorenz system
def modelo(t,L):
    x,y,z=L
    return [10.0*(y - x),70.0*x - y - x*z,x*y - (8/3)*z ]

#Jacobian matrix
def jacobiano_modelo(t,L):
    x,y,z = L
    return [[-10.0,10.0,0],[70.0-z,-1,-x],[y,x,-8/3]]

#Resuelvo el problema hasta t=10000 con condición inicial alguna condición inicial
s = solve_ivp(modelo,t_span=(0,10000),y0=[4,7,3],jac=jacobiano_modelo,method="Radau").y

#Resuelvo el problema de  t=10100, con condición inicial (x[10000],y[10000],z[10000]) y guardo las soluciones 
t = np.arange(10000, 10100.00001, 0.002)
solucion = solve_ivp(modelo,t_span=(10000,10100.00001),y0=[s[0][-1],s[1][-1],s[2][-1]],t_eval=t,jac=jacobiano_modelo,method="Radau").y
del(s)

#Desviaci0nes estandar usando la relación señal a ruido
p=[]
pp=[]
ppp=[]
for j in range(0,len(solucion[0])):
    p.append(abs(solucion[0][j]))
    pp.append(abs(solucion[1][j]))
    ppp.append(abs(solucion[2][j]))
max_x = max(p)
max_y = max(pp)
max_z = max(ppp)
dsx=max_x/100
dsy=max_y/100
dsz=max_z/100
np.savetxt("sdx_exp2.txt",[dsx,0])
np.savetxt("sdy_exp2.txt",[dsy,0])
np.savetxt("sdz_exp2.txt",[dsz,0])


#Fijamos la seed y definimos los vectores de ruido
np.random.seed(1)
ruidox = np.random.normal(0, dsx, size=len(solucion[0]))
np.random.seed(3)
ruidoy = np.random.normal(0, dsy, size=len(solucion[1]))
np.random.seed(7)
ruidoz = np.random.normal(0, dsz, size=len(solucion[2]))

#Definimos el vector de datos
datox = solucion[0] + ruidox
datoy = solucion[1] + ruidoy
datoz = solucion[2] + ruidoz

#Elección de los 1000 datos
#Regrsar el tiempo a 0
t = t - 10000
#Redondear tiempo
tiempo1=[]
for i in range(0,len(t)):
    tiempo1.append(round(t[i],1))
tiempo=[]
#Elegir tiempo
for i in range(0,len(tiempo1),50):
    tiempo.append(tiempo1[i])
np.savetxt("datot_exp2.txt",tiempo)
#Elegir datos de X,Y y Z
x=[]
for i in range(0,len(datox),50):
    x.append(datox[i])
np.savetxt("datox_exp2.txt",x)
y=[]
for i in range(0,len(datoy),50):
    y.append(datoy[i])
np.savetxt("datoy_exp2.txt",y)
z=[]
for i in range(0,len(datoz),50):
    z.append(datoz[i])
np.savetxt("datoz_exp2.txt",z)

#Graficar los datos y la diferencia con los datos y el ruido
x1=[]
for i in range(0,len(solucion[0]),50):
    x1.append(solucion[0][i])
y1=[]
for i in range(0,len(solucion[1]),50):
    y1.append(solucion[1][i])
z1=[]
for i in range(0,len(solucion[2]),50):
    z1.append(solucion[2][i])

diferenciax = plt.figure(1)
plt.plot(tiempo,x,"o",markersize=2,color="red",label="Datos")
plt.plot(tiempo,x1,"o",markersize=2,color="green",label="Solución")
plt.xlabel("tiempo")
plt.ylabel("X")
plt.grid(True)
plt.legend()
plt.savefig("diferenciax_exp2.png")

diferenciay = plt.figure(2)
plt.plot(tiempo,y,"o",markersize=2,color="red",label="Datos")
plt.plot(tiempo,y1,"o",markersize=2,color="green",label="Solución")
plt.xlabel("tiempo")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.savefig("diferenciay_exp2.png")

diferenciaz = plt.figure(3)
plt.plot(tiempo,z,"o",markersize=2,color="red",label="Datos")
plt.plot(tiempo,z1,"o",markersize=2,color="green",label="Solución")
plt.xlabel("tiempo")
plt.ylabel("Z")
plt.grid(True)
plt.legend()
plt.savefig("diferenciaz_exp2.png")

datox = plt.figure(4)
plt.plot(tiempo,x,color="blueviolet",linewidth=0.8)
plt.xlabel(r"$t$")
plt.ylabel(r"$X$")
plt.xlim([0,50])
plt.savefig("datox_exp2.png")

datoy = plt.figure(5)
plt.plot(tiempo,y,color="blueviolet",linewidth=0.8)
plt.xlabel(r"$t$")
plt.ylabel(r"$Y$")
plt.xlim([0,50])
plt.savefig("datoy_exp2.png")

datoz = plt.figure(6)
plt.plot(tiempo,z,color="blueviolet",linewidth=0.8)
plt.xlabel(r"$t$")
plt.ylabel(r"$Z")
plt.xlim([0,50])
plt.savefig("datoz_exp2.png")

