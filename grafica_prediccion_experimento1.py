import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#DATA
#Step of the algorithm
k = 4
#Days of learning
a = 2
#Days of delay
r = a/2
#DAYS OF PREDICTION AND FIT IN GRAPHS
b = 20

#DATA (WE HAVE 10 POINT PER DAY)
times = np.loadtxt("datot_exp1.txt")
datax = np.loadtxt("datox_exp1.txt")
datay = np.loadtxt("datoy_exp1.txt")
dataz = np.loadtxt("datoz_exp1.txt")

#DATA ON THE INTERVALS OF PREDICTION 
time = times[int(k*r*10):int((k*r+b)*10)]
datax = datax[int(k*r*10):int((k*r+b)*10)]
datay = datay[int(k*r*10):int((k*r+b)*10)]
dataz = dataz[int(k*r*10):int((k*r+b)*10)]

#TIMES OF PREDICTION (WE HAVE 100 POINT PER DAY)
btime = np.arange(time[0],time[-1], 0.01)

#RECOVER THE QUATILES
q_5_x = np.loadtxt(F"cuantil_5_x_exp1_{k}.txt")
q_25_x = np.loadtxt(F"cuantil_25_x_exp1_{k}.txt")
q_75_x = np.loadtxt(F"cuantil_75_x_exp1_{k}.txt")
q_95_x = np.loadtxt(F"cuantil_95_x_exp1_{k}.txt")
mediana_x =np.loadtxt(F"mediana_x_exp1_{k}.txt")
#Y
q_5_y = np.loadtxt(F"cuantil_5_y_exp1_{k}.txt")
q_25_y = np.loadtxt(F"cuantil_25_y_exp1_{k}.txt")
q_75_y = np.loadtxt(F"cuantil_75_y_exp1_{k}.txt")
q_95_y = np.loadtxt(F"cuantil_95_y_exp1_{k}.txt")
mediana_y = np.loadtxt(F"mediana_y_exp1_{k}.txt")
#Z
q_5_z =np.loadtxt(F"cuantil_5_z_exp1_{k}.txt")
q_25_z =np.loadtxt(F"cuantil_25_z_exp1_{k}.txt")
q_75_z =np.loadtxt(F"cuantil_75_z_exp1_{k}.txt")
q_95_z =np.loadtxt(F"cuantil_95_z_exp1_{k}.txt")
mediana_z =np.loadtxt(F"mediana_z_exp1_{k}.txt")

#MAP
sol_map = np.loadtxt(F"sol_map_exp1_{k}.txt")

#FRACTION OF DAYS TO PLOT PREDICTION MEDIAN
v = 0.4

#MARKS FOR XLABEL
lab = [i for i in range(int(k*r),int(b+k*r))]
#PLOT OF PREDICTION OF X
plt_X = plt.figure(1)
#DATA
#REGION OF LEARNING
plt.plot(time[:int(a*10)],datax[:int(a*10)],"o",markersize =3,color = "black")
plt.axvline(time[int(a*10)]-0.03,color = 'black',linestyle='--',linewidth=1.5)
plt.axvline(time[0],color = 'black',linestyle='--',linewidth=1.5)
#REGION OF NOWCASTING AND DELAY
plt.plot(time[int(a*10):int((a+2)*10)],datax[int(a*10):int((a+2)*10)],"o",markersize =3,color = "red")
plt.axvline(time[int(a*10)]+0.03,color = 'red',linestyle='--',linewidth=1.5)
plt.axvline(time[int((a+2)*10)]-0.03,color = 'red',linestyle='--',linewidth=1.5)
#REGION OF PREDICTION
plt.plot(time[int((a+2)*10):],datax[int((a+2)*10):],"o",markersize =3,color = "green")
plt.axvline(time[int((a+2)*10)]+0.03,color = 'green',linestyle='--',linewidth=1.5)
plt.axvline(time[int(v*b*10)],color = 'green',linestyle='--',linewidth=1.5)
#FORECAST
plt.plot(btime,mediana_x,color = "royalblue",linewidth=1.5)
plt.fill_between(btime,q_5_x,q_95_x,color="royalblue",alpha=0.3)
plt.fill_between(btime,q_25_x,q_75_x,color="royalblue",alpha=0.3)
plt.plot(btime,sol_map[0],color = "darkorange",linewidth=1.5,alpha=0.7)
#SAVE PLOT
plt.xticks(lab)
plt.ylim([-20, 20])
plt.xlim([time[0]-0.07,time[int(v*b*10)]+0.1])
plt.xlabel(r"$t$",fontweight='bold', color = 'black')
plt.ylabel(r"$x$",fontweight='bold', color = 'black')
plt_X.savefig(F"grafica_prediccion_mediana_x_exp1_{k}.png")

#PLOT OF PREDICTION OF Y
plt_Y = plt.figure(2)
#DATA
#REGION OF LEARNING
plt.plot(time[:int(a*10)],datay[:int(a*10)],"o",markersize =3,color = "black")
plt.axvline(time[int(a*10)]-0.03,color = 'black',linestyle='--',linewidth=1.5)
plt.axvline(time[0],color = 'black',linestyle='--',linewidth=1.5)
#REGION OF NOWCASTING AND DELAY
plt.plot(time[int(a*10):int((a+2)*10)],datay[int(a*10):int((a+2)*10)],"o",markersize =3,color = "red")
plt.axvline(time[int(a*10)]+0.03,color = 'red',linestyle='--',linewidth=1.5)
plt.axvline(time[int((a+2)*10)]-0.03,color = 'red',linestyle='--',linewidth=1.5)
#REGION OF PREDICTION
plt.plot(time[int((a+2)*10):],datay[int((a+2)*10):],"o",markersize =3,color = "green")
plt.axvline(time[int((a+2)*10)]+0.03,color = 'green',linestyle='--',linewidth=1.5)
plt.axvline(time[int(v*b*10)],color = 'green',linestyle='--',linewidth=1.5)
#FORECAST
plt.plot(btime,mediana_y,color = "royalblue",linewidth=1.5)
plt.fill_between(btime,q_5_y,q_95_y,color="royalblue",alpha=0.3)
plt.fill_between(btime,q_25_y,q_75_y,color="royalblue",alpha=0.3)
plt.plot(btime,sol_map[1],color = "darkorange",linewidth=1.5,alpha=0.7)
#SAVE PLOT
plt.xticks(lab)
plt.ylim([-25, 25])
plt.xlim([time[0]-0.07,time[int(v*b*10)]+0.1])
plt.xlabel(r"$t$",fontweight='bold', color = 'black')
plt.ylabel(r"$y$",fontweight='bold', color = 'black')
plt_Y.savefig(F"grafica_prediccion_mediana_y_exp1_{k}.png")

#PLOT OF PREDICTION OF Z
plt_Z = plt.figure(3)
#DATA
#REGION OF LEARNING
plt.plot(time[:int(a*10)],dataz[:int(a*10)],"o",markersize =3,color = "black")
plt.axvline(time[int(a*10)]-0.03,color = 'black',linestyle='--',linewidth=1.5)
plt.axvline(time[0],color = 'black',linestyle='--',linewidth=1.5)
#REGION OF NOWCASTING AND DELAY
plt.plot(time[int(a*10):int((a+2)*10)],dataz[int(a*10):int((a+2)*10)],"o",markersize =3,color = "red")
plt.axvline(time[int(a*10)]+0.03,color = 'red',linestyle='--',linewidth=1.5)
plt.axvline(time[int((a+2)*10)]-0.03,color = 'red',linestyle='--',linewidth=1.5)
#REGION OF PREDICTION
plt.plot(time[int((a+2)*10):],dataz[int((a+2)*10):],"o",markersize =3,color = "green")
plt.axvline(time[int((a+2)*10)]+0.03,color = 'green',linestyle='--',linewidth=1.5)
plt.axvline(time[int(v*b*10)],color = 'green',linestyle='--',linewidth=1.5)
#FORECAST
plt.plot(btime,mediana_z,color = "royalblue",linewidth=1.5)
plt.fill_between(btime,q_5_z,q_95_z,color="royalblue",alpha=0.3)
plt.fill_between(btime,q_25_z,q_75_z,color="royalblue",alpha=0.3)
plt.plot(btime,sol_map[2],color = "darkorange",linewidth=1.5,alpha=0.7)
#SAVE PLOT
plt.xticks(lab)
plt.ylim([0, 50])
plt.xlim([time[0]-0.07,time[int(v*b*10)]+0.1])
plt.xlabel(r"$t$",fontweight='bold', color = 'black')
plt.ylabel(r"$z$",fontweight='bold', color = 'black')
plt_Z.savefig(F"grafica_prediccion_mediana_z_exp1_{k}.png")