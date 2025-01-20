# -*- coding: utf-8 -*-




!pip install rebound
!pip install reboundx

#Phase 1
#Bepi-Colombo Mission
#Imports
import rebound
import reboundx
import numpy as np
import matplotlib.pyplot as plt
from reboundx import constants
from scipy.interpolate import interp1d

params = {'legend.fontsize': 'xx-large',
          'figure.titlesize': 'xx-large',
          'figure.subplot.top': 0.93,
          'figure.subplot.wspace': 0.3,
          'figure.figsize': (7,6),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)

sim = rebound.Simulation()
sim.units = ('days', 'km', 'kg')



#Common parameters
Nobj = 13#considering 8 planets, sun, Bepi, 3 asteriods
Nobj2 = 5
year = 365 #in days
tmax = 2*year
Nout = int(3*tmax) #per day 3 samples
print(Nout)
sim.c = 3e5*86400 #km/day light speed
c2 = sim.c**2
cm2 = 1/c2
sim.G = 6.67430e-20*(86400**2) #gravitational constant km3 kg-1 days-2
times = np.linspace(0, tmax, Nout)

Jcentury = 36525 #days
arcsec = 4.8481368e-06 #in rad
period_mercury= 88 #days

# mercury_x= 46088671.22889903
# mercury_y= 17423615.515467957
# mercury_z= -2742278.1329949
# mercury_vx= -27.278071481898525
# mercury_vy= 47.50246889781066
# mercury_vz= 6.38427653379923

# earth_x= 116535299.24682154
# earth_y= 91090203.7090667
# earth_z= 7002.7180410809815
# earth_vx= -18.886395720072205
# earth_vy= 23.305303655847617
# earth_vz= -0.000608065810169478

# venus_x= 116535299.24682154
# venus_y= 91090203.7090667
# venus_z= 7002.7180410809815
# venus_vx= -18.886395720072205
# venus_vy= 23.305303655847617
# venus_vz= -0.000608065810169478

# bepi_x= 46061192.175843544
# bepi_y= 17252752.368236057
# bepi_z= -2736191.7237738287
# bepi_vx= -27.057110127709606
# bepi_vy= 47.26234809792773
# bepi_vz= 6.404299380352967

#defining simulation setup
def setupsim():
	sim = rebound.Simulation()
	sim.units = ('days', 'km', 'kg')
	sim.add(m = 1.989e30, hash = "sun")
	sim.add(m = 3.285e23, x=46088671.22889903, y=17423615.515467957, z=-2742278.1329949, vx=-27.278071481898525*86400, vy=47.50246889781066*86400, vz=6.38427653379923*86400, hash = "mercury")
	sim.add(m = 1150, x=46061192.175843544, y=17252752.368236057, z=-2736191.7237738287, vx=-27.057110127709606*86400, vy=47.26234809792773*86400, vz=6.404299380352967*86400, hash = "bepi")
	sim.add(m= 4.867e24, x=79010285.57229258, y=72893802.51865904,z= -3542286.046258427,vx=-23.949895449143668*86400, vy=25.496711631982823*86400,vz=1.7325795166332743*86400, hash='Venus')
	sim.add(m= 5.972e24, x=116535299.24682154, y=91090203.7090667,z=7002.7180410809815,vx=-18.886395720072205*86400, vy=23.305303655847617*86400,vz=-0.000608065810169478*86400, hash='Earth')
	sim.add(m=6.39e23, hash='Mars')
	sim.add(m=1.898e27, a=1.0704e6, e=0.0014, hash="jupiter")
	sim.add(m=5.683e26, hash='Saturn')
	sim.add(m=8.681e25, hash='Uranus')
	sim.add(m=1.024e26, hash='Neptune')
	sim.add(m=9.1e20, hash='Ceres')
	sim.add(m=2.108e20, hash='Pallas')
	sim.add(m=2.589e20, hash='Vesta')

	return sim

#defining simulation setup
def setupsim2():
	sim = rebound.Simulation()
	sim.units = ('days', 'km', 'kg')
	sim.add(m = 1.989e30, hash = "sun")
	sim.add(m = 3.285e23, x=46088671.22889903, y=17423615.515467957, z=-2742278.1329949, vx=-27.278071481898525*86400, vy=47.50246889781066*86400, vz=6.38427653379923*86400, hash = "mercury")
	sim.add(m = 1150, x=46061192.175843544, y=17252752.368236057, z=-2736191.7237738287, vx=-27.057110127709606*86400, vy=47.26234809792773*86400, vz=6.404299380352967*86400, hash = "bepi")
	# sim.add(m= 4.867e24, x=79010285.57229258, y=72893802.51865904,z= -3542286.046258427,vx=-23.949895449143668*86400, vy=25.496711631982823*86400,vz=1.7325795166332743*86400, hash='Venus')
	sim.add(m= 5.972e24, x=116535299.24682154, y=91090203.7090667,z=7002.7180410809815,vx=-18.886395720072205*86400, vy=23.305303655847617*86400,vz=-0.000608065810169478*86400, hash='Earth')
	# sim.add(m=6.39e23, hash='Mars')
	sim.add(m=1.898e27, a=1.0704e6, e=0.0014, hash="jupiter")
	# sim.add(m=5.683e26, hash='Saturn')
	# sim.add(m=8.681e25, hash='Uranus')
	# sim.add(m=1.024e26, hash='Neptune')
	# sim.add(m=9.1e20, hash='Ceres')
	# sim.add(m=2.108e20, hash='Pallas')
	# sim.add(m=2.589e20, hash='Vesta')

	return sim



"""##################NEWTON CASE############################

"""

sim0 = setupsim()
sim0.move_to_com()
ps0 = sim0.particles
sim0.integrator = "whfast"

x0 = np.zeros((Nobj,Nout))
y0 = np.zeros((Nobj,Nout))
z0 = np.zeros((Nobj,Nout))
a0 = np.zeros((Nobj,Nout))
e0 = np.zeros((Nobj,Nout))
Omega0 = np.zeros((Nobj,Nout))
omega0 = np.zeros((Nobj,Nout))
pomega0 = np.zeros((Nobj,Nout))
inc0 = np.zeros((Nobj,Nout))

from tqdm import tqdm
for i,time in tqdm(enumerate(times)): #getting all the observables
    sim0.integrate(time)
    for j in (range(Nobj)):
        x0[j][i] = ps0[j].x
        y0[j][i] = ps0[j].y
        z0[j][i] = ps0[j].z
        if j>0:
            a0[j][i] = ps0[j].a
            e0[j][i] = ps0[j].e
            pomega0[j][i] = ps0[j].pomega
            Omega0[j][i] = ps0[j].Omega
            omega0[j][i] = ps0[j].omega
            inc0[j][i] = ps0[j].inc

EM0 = np.sqrt((x0[1]-x0[4])**2+(y0[1]-y0[4])**2+(z0[1]-z0[4])**2)#earth-mercury distance in newton case
#EM0 = np.sqrt((x0[1]-x0[3])**2+(y0[1]-y0[3])**2+(z0[1]-z0[3])**2)#earth-mercury distance in newton case
BC0 = np.sqrt(x0[2]**2+y0[2]**2+z0[2]**2)#bepi colombo orbit in newton case with respect to Barycenter
BCE0 = np.sqrt((x0[2]-x0[4])**2+(y0[2]-y0[4])**2+(z0[2]-z0[4])**2)#earth-bepi distance
#no shapiro delay
print("Earth-mercury distance in netwon case with all perturber effects:", len(EM0))
#Plotting the earth-mercury distance and the orbit of the bepi colombo to see the impact of the perturbers

plt.figure()
plt.plot(times, EM0, label="Earth-Mercury distance", color='blue')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance in Newton case", fontsize='14')
plt.show()
###############################################################################
plt.plot(times, BCE0, label="Earth-Bepi distance", color='red')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Bepi Distance in Newton case", fontsize='14')
plt.show()

EM0

BCSSx = [p.x for p in sim0.particles[2:]]#[x,y,z,vx,vy,vz] in km and km/s
BCSSy = [p.y for p in sim0.particles[2:]]#[x,y,z,vx,vy,vz] in km and km/s
BCSSz = [p.z for p in sim0.particles[2:]]#[x,y,z,vx,vy,vz] in km and km/s
ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(BCSSx,BCSSy,BCSSz, color='red', label='Bepi-Colombo')
ax1.legend()
ax1.set_title('Orbit of Bepi-Colombo')
plt.show()





"""################GR from Reboundx##########################

"""

import reboundx
print("GR effect added to the system")
sim1 = setupsim2()
sim1.integrator = "whfast"
sim1.move_to_com()
ps1 = sim1.particles
Nobj = Nobj2

rebx = reboundx.Extras(sim1)
gr = rebx.load_force("gr")#adding gr effect
rebx.add_force(gr)
gr.params["c"] = 3e5*86400 #km/days
sim1.G = 6.67430e-20*(86400**2)
print("still running at GR")
x1 = np.zeros((Nobj,Nout))
y1 = np.zeros((Nobj,Nout))
z1 = np.zeros((Nobj,Nout))
a1 = np.zeros((Nobj,Nout))
e1 = np.zeros((Nobj,Nout))
Omega1 = np.zeros((Nobj,Nout))
omega1 = np.zeros((Nobj,Nout))
pomega1 = np.zeros((Nobj,Nout))
inc1 = np.zeros((Nobj,Nout))
print("going to loop")

from tqdm import tqdm

sim1.t

# for i,time in tqdm(enumerate(times)):
	# #print(i)
	# sim1.integrate(sim1.t+time)
	# #print("j loop")
	# for j in range(Nobj):
	# 	x1[j][i] = ps1[j].x
	# 	y1[j][i] = ps1[j].y
	# 	z1[j][i] = ps1[j].z
	# 	if j>0:
	# 		a1[j][i] = ps1[j].a
	# 		e1[j][i] = ps1[j].e
	# 		pomega1[j][i] = ps1[j].pomega
	# 		Omega1[j][i] = ps1[j].Omega
	# 		omega1[j][i] = ps1[j].omega
	# 		inc1[j][i] = ps1[j].inc

for i,time in tqdm(enumerate(times)):
	#print(i)
	sim1.integrate(time)
	#print("j loop")
	for j in range(Nobj):
		x1[j][i] = ps1[j].x
		y1[j][i] = ps1[j].y
		z1[j][i] = ps1[j].z
		if j>0:
			a1[j][i] = ps1[j].a
			e1[j][i] = ps1[j].e
			pomega1[j][i] = ps1[j].pomega
			Omega1[j][i] = ps1[j].Omega
			omega1[j][i] = ps1[j].omega
			inc1[j][i] = ps1[j].inc

print("loop done")
#EM1 = np.sqrt((x1[1]-x1[4])**2+(y1[1]-y1[4])**2+(z1[1]-z1[4])**2)#earth-mercury distance from GR reboundx
EM1 = np.sqrt((x1[1]-x1[3])**2+(y1[1]-y1[3])**2+(z1[1]-z1[3])**2)#earth-mercury distance from GR reboundx; considering 5 objects now
#BC1 = np.sqrt(x1[2]**2+y1[2]**2+z1[2]**2)#bepi colombo orbit adding GR effect with respect to Barycenter
#BCE1 = np.sqrt((x1[2]-x1[4])**2+(y1[2]-y1[4])**2+(z1[2]-z1[4])**2)#earth-bepi distance
BCE1 = np.sqrt((x1[2]-x1[3])**2+(y1[2]-y1[3])**2+(z1[2]-z1[3])**2)#earth-bepi distance

EM1

EM1-EM0

plt.figure()
plt.plot(times, EM1-EM0, label="Difference in Earth-Mercury distance", color='blue')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.legend()
plt.title("Earth-Mercury Distance with GR effect(Reboundx)-Newton case", fontsize='14')
plt.show()
#############################################################################################
plt.plot(times, BCE1-BCE0, label="Difference in Earth-Bepi distance", color='red')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Bepi Distance in GR(Reboundx)-Newton case", fontsize='14')
plt.show()

print("Earth-mercury distance in accordance with GR effect with all perturber effects:", EM1)
#Plotting the earth-mercury distance and the orbit of the bepi colombo to see the impact of the perturbers

plt.figure()
plt.plot(times, EM1, label="Earth-Mercury dist", color='blue')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with GR effect(Reboundx)", fontsize='14')
plt.show()





"""#######PPN when beta=gamma=1###########################

"""

def PPN(reb_sim):#refers to GR
    Nobj = Nobj2
    c = 3e5*86400 #km/day
    c2 = c**2
    cm2 = 1/c2
    G = 6.67430e-20*(86400**2) #km^3 kg^-1 days^-2
    gam = 1.
    bet = 1.
    r_vect = np.zeros((Nobj,Nobj,3))
    r2 = np.zeros((Nobj,Nobj))
    r = np.zeros((Nobj,Nobj))
    newt = np.zeros((Nobj,Nobj,3))
    mu = np.zeros(Nobj)
    v2 = np.zeros(Nobj)
    v_dot_v = np.zeros((Nobj,Nobj))
    r_dot_a = np.zeros((Nobj,Nobj))
    r_dot_vt = np.zeros((Nobj,Nobj))
    r_dot_va = np.zeros((Nobj,Nobj))
    for i in range(0,Nobj):
        mu[i] = G*ps[i].m
        v2[i] = ps[i].vx**2 + ps[i].vy**2 + ps[i].vz**2
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                r_vect[i,j,0] = ps[j].x-ps[i].x
                r_vect[i,j,1] = ps[j].y-ps[i].y
                r_vect[i,j,2] = ps[j].z-ps[i].z
                r_vect[j,i,0] = -r_vect[i,j,0]
                r_vect[j,i,1] = -r_vect[i,j,1]
                r_vect[j,i,2] = -r_vect[i,j,2]
                r2[i,j] = r_vect[i,j,0]**2 + r_vect[i,j,1]**2 + r_vect[i,j,2]**2
                r2[j,i] = r2[i,j]
                r[i,j] = np.sqrt(r2[i,j])
                r[j,i] = r[i,j]
                newt[i,j,0] = mu[i]*r_vect[i,j,0]/(r2[i,j]*r[i,j])
                newt[i,j,1] = mu[i]*r_vect[i,j,1]/(r2[i,j]*r[i,j])
                newt[i,j,2] = mu[i]*r_vect[i,j,2]/(r2[i,j]*r[i,j])
                v_dot_v[i,j] = ps[i].vx*ps[j].vx + ps[i].vy*ps[j].vy + ps[i].vz*ps[j].vz
                r_dot_a[i,j] = r_vect[i,j,0]*ps[i].ax + r_vect[i,j,1]*ps[i].ay + r_vect[i,j,2]*ps[i].az
                r_dot_vt[i,j] = r_vect[i,j,0]*ps[j].vx + r_vect[i,j,1]*ps[j].vy + r_vect[i,j,2]*ps[j].vz
                r_dot_va[i,j] = r_vect[i,j,0]*ps[i].vx + r_vect[i,j,1]*ps[i].vy + r_vect[i,j,2]*ps[i].vz
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                ps[j].ax += cm2* (-newt[i,j,0] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vx-ps[i].vx)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ax/r[i,j])
                ps[j].ay += cm2* (-newt[i,j,1] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vy-ps[i].vy)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ay/r[i,j])
                ps[j].az += cm2* (-newt[i,j,2] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vz-ps[i].vz)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].az/r[i,j])
                for b in range(0,Nobj):
                    if (b!=j):
                        ps[j].ax += cm2* (newt[i,j,0]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].ay += cm2* (newt[i,j,1]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].az += cm2* (newt[i,j,2]*2*(gam+bet)*mu[b]/r[j,b])
                    if (b!=i):
                        ps[j].ax += cm2* (newt[i,j,0]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].ay += cm2* (newt[i,j,1]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].az += cm2* (newt[i,j,2]*(2*bet-1)*mu[b]/r[i,b])

sim2 = setupsim2()
sim2.integrator = "whfast"
sim2.move_to_com()
ps2 = sim2.particles
ps = sim2.particles

sim2.additional_forces = PPN
sim2.force_is_velocity_dependent = 1

Nobj2 = 5
x = np.zeros((Nobj2,Nout))
y = np.zeros((Nobj2,Nout))
z = np.zeros((Nobj2,Nout))
a = np.zeros((Nobj2,Nout))
e = np.zeros((Nobj2,Nout))
Omega = np.zeros((Nobj2,Nout))
omega = np.zeros((Nobj2,Nout))
pomega = np.zeros((Nobj2,Nout))
inc = np.zeros((Nobj2,Nout))
print("loop starting")

import tqdm

## add tqdm
import tqdm
for i,time in tqdm.tqdm(enumerate(times)):
	sim2.integrate(time)
	# print("loop", i)
	for j in range(Nobj2):
		x[j][i] = ps2[j].x
		y[j][i] = ps2[j].y
		z[j][i] = ps2[j].z
		if j>0:
			a[j][i] = ps2[j].a
			e[j][i] = ps2[j].e
			pomega[j][i] = ps2[j].pomega
			Omega[j][i] = ps2[j].Omega
			omega[j][i] = ps2[j].omega
			inc[j][i] = ps2[j].inc
print("loop done")

EM2 = np.sqrt((x[1]-x[3])**2+(y[1]-y[3])**2+(z[1]-z[3])**2)#earth-mercury distance
BE2 = np.sqrt(x[3]**2+y[3]**2+z[3]**2)#earth distance from barycenter
BM2 = np.sqrt(x[1]**2+y[1]**2+z[1]**2)#mercury distance from barycenter
BCE2 = np.sqrt((x[3]-x[2])**2+(y[3]-y[2])**2+(z[3]-z[2])**2)#earth-bepi distance

#Shapiro delay in accordance with Jupiter
JE2 = np.sqrt((x[3]-x[4])**2+(y[3]-y[4])**2+(z[3]-z[4])**2)#jupiter-earth distance
JM2 = np.sqrt((x[4]-x[1])**2+(y[4]-y[1])**2+(z[4]-z[1])**2)#jupiter-mercury distance

#now we are considering the shapiro delay
def shapiro(BE,BM,EM):
	upp=BE+BM+EM
	low=BE+BM-EM
	G = 6.67430e-20*(86400**2)
	c = 3e5*86400
	m = 1.989e30
	gam = 1
	T1=np.log(upp/low)
	shc=2*(1+gam)*(G*m)/(c**3)
	delay=shc*T1#shapiro time delay
	return delay

c = 3e5*86400 #km/day
c2 = c**2
cm2 = 1/c2
G = 6.67430e-20*(86400**2) #km^3 kg^-1 days^-2
gam = 1.
bet = 1.
shap_gr = shapiro(BE2, BM2, EM2)
shap_j = shapiro(JE2, JM2, EM2)#shapiro delay due to jupiter
shap_d = c*shap_gr#shapiro time delay distance
shap_dj = c*shap_j
EM2_shap = EM2+shap_d #adding the effect of the shapiro delay to the earth-mercury distance
EM2_j = EM2+shap_dj

print("Earth-mercury distance in accordance with GR effect with all perturber effects:", EM2)
print("Earth-mercury distance in accordance with GR effect+shapiro with all perturber effects:", EM2_shap)
print("Earth-mercury distance in accordance with GR effect+shapiro(jupiter) with all perturber effects:", EM2_j)
#Plotting the earth-mercury distance to see the impact of the perturbers

plt.figure()
plt.plot(times, EM2, label="Earth-Mercury dist(GRT)", color='blue')
plt.plot(times, EM2_shap, label="Earth-Mercury dist(shapiro+GRT)", color='red')
plt.plot(times, EM2_j, label="Earth-Mercury dist(shapiro-jupiter+GRT)", color='green')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with GR effect+Shapiro", fontsize='14')
plt.show()

##############################################################################
plt.plot(times, BCE2, label="Earth-Bepi distance", color='red')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Bepi Distance with PPN=Gamma=Beta=1(No shapiro)", fontsize='14')
plt.show()

plt.figure()
plt.plot(times, EM2-EM1, label="Earth-Mercury dist", color='blue')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with PPN=Gamma=Beta=1", fontsize='14')
plt.show()
########################################################################################
plt.figure()
plt.plot(times, EM2_shap-EM2_j, label="Earth-Mercury dist", color='blue')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with PPN=Gamma=Beta=1(shapiro-shapiro-Jupiter)", fontsize='14')
plt.show()

plt.figure()
plt.plot(times, EM2-EM2_shap, label="Earth-Mercury dist(PPN fixed-PPN shapiro)", color='blue')
plt.plot(times,EM2-EM2_j, label="Jupiter_shapiro", color='red')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance ", fontsize='14')
plt.show()

"""#######PPN when beta=gamma=variable###########################"""

def PPN2(reb_sim):
    Nobj = Nobj2
    # Constants
    c = 3e5*86400
    c2 = c**2
    cm2 = 1/c2
    G = 6.67430e-20*(86400**2)
    gam=1.+1e-3
    bet=1.-2e-3
    r_vect = np.zeros((Nobj,Nobj,3))
    r2 = np.zeros((Nobj,Nobj))
    r = np.zeros((Nobj,Nobj))
    newt = np.zeros((Nobj,Nobj,3))
    mu = np.zeros(Nobj)
    v2 = np.zeros(Nobj)
    v_dot_v = np.zeros((Nobj,Nobj))
    r_dot_a = np.zeros((Nobj,Nobj))
    r_dot_vt = np.zeros((Nobj,Nobj))
    r_dot_va = np.zeros((Nobj,Nobj))
    for i in range(0,Nobj):
        mu[i] = G*ps[i].m
        v2[i] = ps[i].vx**2 + ps[i].vy**2 + ps[i].vz**2
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                r_vect[i,j,0] = ps[j].x-ps[i].x
                r_vect[i,j,1] = ps[j].y-ps[i].y
                r_vect[i,j,2] = ps[j].z-ps[i].z
                r_vect[j,i,0] = -r_vect[i,j,0]
                r_vect[j,i,1] = -r_vect[i,j,1]
                r_vect[j,i,2] = -r_vect[i,j,2]
                r2[i,j] = r_vect[i,j,0]**2 + r_vect[i,j,1]**2 + r_vect[i,j,2]**2
                r2[j,i] = r2[i,j]
                r[i,j] = np.sqrt(r2[i,j])
                r[j,i] = r[i,j]
                newt[i,j,0] = mu[i]*r_vect[i,j,0]/(r2[i,j]*r[i,j])
                newt[i,j,1] = mu[i]*r_vect[i,j,1]/(r2[i,j]*r[i,j])
                newt[i,j,2] = mu[i]*r_vect[i,j,2]/(r2[i,j]*r[i,j])
                v_dot_v[i,j] = ps[i].vx*ps[j].vx + ps[i].vy*ps[j].vy + ps[i].vz*ps[j].vz
                r_dot_a[i,j] = r_vect[i,j,0]*ps[i].ax + r_vect[i,j,1]*ps[i].ay + r_vect[i,j,2]*ps[i].az
                r_dot_vt[i,j] = r_vect[i,j,0]*ps[j].vx + r_vect[i,j,1]*ps[j].vy + r_vect[i,j,2]*ps[j].vz
                r_dot_va[i,j] = r_vect[i,j,0]*ps[i].vx + r_vect[i,j,1]*ps[i].vy + r_vect[i,j,2]*ps[i].vz
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                ps[j].ax += cm2* (-newt[i,j,0] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vx-ps[i].vx)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ax/r[i,j])
                ps[j].ay += cm2* (-newt[i,j,1] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vy-ps[i].vy)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ay/r[i,j])
                ps[j].az += cm2* (-newt[i,j,2] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vz-ps[i].vz)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].az/r[i,j])
                for b in range(0,Nobj):
                    if (b!=j):
                        ps[j].ax += cm2* (newt[i,j,0]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].ay += cm2* (newt[i,j,1]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].az += cm2* (newt[i,j,2]*2*(gam+bet)*mu[b]/r[j,b])
                    if (b!=i):
                        ps[j].ax += cm2* (newt[i,j,0]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].ay += cm2* (newt[i,j,1]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].az += cm2* (newt[i,j,2]*(2*bet-1)*mu[b]/r[i,b])

sim3 = setupsim2()
sim3.integrator = "whfast"
sim3.move_to_com()
ps3 = sim3.particles
ps = sim3.particles

gam=1.+1e-3
bet=1.-2e-3

sim3.additional_forces = PPN2
sim3.force_is_velocity_dependent = 1

Nobj = Nobj2
x3 = np.zeros((Nobj,Nout))
y3 = np.zeros((Nobj,Nout))
z3 = np.zeros((Nobj,Nout))
a3 = np.zeros((Nobj,Nout))
e3 = np.zeros((Nobj,Nout))
Omega3 = np.zeros((Nobj,Nout))
omega3 = np.zeros((Nobj,Nout))
pomega3 = np.zeros((Nobj,Nout))
inc3 = np.zeros((Nobj,Nout))
from tqdm import tqdm
for i,time in tqdm(enumerate(times)):
    sim3.integrate(time)
    for j in range(Nobj):
        x3[j][i] = ps3[j].x
        y3[j][i] = ps3[j].y
        z3[j][i] = ps3[j].z
        if j>0:
            a3[j][i] = ps3[j].a
            e3[j][i] = ps3[j].e
            pomega3[j][i] = ps3[j].pomega
            Omega3[j][i] = ps3[j].Omega
            omega3[j][i] = ps3[j].omega
            inc3[j][i] = ps3[j].inc

EM3 = np.sqrt((x3[1]-x3[3])**2+(y3[1]-y3[3])**2+(z3[1]-z3[3])**2)#earth-mercury distance
BE3 = np.sqrt(x3[3]**2+y3[3]**2+z3[3]**2)#earth distance from barycenter
BM3 = np.sqrt(x3[1]**2+y3[1]**2+z3[1]**2)#mercury distance from barycenter
BCE3 = np.sqrt((x3[3]-x3[2])**2+(y3[3]-y3[2])**2+(z3[3]-z3[2])**2)#earth-bepi distance

#Shapiro delay in accordance with Jupiter
JE3 = np.sqrt((x3[4]-x3[3])**2+(y3[4]-y3[3])**2+(z3[4]-z3[3])**2)#jupiter-earth distance
JM3 = np.sqrt((x3[4]-x3[1])**2+(y3[4]-y3[1])**2+(z3[4]-z3[1])**2)#jupiter-mercury distance

#now we are considering the shapiro delay
def shapiro(BE,BM,EM):
    upp=BE+BM+EM
    low=BE+BM-EM
    G = 6.67430e-20
    c = 3e5
    m = 1.989e30
    gam=1.+1e-3
    T1=np.log(upp/low)
    shc=2*(1+gam)*(G*m)/(c**3)
    delay=shc*T1#shapiro time delay
    return delay

shap_gr3 = shapiro(BE3, BM3, EM3)
shap_j3 = shapiro(JE3, JM3, EM3)
shap_d3 = c*shap_gr3#shapiro time delay distance
shap_dj3 = c*shap_j3
EM3_shap = EM3+shap_d3 #adding the effect of the shapiro delay to the earth-mercury distance
EM3_j = EM3+shap_dj3



print("Earth-mercury distance in accordance with GR effect with all perturber effects:", EM3)
print("Earth-mercury distance in accordance with GR effect+shapiro with all perturber effects:", EM3_shap)
print("Earth-mercury distance in accordance with GR effect+shapiro-jupiter with all perturber effects:", EM3_j)

#Plotting the earth-mercury distance to see the impact of the perturbers
plt.figure()
plt.plot(times, EM3, label="Earth-Mercury dist(GRT-PPN)", color='blue')
plt.plot(times, EM3_shap, label="Earth-Mercury dist(shapiro+GRT-PPN)", color='red')
plt.plot(times, EM3_j, label="Earth-Mercury dist(shapiro-jupiter+GRT-PPN)", color='green')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with GR(PPN) effect+Shapiro", fontsize='14')
plt.show()

"""############do the comparison plot same as previous and plot for bepi-earth distance too##########

#############################################################################################
###################SUN J2 Effect added to newton case###################
"""

import reboundx
print("Sun J2 effect adding to the system")
sim4 = setupsim2()
sim4.integrator = "whfast"
sim4.move_to_com()
ps4 = sim4.particles

rebx1 = reboundx.Extras(sim4)
gh = rebx1.load_force("gravitational_harmonics")
rebx1.add_force(gh)
ps4["sun"].params["J2"] = 2.2e-7
ps4["sun"].params["R_eq"] = 695700

print("still running at J2")
Nobj = Nobj2
x4 = np.zeros((Nobj,Nout))
y4 = np.zeros((Nobj,Nout))
z4 = np.zeros((Nobj,Nout))
a4 = np.zeros((Nobj,Nout))
e4 = np.zeros((Nobj,Nout))
Omega4 = np.zeros((Nobj,Nout))
omega4 = np.zeros((Nobj,Nout))
pomega4 = np.zeros((Nobj,Nout))
inc4 = np.zeros((Nobj,Nout))
print("going to loop")
for i,time in tqdm(enumerate(times)):
	#print(i)
	sim4.integrate(time)
	#print("j loop")
	for j in range(Nobj):
		x4[j][i] = ps4[j].x
		y4[j][i] = ps4[j].y
		z4[j][i] = ps4[j].z
		if j>0:
			a4[j][i] = ps4[j].a
			e4[j][i] = ps4[j].e
			pomega4[j][i] = ps4[j].pomega
			Omega4[j][i] = ps4[j].Omega
			omega4[j][i] = ps4[j].omega
			inc4[j][i] = ps4[j].inc



print("loop done")

EM4 = np.sqrt((x4[1]-x4[3])**2+(y4[1]-y4[3])**2+(z4[1]-z4[3])**2)#earth-mercury distance from Sun J2
BC4 = np.sqrt(x4[2]**2+y4[2]**2+z4[2]**2)#bepi colombo orbit adding GR effect with respect to Barycenter
BCE4 = np.sqrt((x4[3]-x4[2])**2+(y4[3]-y4[2])**2+(z4[3]-z4[2])**2)#earth-bepi distance

print("Earth-mercury distance in accordance with Sun J2 effect with all perturber effects:", EM4)
print("Earth-Bepi distance in accordance with Sun J2 effect with all perturber effects:", BCE4)

plt.figure()
plt.plot(times, EM4, label="Earth-Mercury dist", color='blue')
plt.xlabel("Time(in days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with Sun J2 effect(Reboundx)", fontsize='14')
plt.show()
#############################################################################################
plt.figure()
plt.plot(times, BCE4, label="Earth-Bepi distance", color='red')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Bepi Distance with Sun J2 effect(Reboundx)", fontsize='14')
plt.show()

"""#################################################################################################
############SUN J2 to PPN(BETA=GAMMA=1)#####################
"""

def PPN_J(reb_sim, particles, force, dt):#refers to GR
    c = 3e5*86400 #km/day
    Nobj = Nobj2
    c2 = c**2
    cm2 = 1/c2
    G = 6.67430e-20*(86400**2) #km^3 kg^-1 days^-2
    gam = 1.
    bet = 1.
    r_vect = np.zeros((Nobj,Nobj,3))
    r2 = np.zeros((Nobj,Nobj))
    r = np.zeros((Nobj,Nobj))
    newt = np.zeros((Nobj,Nobj,3))
    mu = np.zeros(Nobj)
    v2 = np.zeros(Nobj)
    v_dot_v = np.zeros((Nobj,Nobj))
    r_dot_a = np.zeros((Nobj,Nobj))
    r_dot_vt = np.zeros((Nobj,Nobj))
    r_dot_va = np.zeros((Nobj,Nobj))
    for i in range(0,Nobj):
        mu[i] = G*ps[i].m
        v2[i] = ps[i].vx**2 + ps[i].vy**2 + ps[i].vz**2
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                r_vect[i,j,0] = ps[j].x-ps[i].x
                r_vect[i,j,1] = ps[j].y-ps[i].y
                r_vect[i,j,2] = ps[j].z-ps[i].z
                r_vect[j,i,0] = -r_vect[i,j,0]
                r_vect[j,i,1] = -r_vect[i,j,1]
                r_vect[j,i,2] = -r_vect[i,j,2]
                r2[i,j] = r_vect[i,j,0]**2 + r_vect[i,j,1]**2 + r_vect[i,j,2]**2
                r2[j,i] = r2[i,j]
                r[i,j] = np.sqrt(r2[i,j])
                r[j,i] = r[i,j]
                newt[i,j,0] = mu[i]*r_vect[i,j,0]/(r2[i,j]*r[i,j])
                newt[i,j,1] = mu[i]*r_vect[i,j,1]/(r2[i,j]*r[i,j])
                newt[i,j,2] = mu[i]*r_vect[i,j,2]/(r2[i,j]*r[i,j])
                v_dot_v[i,j] = ps[i].vx*ps[j].vx + ps[i].vy*ps[j].vy + ps[i].vz*ps[j].vz
                r_dot_a[i,j] = r_vect[i,j,0]*ps[i].ax + r_vect[i,j,1]*ps[i].ay + r_vect[i,j,2]*ps[i].az
                r_dot_vt[i,j] = r_vect[i,j,0]*ps[j].vx + r_vect[i,j,1]*ps[j].vy + r_vect[i,j,2]*ps[j].vz
                r_dot_va[i,j] = r_vect[i,j,0]*ps[i].vx + r_vect[i,j,1]*ps[i].vy + r_vect[i,j,2]*ps[i].vz
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                ps[j].ax += cm2* (-newt[i,j,0] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vx-ps[i].vx)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ax/r[i,j])
                ps[j].ay += cm2* (-newt[i,j,1] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vy-ps[i].vy)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ay/r[i,j])
                ps[j].az += cm2* (-newt[i,j,2] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vz-ps[i].vz)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].az/r[i,j])
                for b in range(0,Nobj):
                    if (b!=j):
                        ps[j].ax += cm2* (newt[i,j,0]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].ay += cm2* (newt[i,j,1]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].az += cm2* (newt[i,j,2]*2*(gam+bet)*mu[b]/r[j,b])
                    if (b!=i):
                        ps[j].ax += cm2* (newt[i,j,0]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].ay += cm2* (newt[i,j,1]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].az += cm2* (newt[i,j,2]*(2*bet-1)*mu[b]/r[i,b])

sim5 = setupsim2()
sim5.integrator = "whfast"
sim5.move_to_com()
ps5 = sim5.particles
ps = sim5.particles

import reboundx
rebx2 = reboundx.Extras(sim5)

ppn_force = rebx2.create_force("ppn_force")
ppn_force.force_type = "vel"  # Since it's velocity dependent
ppn_force.update_accelerations = PPN_J
rebx2.add_force(ppn_force)

# Set velocity dependence flag
sim5.force_is_velocity_dependent = 1

gh1 = rebx2.load_force("gravitational_harmonics")
rebx2.add_force(gh1)
ps5["sun"].params["J2"] = 2.2e-7
ps5["sun"].params["R_eq"] = 695700

Nobj = Nobj2
x5 = np.zeros((Nobj,Nout))
y5 = np.zeros((Nobj,Nout))
z5 = np.zeros((Nobj,Nout))
a5 = np.zeros((Nobj,Nout))
e5 = np.zeros((Nobj,Nout))
Omega5 = np.zeros((Nobj,Nout))
omega5 = np.zeros((Nobj,Nout))
pomega5 = np.zeros((Nobj,Nout))
inc5 = np.zeros((Nobj,Nout))
print("loop starting")
for i,time in tqdm(enumerate(times)):
	sim5.integrate(time)
	print("loop", i)
	for j in range(Nobj):
		x5[j][i] = ps5[j].x
		y5[j][i] = ps5[j].y
		z5[j][i] = ps5[j].z
		if j>0:
			a5[j][i] = ps5[j].a
			e5[j][i] = ps5[j].e
			pomega5[j][i] = ps5[j].pomega
			Omega5[j][i] = ps5[j].Omega
			omega5[j][i] = ps5[j].omega
			inc5[j][i] = ps5[j].inc
print("loop done")

EM5 = np.sqrt((x5[1]-x5[3])**2+(y5[1]-y5[3])**2+(z5[1]-z5[3])**2)#earth-mercury distance
BE5 = np.sqrt(x5[3]**2+y5[3]**2+z5[3]**2)#earth distance from barycenter
BM5 = np.sqrt(x5[1]**2+y5[1]**2+z5[1]**2)#mercury distance from barycenter
BCE5 = np.sqrt((x5[3]-x5[2])**2+(y5[3]-y5[2])**2+(z5[3]-z5[2])**2)#earth-bepi distance

#Shapiro delay in accordance with Jupiter
JE5 = np.sqrt((x5[3]-x5[4])**2+(y5[3]-y5[4])**2+(z5[3]-z5[4])**2)#jupiter-earth distance
JM5 = np.sqrt((x5[4]-x5[1])**2+(y5[4]-y5[1])**2+(z5[4]-z5[1])**2)#jupiter-mercury distance

#now we are considering the shapiro delay
def shapiro(BE,BM,EM):
	upp=BE+BM+EM
	low=BE+BM-EM
	G = 6.67430e-20*(86400**2)
	c = 3e5*86400
	m = 1.989e30
	gam=1
	T1=np.log(upp/low)
	shc=2*(1+gam)*(G*m)/(c**3)
	delay=shc*T1#shapiro time delay
	return delay

shap_gr5 = shapiro(BE5, BM5, EM5)
shap_j5 = shapiro(JE5, JM5, EM5)#shapiro delay due to jupiter
shap_d5 = c*shap_gr5#shapiro time delay distance
shap_dj5 = c*shap_j5
EM5_shap = EM5+shap_d5 #adding the effect of the shapiro delay to the earth-mercury distance
EM5_j = EM2+shap_dj5

print("Earth-mercury distance in accordance with GR effect with all perturber effects:", EM5)
print("Earth-mercury distance in accordance with GR effect+shapiro with all perturber effects:", EM5_shap)
print("Earth-mercury distance in accordance with GR effect+shapiro(jupiter) with all perturber effects:", EM5_j)

plt.figure()
plt.plot(times, EM5, label="Earth-Mercury dist(GRT)", color='blue')
plt.plot(times, EM5_shap, label="Earth-Mercury dist(shapiro+GRT)", color='red')
plt.plot(times, EM5_j, label="Earth-Mercury dist(shapiro-jupiter+GRT)", color='green')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with GR effect+Shapiro", fontsize='14')
plt.show()
###########################################################################################
plt.plot(times, BCE5, label="Earth-Bepi distance", color='red')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Bepi Distance with GR effect+Shapiro+Sun J2", fontsize='14')
plt.show()

"""##################################################################################################
############SUN J2 to PPN= GAMMA=BETA=variable#########################
"""

def PPN_J2(reb_sim, particles, force, dt):#refers to GR
    Nobj = Nobj2
    c = 3e5*86400 #km/day
    c2 = c**2
    cm2 = 1/c2
    G = 6.67430e-20*(86400**2) #km^3 kg^-1 days^-2
    gam=1.+1e-3
    bet=1.-2e-3
    r_vect = np.zeros((Nobj,Nobj,3))
    r2 = np.zeros((Nobj,Nobj))
    r = np.zeros((Nobj,Nobj))
    newt = np.zeros((Nobj,Nobj,3))
    mu = np.zeros(Nobj)
    v2 = np.zeros(Nobj)
    v_dot_v = np.zeros((Nobj,Nobj))
    r_dot_a = np.zeros((Nobj,Nobj))
    r_dot_vt = np.zeros((Nobj,Nobj))
    r_dot_va = np.zeros((Nobj,Nobj))
    for i in range(0,Nobj):
        mu[i] = G*ps[i].m
        v2[i] = ps[i].vx**2 + ps[i].vy**2 + ps[i].vz**2
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                r_vect[i,j,0] = ps[j].x-ps[i].x
                r_vect[i,j,1] = ps[j].y-ps[i].y
                r_vect[i,j,2] = ps[j].z-ps[i].z
                r_vect[j,i,0] = -r_vect[i,j,0]
                r_vect[j,i,1] = -r_vect[i,j,1]
                r_vect[j,i,2] = -r_vect[i,j,2]
                r2[i,j] = r_vect[i,j,0]**2 + r_vect[i,j,1]**2 + r_vect[i,j,2]**2
                r2[j,i] = r2[i,j]
                r[i,j] = np.sqrt(r2[i,j])
                r[j,i] = r[i,j]
                newt[i,j,0] = mu[i]*r_vect[i,j,0]/(r2[i,j]*r[i,j])
                newt[i,j,1] = mu[i]*r_vect[i,j,1]/(r2[i,j]*r[i,j])
                newt[i,j,2] = mu[i]*r_vect[i,j,2]/(r2[i,j]*r[i,j])
                v_dot_v[i,j] = ps[i].vx*ps[j].vx + ps[i].vy*ps[j].vy + ps[i].vz*ps[j].vz
                r_dot_a[i,j] = r_vect[i,j,0]*ps[i].ax + r_vect[i,j,1]*ps[i].ay + r_vect[i,j,2]*ps[i].az
                r_dot_vt[i,j] = r_vect[i,j,0]*ps[j].vx + r_vect[i,j,1]*ps[j].vy + r_vect[i,j,2]*ps[j].vz
                r_dot_va[i,j] = r_vect[i,j,0]*ps[i].vx + r_vect[i,j,1]*ps[i].vy + r_vect[i,j,2]*ps[i].vz
    for i in range(0,Nobj):
        for j in range(0,Nobj):
            if (j!=i):
                ps[j].ax += cm2* (-newt[i,j,0] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vx-ps[i].vx)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ax/r[i,j])
                ps[j].ay += cm2* (-newt[i,j,1] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vy-ps[i].vy)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].ay/r[i,j])
                ps[j].az += cm2* (-newt[i,j,2] * (gam*v2[j] + (1+gam)*v2[i] - 2*(1+gam)*v_dot_v[i,j]
                                                  - 1.5*(r_dot_va[i,j]/r[i,j])**2 - 0.5*r_dot_a[i,j])
                                  + mu[i]/(r2[i,j]*r[i,j]) * (2*(1+gam)*r_dot_vt[i,j]
                                                             - (1+2*gam)*r_dot_va[i,j])*(ps[j].vz-ps[i].vz)
                                  + 0.5*(3+4*gam)*mu[i]*ps[i].az/r[i,j])
                for b in range(0,Nobj):
                    if (b!=j):
                        ps[j].ax += cm2* (newt[i,j,0]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].ay += cm2* (newt[i,j,1]*2*(gam+bet)*mu[b]/r[j,b])
                        ps[j].az += cm2* (newt[i,j,2]*2*(gam+bet)*mu[b]/r[j,b])
                    if (b!=i):
                        ps[j].ax += cm2* (newt[i,j,0]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].ay += cm2* (newt[i,j,1]*(2*bet-1)*mu[b]/r[i,b])
                        ps[j].az += cm2* (newt[i,j,2]*(2*bet-1)*mu[b]/r[i,b])

sim6 = setupsim2()
sim6.integrator = "whfast"
sim6.move_to_com()
ps6 = sim6.particles
ps = sim6.particles

import reboundx
rebx3 = reboundx.Extras(sim6)

ppn_var_force = rebx3.create_force("ppn_var_force")
ppn_var_force.force_type = "vel"  # Since it's velocity dependent
ppn_var_force.update_accelerations = PPN_J2
rebx3.add_force(ppn_var_force)

# Set velocity dependence flag
sim6.force_is_velocity_dependent = 1

gh2 = rebx3.load_force("gravitational_harmonics")
rebx3.add_force(gh2)
ps6["sun"].params["J2"] = 2.2e-7
ps6["sun"].params["R_eq"] = 695700

gam=1.+1e-3
bet=1.-2e-3
Nobj = Nobj2
x6 = np.zeros((Nobj,Nout))
y6 = np.zeros((Nobj,Nout))
z6 = np.zeros((Nobj,Nout))
a6 = np.zeros((Nobj,Nout))
e6 = np.zeros((Nobj,Nout))
Omega6 = np.zeros((Nobj,Nout))
omega6 = np.zeros((Nobj,Nout))
pomega6 = np.zeros((Nobj,Nout))
inc6 = np.zeros((Nobj,Nout))
print("loop starting")
for i,time in tqdm(enumerate(times)):
	sim6.integrate(time)
	print("loop", i)
	for j in range(Nobj):
		x6[j][i] = ps6[j].x
		y6[j][i] = ps6[j].y
		z6[j][i] = ps6[j].z
		if j>0:
			a6[j][i] = ps6[j].a
			e6[j][i] = ps6[j].e
			pomega6[j][i] = ps6[j].pomega
			Omega6[j][i] = ps6[j].Omega
			omega6[j][i] = ps6[j].omega
			inc6[j][i] = ps6[j].inc
print("loop done")

EM6 = np.sqrt((x6[1]-x6[3])**2+(y6[1]-y6[3])**2+(z6[1]-z6[3])**2)#earth-mercury distance
BE6 = np.sqrt(x6[3]**2+y6[3]**2+z6[3]**2)#earth distance from barycenter
BM6 = np.sqrt(x6[1]**2+y6[1]**2+z6[1]**2)#mercury distance from barycenter
BCE6 = np.sqrt((x6[3]-x6[2])**2+(y6[3]-y6[2])**2+(z6[3]-z6[2])**2)#earth-bepi distance


#Shapiro delay in accordance with Jupiter
JE6 = np.sqrt((x6[3]-x6[4])**2+(y6[3]-y6[4])**2+(z6[3]-z6[4])**2)#jupiter-earth distance
JM6 = np.sqrt((x6[4]-x6[1])**2+(y6[4]-y6[1])**2+(z6[4]-z6[1])**2)#jupiter-mercury distance

#now we are considering the shapiro delay
def shapiro(BE,BM,EM):
	upp=BE+BM+EM
	low=BE+BM-EM
	G = 6.67430e-20*(86400**2)
	c = 3e5*86400
	m = 1.989e30
	gam=1.+1e-3
	T1=np.log(upp/low)
	shc=2*(1+gam)*(G*m)/(c**3)
	delay=shc*T1#shapiro time delay
	return delay

shap_gr6 = shapiro(BE6, BM6, EM6)
shap_j6 = shapiro(JE6, JM6, EM6)#shapiro delay due to jupiter
shap_d6 = c*shap_gr6#shapiro time delay distance
shap_dj6 = c*shap_j6
EM6_shap = EM6+shap_d6 #adding the effect of the shapiro delay to the earth-mercury distance
EM6_j = EM6+shap_dj6

print("Earth-mercury distance in accordance with GR effect with all perturber effects:", EM6)
print("Earth-mercury distance in accordance with GR effect+shapiro with all perturber effects:", EM6_shap)
print("Earth-mercury distance in accordance with GR effect+shapiro(jupiter) with all perturber effects:", EM6_j)

plt.figure()
plt.plot(times, EM6, label="Earth-Mercury dist(GRT)", color='blue')
plt.plot(times, EM6_shap, label="Earth-Mercury dist(shapiro+GRT)", color='red')
plt.plot(times, EM6_j, label="Earth-Mercury dist(shapiro-jupiter+GRT)", color='green')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Mercury Distance with GR effect+Shapiro+Sun J2", fontsize='14')
plt.show()
#########################################################
plt.plot(times, BCE6, label="Earth-Bepi distance", color='red')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
plt.title("Earth-Bepi Distance with GR effect+Shapiro+Sun J2", fontsize='14')
plt.show()

import pickle as pkl#refer to pickle documentation

def save_data_to_pickle(filename, data):
    with open(filename, 'wb') as file:
        pkl.dump(data, file)

def load_data_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pkl.load(file)
    return data

data = {"NEWTON":{"EM":EM0, "BC":BC0, "BCE":BCE0}, "GR":{"EM":EM1, "BC":BC1, "BCE":BCE1},"PPN_BETA_GAMMA_1":{"EM_shap":EM2_shap, "EM":EM2, "BCE":BCE2, "EM_J":EM2_j, "BE":BE2,"BM":BM2,"JE":JE2,"JM":JM2},
 "PPN_BETA_GAMMA_VAR":{"EM_shap":EM3_shap, "EM":EM3, "BCE":BCE3, "EM_J":EM3_j, "BE":BE3,"BM":BM3,"JE":JE3,"JM":JM3},
 "J2_NEWTON":{"EM":EM4,"BC":BC4,"BCE":BCE4},"PPN_J2_GAMMA_BETA_1":{"EM":EM5,"BE":BE5,"BM":BM5,"BCE":BCE5,"JE":JE5,"JM":JM5,"EM_shap":EM5_shap, "EM_J":EM5_j},
 "PPN_J2_BETA_GAMMA_VAR":{"EM":EM6,"BE":BE6,"BM":BM6,"BCE":BCE6,"JE":JE6,"JM":JM6,"EM_shap":EM6_shap,"EM_J":EM6_j}}



data.keys()

save_data_to_pickle("phase_data.pkl",data)



type(data)

data.keys()

x = data["NEWTON"]
x.keys()

data["PPN_BETA_GAMMA_1"]["EM"]

data = load_data_from_pickle("phase_data.pkl")

data.keys()

'''
To use
 em_newton = data['NEWTON']['EM']
       em_gr     = data['GR']['EM']
       em_ppn1   = data['PPN_BETA_GAMMA_1']['EM']
       em_ppn2   = data['PPN_BETA_GAMMA_VAR']['EM']


'''
em_newton = data['NEWTON']['EM']
em_gr     = data['GR']['EM']
em_ppn1   = data['PPN_BETA_GAMMA_1']['EM']
bce_ppn1 = data['PPN_BETA_GAMMA_1']['BCE']
em_ppn2   = data['PPN_BETA_GAMMA_VAR']['EM']
bce_ppn2 = data['PPN_BETA_GAMMA_VAR']['BCE']

#comparing earth-mercury distance(primarily) between PPN=gamma=beta=1 and GR from reboundx with 5 objects
em_ppn1 = data['PPN_BETA_GAMMA_1']['EM']

plt.figure()
plt.plot(times, em_ppn1-EM1, label="PPN_1-GR reboundx(no shapiro)", color='blue')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Difference in Earth-Mercury distance(PPN_1-GR reboundx(no shapiro))", fontsize='14')
#plt.show()
#comparing earth-mercury distance(primarily) between PPN=gamma=beta=variable and GR from reboundx with 5 objects
plt.figure()
plt.plot(times, em_ppn2-EM1, label="PPN_var-GR reboundx(no shapiro)", color='blue')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Difference in Earth-Mercury distance(PPN_variable-GR reboundx(no shapiro))", fontsize='14')
plt.show()

plt.figure()
plt.plot(times, bce_ppn1-BCE1, label="PPN_1-GR reboundx(no shapiro)", color='blue')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Difference in Earth-Bepi distance(PPN_1-GR reboundx(no shapiro))", fontsize='14')
#plt.show()
##########################################################
plt.figure()
plt.plot(times, bce_ppn2-BCE1, label="PPN_var-GR reboundx(no shapiro)", color='blue')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Difference in Earth-Bepi distance(PPN_variable-GR reboundx(no shapiro))", fontsize='14')
plt.show()

"""data = {"NEWTON":{"EM":EM0, "BC":BC0, "BCE":BCE0},



"GR":{"EM":EM1, "BC":BC1, "BCE":BCE1},



"PPN_BETA_GAMMA_1":{"EM_shap":EM2_shap, "EM":EM2, "BCE":BCE2, "EM_J":EM2_j, "BE":BE2,"BM":BM2,"JE":JE2,"JM":JM2},



"PPN_BETA_GAMMA_VAR":{"EM_shap":EM3_shap, "EM":EM3, "BCE":BCE3, "EM_J":EM3_j, "BE":BE3,"BM":BM3,"JE":JE3,"JM":JM3},




"J2_NEWTON":{"EM":EM4,"BC":BC4,"BCE":BCE4},




"PPN_J2_GAMMA_BETA_1":{"EM":EM5,"BE":BE5,"BM":BM5,"BCE":BCE5,"JE":JE5,"JM":JM5,"EM_shap":EM5_shap, "EM_J":EM5_j},




"PPN_J2_BETA_GAMMA_VAR":{"EM":EM6,"BE":BE6,"BM":BM6,"BCE":BCE6,"JE":JE6,"JM":JM6,"EM_shap":EM6_shap,"EM_J":EM6_j}}
"""

#computing the earth-mercury distances including shapiro delay for sun and shapiro delay for jupiter in PPN=gamma=beta=1
print("Earth-mercury distance from PPN=Gamma=Beta=1(shapiro delay for sun): ", data['PPN_BETA_GAMMA_1']['EM_shap'])
print("Earth-mercury distance from PPN=Gamma=Beta=1(shapiro delay for Jupiter): ", data['PPN_BETA_GAMMA_1']['EM_J'])
#computing the earth-mercury distances including shapiro delay for sun and shapiro delay for jupiter in PPN=gamma=beta=var
print("Earth-mercury distance from PPN=Gamma=Beta=variable(shapiro delay for sun): ", data['PPN_BETA_GAMMA_VAR']['EM_shap'])
print("Earth-mercury distance from PPN=Gamma=Beta=variable(shapiro delay for Jupiter): ", data['PPN_BETA_GAMMA_VAR']['EM_J'])

1.01965943e+08-1.01965943e+08

#comparing earth-mercury distance(primarily) from PPN=gamma=beta=1 between shapiro delay for sun and shapiro delay for jupiter
plt.figure()
plt.plot(times,  data['PPN_BETA_GAMMA_1']['EM_shap']-data['PPN_BETA_GAMMA_1']['EM_J'], label="PPN_1(shapiro delay for sun-jupiter)", color='blue')
#plt.plot(times,  data['PPN_BETA_GAMMA_1']['EM_J'], label="PPN_1(shapiro delay for Jupiter)", color='red')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Comparing earth-mercury distance(PPN_1) including (shapiro delay for sun-shapiro delay for jupiter)", fontsize='14')
#plt.show()
#comparing earth-mercury distance(primarily) from PPN=gamma=beta=var between shapiro delay for sun and shapiro delay for jupiter
plt.figure()
plt.plot(times,  data['PPN_BETA_GAMMA_VAR']['EM_shap']-data['PPN_BETA_GAMMA_VAR']['EM_J'], label="PPN_1(shapiro delay for sun-jupiter)", color='blue')
#plt.plot(times,  data['PPN_BETA_GAMMA_1']['EM_J'], label="PPN_1(shapiro delay for Jupiter)", color='red')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Comparing earth-mercury distance(PPN_VAR) including (shapiro delay for sun-shapiro delay for jupiter)", fontsize='14')
#plt.show()

#comparing earth-mercury distance(primarily) between PPN=gamma=beta=var and PPN=gamma=beta=1 for shapiro delay for sun
plt.figure()
plt.plot(times,  data['PPN_BETA_GAMMA_VAR']['EM_shap']-data['PPN_BETA_GAMMA_1']['EM_shap'], color='blue')
#plt.plot(times,  data['PPN_BETA_GAMMA_1']['EM_J'], label="PPN_1(shapiro delay for Jupiter)", color='red')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Comparing earth-mercury distance between PPN=gamma=beta=var and PPN=gamma=beta=1 for shapiro delay for sun ", fontsize='14')
plt.show()

#comparing earth-mercury distance between PPN_1+shapiro delay+sun J2 and PPN_var+shapiro delay+sun J2
plt.figure()
plt.plot(times,  data['PPN_J2_BETA_GAMMA_VAR']['EM_shap']-data['PPN_J2_GAMMA_BETA_1']['EM_shap'], color='blue')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Comparing earth-mercury distance between PPN_1+J2+shapiro and PPN_Var+J2+shapiro", fontsize='14')
#plt.show()
#comparing earth-bepi distance between PPN_1+sun J2 and PPN_var+sun J2
plt.figure()
plt.plot(times,  data['PPN_J2_BETA_GAMMA_VAR']['BCE']-data['PPN_J2_GAMMA_BETA_1']['BCE'], color='blue')
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Comparing earth-bepi distance between PPN_1+J2 and PPN_Var+J2", fontsize='14')
plt.show()

plt.figure()
plt.plot(times,  data['PPN_J2_GAMMA_BETA_1']['EM_shap'], color='blue')#did for ppn_var too
plt.xlabel("Time(in Days)")
plt.ylabel("Distance(km)")
#plt.legend()
plt.title("Earth-mercury distance in PPN_1+J2+shapiro", fontsize='14')
plt.show()

